import numpy as np
import tensorflow as tf

from ac.common.prints import console_print
from ai.nas.core.controller import Controller
from ai.nas.everyone.micro.train_ops import TrainOps
from ai.nas.params.controller import ControllerParams
from ai.libs.tf.ops.seqs import stack_lstm
from ai.libs.tf.utils import print_tf_vars, get_trainable_vars


class MicroController(Controller):

    def __init__(self, params: ControllerParams):
        super(self.__class__, self).__init__(params)

    def create_params(self, num_branches, lstm_size, num_lstm_layers):
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(self.name, initializer=initializer):
            with tf.variable_scope("lstm"):
                for layer_id in range(num_lstm_layers):
                    with tf.variable_scope("layer_{}".format(layer_id)):
                        w = tf.get_variable("w", [2 * lstm_size, 4 * lstm_size])
                        self.w_lstm.append(w)

            self.g_emb = tf.get_variable("g_emb", [1, lstm_size])
            with tf.variable_scope("emb"):
                self.w_emb = tf.get_variable("w", [num_branches, lstm_size])
            with tf.variable_scope("softmax"):
                self.w_soft = tf.get_variable("w", [lstm_size, num_branches])
                b_init = np.array([10.0, 10.0] + [0] * (num_branches - 2), dtype=np.float32)
                self.b_soft = tf.get_variable("b", [1, num_branches], initializer=tf.constant_initializer(b_init))
                b_soft_no_learn = np.array([0.25, 0.25] + [-0.25] * (num_branches - 2), dtype=np.float32)
                b_soft_no_learn = np.reshape(b_soft_no_learn, [1, num_branches])
                self.b_soft_no_learn = tf.constant(b_soft_no_learn, dtype=tf.float32)

            with tf.variable_scope("attention"):
                self.w_attn_1 = tf.get_variable("w_1", [lstm_size, lstm_size])
                self.w_attn_2 = tf.get_variable("w_2", [lstm_size, lstm_size])
                self.v_attn = tf.get_variable("v", [lstm_size, 1])

    def build_arc_samplers(self):
        lstm_size = self.params.lstm_size
        num_layers = self.params.lstm_num_layers
        num_branches = self.params.num_branches

        self.create_params(num_branches, lstm_size, num_layers)
        arc_conv, entropy_conv, log_prob_conv, c, h = self.make_sampler(lstm_size, num_layers, use_bias=True)
        arc_pool, entropy_pool, log_prob_pool, _, _ = self.make_sampler(lstm_size, num_layers, prev_c=c, prev_h=h)

        self.sample_arc = (arc_conv, arc_pool)
        self.sample_entropy = entropy_conv + entropy_pool
        self.sample_log_prob = log_prob_conv + log_prob_pool

    def make_sampler(self, lstm_size, num_lstm_layers, prev_c=None, prev_h=None, use_bias=False):
        console_print("Build sampler in ctrl")

        num_cells = self.params.num_cells
        temperature = self.params.temperature
        tanh_constant = self.params.tanh_constant
        op_tanh_reduce = self.params.op_tanh_reduce

        arc_seq = tf.TensorArray(tf.int32, size=num_cells * 4)
        anchors = tf.TensorArray(tf.float32, size=num_cells + 2, clear_after_read=False)
        anchors_w_1 = tf.TensorArray(tf.float32, size=num_cells + 2, clear_after_read=False)

        if prev_c is None:
            assert prev_h is None, "prev_c and prev_h must both be None"
            prev_c = [tf.zeros([1, lstm_size], tf.float32) for _ in range(num_lstm_layers)]
            prev_h = [tf.zeros([1, lstm_size], tf.float32) for _ in range(num_lstm_layers)]
        inputs = self.g_emb

        for layer_id in range(2):
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            anchors = anchors.write(layer_id, tf.zeros_like(next_h[-1]))
            anchors_w_1 = anchors_w_1.write(layer_id, tf.matmul(next_h[-1], self.w_attn_1))
        print("Done")

        def condition_fn(layer_id, *args):
            return tf.less(layer_id, num_cells + 2)

        def sample_gen_fn(layer_id, inputs, prev_c, prev_h, anchors, anchors_w_1, arc_seq, entropy, log_prob):
            sm_crx = tf.nn.softmax_cross_entropy_with_logits_v2
            sparse_sm_crx = tf.nn.sparse_softmax_cross_entropy_with_logits

            indices = tf.range(0, layer_id, dtype=tf.int32)
            start_id = 4 * (layer_id - 2)
            prev_layers = []
            for i in range(2):  # index_1, index_2
                next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h
                query = anchors_w_1.gather(indices)
                query = tf.reshape(query, [layer_id, lstm_size])
                query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
                query = tf.matmul(query, self.v_attn)
                logits = tf.reshape(query, [1, layer_id])
                if temperature is not None:
                    logits /= temperature
                if tanh_constant is not None:
                    logits = tanh_constant * tf.tanh(logits)
                index = tf.multinomial(logits, 1)
                index = tf.to_int32(index)
                index = tf.reshape(index, [1])
                arc_seq = arc_seq.write(start_id + 2 * i, index)
                curr_log_prob = sparse_sm_crx(logits=logits, labels=index)
                log_prob += curr_log_prob
                curr_ent = tf.stop_gradient(sm_crx(logits=logits, labels=tf.nn.softmax(logits)))
                entropy += curr_ent
                prev_layers.append(anchors.load(tf.reduce_sum(index)))
                inputs = prev_layers[-1]

            for i in range(2):  # op_1, op_2
                next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h
                logits = tf.matmul(next_h[-1], self.w_soft) + self.b_soft
                if temperature is not None:
                    logits /= temperature
                if tanh_constant is not None:
                    op_tanh = tanh_constant / op_tanh_reduce
                    logits = op_tanh * tf.tanh(logits)
                if use_bias:
                    logits += self.b_soft_no_learn
                op_id = tf.multinomial(logits, 1)
                op_id = tf.to_int32(op_id)
                op_id = tf.reshape(op_id, [1])
                arc_seq = arc_seq.write(start_id + 2 * i + 1, op_id)
                curr_log_prob = sparse_sm_crx(logits=logits, labels=op_id)
                log_prob += curr_log_prob
                curr_ent = tf.stop_gradient(sm_crx(logits=logits, labels=tf.nn.softmax(logits)))
                entropy += curr_ent
                inputs = tf.nn.embedding_lookup(self.w_emb, op_id)

            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            anchors = anchors.write(layer_id, next_h[-1])
            anchors_w_1 = anchors_w_1.write(layer_id, tf.matmul(next_h[-1], self.w_attn_1))
            inputs = self.g_emb

            return layer_id + 1, inputs, next_c, next_h, anchors, anchors_w_1, arc_seq, entropy, log_prob

        loop_vars = [
            tf.constant(2, dtype=tf.int32, name="layer_id"),
            inputs,
            prev_c,
            prev_h,
            anchors,
            anchors_w_1,
            arc_seq,
            tf.constant([0.0], dtype=tf.float32, name="entropy"),
            tf.constant([0.0], dtype=tf.float32, name="log_prob"),
        ]

        loop_outputs = tf.while_loop(condition_fn, sample_gen_fn, loop_vars, parallel_iterations=1)

        arc_seq = loop_outputs[-3].stack()
        arc_seq = tf.reshape(arc_seq, [-1])
        entropy = tf.reduce_sum(loop_outputs[-2])
        log_prob = tf.reduce_sum(loop_outputs[-1])

        last_c = loop_outputs[-7]
        last_h = loop_outputs[-6]

        return arc_seq, entropy, log_prob, last_c, last_h

    def create_trainer(self, child_model):
        entropy_weight = self.params.entropy_weight
        baseline_decay = self.params.baseline_decay

        child_model.build_valid_rl()
        reward, rl_loss = child_model.get_rl_reward()

        if entropy_weight is not None:
            reward += entropy_weight * self.sample_entropy

        baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        baseline_update = tf.assign_sub(baseline, (1 - baseline_decay) * (baseline - reward))
        with tf.control_dependencies([baseline_update]):
            reward = tf.identity(reward)

        self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
        train_loss = self.sample_log_prob * (reward - baseline)

        tf_variables = get_trainable_vars(self.name, False)
        print_tf_vars(tf_variables)

        train_op_params = self.params.get_for_train_op()
        train_op_params.pretty_print()

        train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")
        train_ops = TrainOps(train_loss, tf_variables, train_step, train_op_params)
        skip_rate = tf.constant(0.0, dtype=tf.float32)  # Micro 에서는 사용하지 않는 옵션 Macro 호환용

        trainer_outs = {
            "lr": train_ops.learning_rate,
            "train_op": train_ops.train_op,
            "optimizer": train_ops.optimizer,
            "grad_norm": train_ops.grad_norm,
            "reward": reward,
            "rl_loss": rl_loss,
            "train_loss": train_loss,
            "train_step": train_step,
            "baseline": baseline,
            "skip_rate": skip_rate,
        }

        return trainer_outs
