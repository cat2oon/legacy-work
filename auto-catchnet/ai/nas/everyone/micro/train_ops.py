import tensorflow as tf

from ai.nas.params.train_ops import TrainOpParams

"""
TODO: https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation
Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
"""


class TrainOps:
    def __init__(self, loss, tf_vars, train_step, params: TrainOpParams):
        self.loss = loss
        self.params = params
        self.tf_vars = tf_vars
        self.train_step = train_step

        """ trainer outs """
        self.train_op = None
        self.learning_rate = None
        self.optimizer = None
        self.grad_norm = None
        self.grad_norms = None

        self.build()

    def build(self):
        tf_vars = self.tf_vars
        loss = self.apply_l2_reg(self.loss)

        grads = tf.gradients(loss, tf_vars)
        grad_norm, grad_norms = self.compute_grad_norms(grads)
        grads = self.apply_grad_clip(grads)

        if self.use_lr_annealing():
            learning_rate = self.annealing_learning_rate()
        else:
            learning_rate = self.decaying_learning_rate()

        if self.use_lr_warming_up():
            learning_rate = self.warming_up_learning_rate(learning_rate)

        optimizer = self.get_optimizer(learning_rate)
        if self.use_moving_average():
            optimizer = self.apply_moving_average(optimizer)

        train_op = optimizer.apply_gradients(zip(grads, tf_vars), global_step=self.train_step)

        self.train_op = train_op
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.grad_norm = grad_norm
        self.grad_norms = grad_norms

    """ methods """

    def warming_up_learning_rate(self, learning_rate):
        lr_warm_up_val = self.params.lr_warmup_val
        lr_warm_up_steps = self.params.lr_warmup_steps
        learning_rate = tf.cond(tf.less(self.train_step, lr_warm_up_steps),
                                lambda: lr_warm_up_val,
                                lambda: learning_rate)
        return learning_rate

    def get_optimizer(self, learning_rate):
        optimizer_name = self.params.optimizer_name
        if optimizer_name == "momentum":
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_locking=True, use_nesterov=True)
        elif optimizer_name == "sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True)
        elif optimizer_name == "adam":
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.0, epsilon=1e-3, use_locking=True)
        else:
            raise ValueError("Unknown optimizer name {}".format(optimizer_name))

        if self.use_sync_replicas():
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=self.params.num_aggregate,
                                                 total_num_replicas=self.params.num_replicas, use_locking=True)
        return opt

    def annealing_learning_rate(self):
        lr_max = self.params.lr_max
        lr_min = self.params.lr_min
        lr_T_0 = self.params.lr_T_0
        lr_T_mul = self.params.lr_T_mul
        num_train_batches = self.params.num_train_batches

        curr_epoch = self.train_step // num_train_batches
        last_reset = tf.Variable(0, dtype=tf.int32, trainable=False, name="last_reset")
        T_i = tf.Variable(lr_T_0, dtype=tf.int32, trainable=False, name="T_i")
        T_curr = curr_epoch - last_reset

        def _update():
            update_last_reset = tf.assign(last_reset, curr_epoch, use_locking=True)
            update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)
            with tf.control_dependencies([update_last_reset, update_T_i]):
                rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
                lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr

        def _no_update():
            rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
            return lr

        learning_rate = tf.cond(tf.greater_equal(T_curr, T_i), _update, _no_update)
        return learning_rate

    def decaying_learning_rate(self):
        params = self.params
        lr_init = params.lr_init
        lr_dec_min = params.lr_dec_min
        lr_dec_rate = params.lr_dec_rate
        lr_dec_start = params.lr_dec_start
        lr_dec_every_batches = params.lr_dec_every_batches

        learning_rate = tf.train.exponential_decay(lr_init, tf.maximum(self.train_step - lr_dec_start, 0),
                                                   lr_dec_every_batches, lr_dec_rate, staircase=True)
        if lr_dec_min is not None:
            learning_rate = tf.maximum(learning_rate, lr_dec_min)
        return learning_rate

    def apply_moving_average(self, opt):
        opt = tf.contrib.opt.MovingAverageOptimizer(opt, average_decay=self.params.moving_average)
        return opt

    def apply_grad_clip(self, grads):
        clip_mode = self.params.clip_mode
        grad_bound = self.params.grad_bound

        if clip_mode is None:
            return grads

        assert grad_bound is not None, "Need grad_bound to clip gradients."
        if clip_mode == "global":
            grads, _ = tf.clip_by_global_norm(grads, grad_bound)
        elif clip_mode == "norm":
            clipped = []
            for g in grads:
                if isinstance(g, tf.IndexedSlices):
                    c_g = tf.clip_by_norm(g.values, grad_bound)
                    c_g = tf.IndexedSlices(g.indices, c_g)
                else:
                    c_g = tf.clip_by_norm(g, grad_bound)
                clipped.append(g)
            grads = clipped
        else:
            raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))
        return grads

    def compute_grad_norms(self, grads):
        grad_norms = {}
        for v, g in zip(self.tf_vars, grads):
            if v is None or g is None:
                continue
            if isinstance(g, tf.IndexedSlices):
                grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
            else:
                grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))
        grad_norm = tf.global_norm(grads)
        return grad_norm, grad_norms

    def apply_l2_reg(self, loss):
        if self.params.l2_reg > 0:
            l2_losses = []
            for var in self.tf_vars:
                l2_losses.append(tf.reduce_sum(var ** 2))
            l2_loss = tf.add_n(l2_losses)
            loss += self.params.l2_reg * l2_loss
        return loss

    """ predicates """

    def use_lr_annealing(self):
        params = self.params
        if params.lr_cosine:
            assert params.lr_max is not None, "Need lr_max to use lr_cosine"
            assert params.lr_min is not None, "Need lr_min to use lr_cosine"
            assert params.lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
            assert params.lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
            assert params.num_train_batches is not None, "Need num_train_batches to use" " lr_cosine"
            return True
        return False

    def use_moving_average(self):
        return self.params.moving_average is not None

    def use_lr_warming_up(self):
        return self.params.lr_warmup_val is not None

    def use_sync_replicas(self):
        if self.params.sync_replicas:
            assert self.params.num_aggregate is not None, "Need num_aggregate to sync."
            assert self.params.num_replicas is not None, "Need num_replicas to sync."
            return True
        return False
