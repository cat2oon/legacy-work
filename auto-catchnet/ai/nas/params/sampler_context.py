import tensorflow as tf

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits


class SamplerContext:

    def __init__(self):
        self.arc_seq = []
        self.entropys = []
        self.log_probs = []
        self.skip_count = []
        self.skip_penaltys = []
        self.anchors = []
        self.anchors_w_1 = []

        self.sample_arc = None
        self.sample_entropy = None
        self.sample_log_prob = None
        self.sample_skip_count = None
        self.sample_skip_penaltys = None

    def clear(self):
        self.arc_seq.clear()
        self.entropys.clear()
        self.log_probs.clear()
        self.skip_count.clear()
        self.skip_penaltys.clear()
        self.anchors.clear()
        self.anchors_w_1.clear()
        self.sample_arc.clear()
        self.sample_entropy.clear()
        self.sample_log_prob.clear()
        self.sample_skip_count.clear()
        self.sample_skip_penaltys.clear()

    def wrap_up(self):
        arc_seq = tf.concat(self.arc_seq, axis=0)
        self.sample_arc = tf.reshape(arc_seq, [-1])

        entropys = tf.stack(self.entropys)
        self.sample_entropy = tf.reduce_sum(entropys)

        log_probs = tf.stack(self.log_probs)
        self.sample_log_prob = tf.reduce_sum(log_probs)

        skip_count = tf.stack(self.skip_count)
        self.sample_skip_count = tf.reduce_sum(skip_count)

        skip_penaltys = tf.stack(self.skip_penaltys)
        self.sample_skip_penaltys = tf.reduce_mean(skip_penaltys)

    def add_branch(self, branch_id, logit):
        self.arc_seq.append(branch_id)
        log_prob = cross_entropy(logits=logit, labels=branch_id)
        self.log_probs.append(log_prob)
        entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
        self.entropys.append(entropy)

    def add_start(self, start, logit):
        self.arc_seq.append(start)
        log_prob = cross_entropy(logits=logit, labels=start)
        self.log_probs.append(log_prob)
        entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
        self.entropys.append(entropy)

    def add_count(self, count, logit):
        self.arc_seq.append(count + 1)
        log_prob = cross_entropy(logits=logit, labels=count)
        self.log_probs.append(log_prob)
        entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
        self.entropys.append(entropy)

    def add_skip(self, layer_id, skip, logit, skip_targets):
        self.arc_seq.append(skip)
        skip_prob = tf.sigmoid(logit)
        kl = skip_prob * tf.log(skip_prob / skip_targets)
        kl = tf.reduce_sum(kl)
        self.skip_penaltys.append(kl)

        log_prob = cross_entropy(logits=logit, labels=skip)
        self.log_probs.append(tf.reduce_sum(log_prob, keep_dims=True))
        entropy = tf.stop_gradient(tf.reduce_sum(log_prob * tf.exp(-log_prob), keep_dims=True))
        self.entropys.append(entropy)

        skip = tf.to_float(skip)
        skip = tf.reshape(skip, [1, layer_id])
        self.skip_count.append(tf.reduce_sum(skip))

        return skip

    def add_anchor(self, anchor, w_attn_1):
        self.anchors.append(anchor)
        self.anchors_w_1.append(tf.matmul(anchor, w_attn_1))
