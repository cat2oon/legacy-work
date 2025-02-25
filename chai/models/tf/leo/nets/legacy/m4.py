import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from nets.layers import *
from nets.eyenet import EyeNet
from context import Context
from tensorflow import losses
from tensorflow.keras import Model
from tensorflow.keras.layers import *
                                     

def clip_gradients(grads, grad_threshold, grad_norm_threshold):
    if grad_threshold > 0:
        grads = [tf.clip_by_value(g, -grad_threshold, grad_threshold) for g in grads]
    if grad_norm_threshold > 0:
        grads = [tf.clip_by_norm(g, grad_norm_threshold) for g in grads]
    return grads


"""
    LEO M4
"""
class Leo(tf.keras.Model):

    @classmethod
    def create(cls, ctx, *args, **kwargs):
        return Leo(ctx, *args, **kwargs)
        
    def __init__(self, ctx, *args, **kwargs):
        super().__init__(name='Leo', *args, **kwargs)
        self.init_random_seed(ctx.seed)
        self.setup_props()
        self._build(ctx)
        
    def init_random_seed(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)  
        
    def setup_props(self):
        self.num_classes = 200 
        self.result_bag = []
        self.is_meta_training = True

    def make_loss_fn(self):
        def gaze_loss(label, pred):
            return tf.reduce_mean(tf.losses.mse(label, pred))

        def cos_loss(label, pred):
            cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
            return cosine_loss(label, pred)

        def euclidean_loss(label, pred):
            square = tf.math.square(pred - label)
            reduce_sum = tf.math.reduce_sum(square, axis=1)
            dists = tf.math.sqrt(reduce_sum)
            return tf.math.reduce_mean(dists)

        def compound_loss(label, pred):
            pred_vec,  pred_xy  = tf.split(pred,  [3, 2], 1)
            label_vec, label_xy = tf.split(label, [3, 2], 1)
            xy_loss = gaze_loss(label_xy, pred_xy)
            vec_loss = gaze_loss(label_vec, pred_vec)
            xy_euc_loss = euclidean_loss(label_xy, pred_xy)
            return xy_loss + vec_loss + xy_euc_loss
        
        return compound_loss

    def _build(self, ctx):
        self.ctx = ctx
        self.int_dtype = tf.int64 if ctx.use_64bits else tf.int32
        self.float_dtype = tf.float64 if ctx.use_64bits else tf.float32
        if ctx.use_64bits:
            tf.keras.backend.set_floatx('float64')
            
        self.loss_fn = self.make_loss_fn()
        meta_lr = tf.keras.experimental.CosineDecayRestarts(self.ctx.meta_lr,
                                                            self.ctx.first_decay_steps)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
        
        self._build_encoder()
        self._build_relation()
        self._build_decoder()
        self._build_gaze_model()
        
        
    """
        Encoder
        - 차후 conv를 포함한 인코더를 사용해 볼 수도 있을 듯
    """
    def _build_encoder(self):
        initializer = tf.keras.initializers.glorot_uniform(seed=self.ctx.seed)
        regularizer = tf.keras.regularizers.l2(self.ctx.l2_penalty_weight)
        
        self.encoder_l1 = Dense(128, use_bias=False, 
                             activation='selu', name='encoder_l1',
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer)
        self.encoder_l2 = Dense(self.ctx.num_latents, use_bias=False, 
                             activation='selu', name='encoder_l2',
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer)
        
        self.encoder_dropout_l1 = Dropout(self.ctx.dropout_rate)
        self.encoder_dropout_l2 = Dropout(self.ctx.dropout_rate)
        
        """ latents / finetuning gradient learning-rate """
        latent_lr = np.tile(self.ctx.latent_lr, [1, self.ctx.num_latents])
        self.latent_lr = tf.Variable(latent_lr, name='latent_lr', dtype=self.float_dtype)
        
        finetuning_lr = np.tile(self.ctx.finetuning_lr, [1, self.ctx.gen_theta_dim])
        self.finetuning_lr = tf.Variable(finetuning_lr, name='finetuning_lr', dtype=self.float_dtype)
        
    def encode(self, task):
        """ prior knowledge """
        preds = self.predict(task)  # preds, fv = self.predict(task)
        diffs = preds - task.tr_output
        # sign = tf.math.sign(task.tr_output - preds)
        # ssd = sign * tf.math.squared_difference(preds, task.tr_output)
        
        prior = diffs
        # prior = tf.concat([task., diffs], axis=1, name='prior')
        
        x = self.encoder_l1(prior)
        if self.is_meta_training:
            x = self.encoder_dropout_l1(x)
        x = self.encoder_l2(x)
        if self.is_meta_training:
            x = self.encoder_dropout_l2(x)
            
        return x
    
    
    """
        Relation Network 
    """
    def _build_relation(self):
        num_units = 2 * self.ctx.num_latents
        initializer = tf.keras.initializers.glorot_uniform(seed=self.ctx.seed)
        regularizer = tf.keras.regularizers.l2(self.ctx.l2_penalty_weight)
        num_input_shape = self.ctx.num_latents * self.ctx.num_k_shot
        self.relation_l1 = Dense(num_units, input_shape=(num_input_shape,), 
                                 use_bias=False,  activation='selu', name='relation_l1', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        self.relation_l2 = Dense(num_units, use_bias=False, 
                                 activation='selu', name='relation_l2', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        self.relation_l3 = Dense(num_units, use_bias=False, 
                                 activation='selu', name='relation_l3', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        
    def relation(self, inputs):
        x = self.relation_l1(inputs)
        x = self.relation_l2(x)
        x = self.relation_l3(x)
        return x
    
    
    """
        Decoder
    """
    def compute_orthogonality_reg(self, weight):
        base_dtype = self.float_dtype
        penalty_weight = self.ctx.orthogonality_penalty_weight
        w2 = tf.matmul(weight, weight, transpose_b=True)
        wn = tf.norm(weight, ord=2, axis=1, keepdims=True) + 1e-32
        correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
        matrix_size = correlation_matrix.get_shape().as_list()[0]
        identity = tf.eye(matrix_size, dtype=base_dtype)
        weight_corr = tf.reduce_mean(tf.math.squared_difference(correlation_matrix, identity))
        return tf.multiply(tf.cast(penalty_weight, base_dtype), weight_corr, "orthogonality")
    
    @tf.function(experimental_compile=True)
    def _build_decoder(self):
        initializer = tf.keras.initializers.glorot_uniform(seed=self.ctx.seed)
        l2_regularizer = tf.keras.regularizers.l2(self.ctx.l2_penalty_weight)
        num_units = 2 * self.ctx.gen_theta_dim
        self.decoder = Dense(num_units, activation='selu', 
                             use_bias=False,  name='decoder',
                             kernel_initializer=initializer, 
                             kernel_regularizer=l2_regularizer)
    
    """
        Sampler
    """
    def possibly_sample(self, dist_params, stddev_offset=0.):
        means, unnormalized_stddev = tf.split(dist_params, 2, axis=-1)
        if not self.is_meta_training:
            return means, tf.constant(0., dtype=self.float_dtype)

        stddev = tf.exp(unnormalized_stddev)
        stddev -= (1. - stddev_offset)
        stddev = tf.maximum(stddev, 1e-10)
        distribution = tfp.distributions.Normal(loc=means, scale=stddev)
        samples = distribution.sample()
        kl_divergence = self.kl_divergence(samples, distribution)
        return samples, kl_divergence
    
    def kl_divergence(self, samples, normal_distribution):
        random_prior = tfp.distributions.Normal(loc=tf.zeros_like(samples),
                                                scale=tf.ones_like(samples))
        kl = tf.reduce_mean(normal_distribution.log_prob(samples) - random_prior.log_prob(samples))
        return kl
    
        
    """
        Gaze Model
    """
    def _build_gaze_model(self, act='selu'):
        self.eyenet    = EyeNet()
        self.fc_eye    = Dense(256, use_bias=False, activation='selu',   name='fc_eye')
        self.fc_gaze   = Dense(3  , use_bias=True,  activation='linear', name='fc_gaze_vec')
        self.fc_target = Dense(2  , use_bias=False, activation='linear', name='fc_target')

    def eye_gaze(self, le, re):
        fle = tf.image.flip_left_right(le)
        fre = tf.image.flip_left_right(re)
        le = self.eyenet(le)
        re = self.eyenet(re)
        fle = self.eyenet(fle)
        fre = self.eyenet(fre)
        x = Concatenate(name='eye_feature')([le, re, fle, fre])
        x = self.fc_eye(x)
        return x
    
    def get_inputs(self, t, from_train=True):
        if from_train:
            return t.tr_le, t.tr_re, t.tr_rot_mat, t.tr_z_scale
        return t.val_le, t.val_re, t.val_rot_mat, t.val_z_scale

    def predict(self, task):
        le, re, rm, zs = self.get_inputs(task, from_train=True)
        
        ex = self.eye_gaze(le, re)
        v = self.fc_gaze(ex)
        z = Concatenate(name='gaze')([v, rm, zs])
        z = self.fc_target(z)
        out = Concatenate(name='out')([v, z])
        return out
    
    def predict_with(self, task, theta, use_train=True):
        le, re, rm, zs =  self.get_inputs(task, use_train)
          
        ex = self.eye_gaze(le, re)
        k = self.trainable_variables[7]   # fc_gaze_vec/kernel
        b = self.trainable_variables[8]   # fc_gaze_vec/kernel
        b = b + theta                     # fc_gaze_vec/bias delta
        v = self.dense(ex, k, b)          # fc_gaze
        
        z = Concatenate(name='gaze')([v, rm, zs])
        z = self.fc_target(z)
        out = Concatenate(name='out')([v, z])
        
        return out

    def dense(self, x, k, b):
        x = tf.einsum("ij,jk->ik", x, k)
        x = x + b
        x = tf.nn.selu(x)
        return x 
      
    def dense_from(self, x, theta, shape_i, shape_j):
        k, b = tf.split(theta, [shape_i*shape_j, shape_j*1], axis=1)
        k = tf.reshape(k, (shape_i, shape_j))
        b = tf.reshape(b, (shape_j, ))
        
        x = tf.einsum("ij,jk->ik", x, k)
        x = x + b
        x = tf.nn.selu(x)
        return x

      
    """
        High Level
    """
    def forward_encoder(self, task):
        z = self.encode(task)
        z = tf.reshape(z, [1, np.prod(z.shape)])    # k-shot mixing
        z = self.relation(z)
        latents, kl = self.possibly_sample(z)
        return latents, kl
    
    def forward_decoder(self, latents):
        weights_dist_params = self.decoder(latents)
        fan_in = self.ctx.gen_theta_dim
        fan_out = self.num_classes
        stddev_offset = np.sqrt(2. / (fan_out + fan_in))
        gen_theta, kl = self.possibly_sample(weights_dist_params, stddev_offset)
        return gen_theta, kl
    
    def calculate_inner_loss(self, task, gen_theta, use_train=True):
        model_outputs = self.predict_with(task, gen_theta, use_train)
        truth_outputs = task.tr_output if use_train else task.val_output
        if not use_train:
            self.result_bag.append({'pred':model_outputs, 'true':truth_outputs})
        return self.loss_fn(model_outputs, truth_outputs)
    
    def leo_inner_loop(self, task, latents):
        starting_latents = latents
        theta, _ = self.forward_decoder(latents)
        loss = self.calculate_inner_loss(task, theta)
        
        for _ in range(self.ctx.num_latent_grad_steps):
            loss_grad = tf.gradients(loss, latents)
            latents -= self.latent_lr * loss_grad[0]
            theta, _ = self.forward_decoder(latents)
            loss = self.calculate_inner_loss(task, theta)
            
        penalty = 0.0 
        if self.is_meta_training:
            penalty = losses.mse(tf.stop_gradient(latents), starting_latents)
        encoder_penalty = tf.cast(penalty, self.float_dtype)
        return loss, theta, encoder_penalty
    
    def finetuning_inner_loop(self, task, leo_loss, theta):
        tr_loss = leo_loss
        for _ in range(self.ctx.num_finetune_grad_steps):
            loss_grad = tf.gradients(tr_loss, theta)
            theta -= self.finetuning_lr * loss_grad[0]
            tr_loss = self.calculate_inner_loss(task, theta)
        return tr_loss, theta

    def run_with_test(self, num_epochs, data, metrics, test_task_ids, prior_task_id=0):
        for e in range(num_epochs):
            for i, tasks in enumerate(data):
                if i not in test_task_ids:
                    res, loss, theta = self(tasks)
                    metrics.add(res, loss)
                    metrics.next_batch()
            for id in test_task_ids:
                test_tasks = data[id]
                metrics.add_test(self.test(test_tasks, prior_task_id))
            metrics.next_epoch()
    
    @tf.function(experimental_compile=True)
    def test(self, tasks, prior_task_id=0):
        self.is_meta_training = False
        
        self.result_bag = []
        task_for_gen = tasks[prior_task_id]
        theta, kl, encoder_penalty = self.generate_theta(task_for_gen)
        for task in tasks:
            self.calculate_inner_loss(task, theta, False)   
        return self.result_bag
        
    @tf.function(experimental_compile=True)
    def __call__(self, tasks):
        val_loss = []
        self.result_bag = []
        self.is_meta_training = True
        for task in tasks:
            loss, theta = self.run_per_task(task)
            self.learn(loss)
            val_loss.append(loss)
        batch_val_loss = tf.reduce_mean(val_loss)
        self.learn(batch_val_loss)
        
        return self.result_bag, val_loss, theta
    
    def generate_theta(self, task):
        # sample init latents from input prior
        latents, kl = self.forward_encoder(task)
        
        # adapt in latent embedding space
        loss, theta, encoder_penalty = self.leo_inner_loop(task, latents)
        
        # adapt directly parameter space
        if self.ctx.num_finetune_grad_steps > 0:
            _, theta = self.finetuning_inner_loop(task, loss, theta)
            
        return theta, kl, encoder_penalty
    
    def run_per_task(self, task):
        theta, kl, encoder_penalty = self.generate_theta(task)
        
        # compute validate loss by final theta
        val_loss = self.calculate_inner_loss(task, theta, False)   
        
        # append regularizer term in loss 
        val_loss += self.ctx.kl_weight * kl
        val_loss += self.ctx.encoder_penalty_weight * encoder_penalty
        decoder_orthogonality_reg = self.compute_orthogonality_reg(self.decoder.weights)
        regularization_penalty = self.l2_regularization + decoder_orthogonality_reg
        
        loss = val_loss + regularization_penalty
                                  
        return loss, theta
    
    def learn(self, meta_loss):
        meta_grads, meta_vars = self.grads_and_vars(meta_loss)
        meta_grads = clip_gradients(meta_grads, 
                                    self.ctx.gradient_threshold,
                                    self.ctx.gradient_norm_threshold) 
        self.optimizer.apply_gradients(list(zip(meta_grads, meta_vars)))
        
    def get_trainable_variables(self):
        meta_vars = self.trainable_variables
        filter_names = []
        meta_vars = [v for v in meta_vars if v.name not in filter_names]
        
        if self.ctx.num_finetune_grad_steps > 0:
            return meta_vars
        else: # NOTE: gradinet of theta finetune-lr will be None.
            meta_vars = [v for v in meta_vars if 'finetuning_lr' not in v.name]
        return meta_vars
        
    def grads_and_vars(self, meta_loss):
        meta_vars = self.get_trainable_variables()
        meta_grads = tf.gradients(meta_loss, meta_vars)
        
        gvs = [z for z in list(zip(meta_grads, meta_vars)) if z[0] is None]
        if len(gvs) != 0: 
            print("Check broken gradients: ", gvs)
        
        is_nan_loss = tf.math.is_nan(meta_loss)
        is_nan_grad = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in meta_grads])
        nan_loss_or_grad = tf.logical_or(is_nan_loss, is_nan_grad)
        
        reg_penalty = (1e-4 / self.ctx.l2_penalty_weight * self.l2_regularization)
        zero_or_regularization_gradients = [g if g is not None else tf.zeros_like(v)
            for v, g in zip(tf.gradients(reg_penalty, meta_vars), meta_vars)]

        meta_grads = tf.cond(nan_loss_or_grad, 
                             lambda: zero_or_regularization_gradients, 
                             lambda: meta_grads)

        return meta_grads, meta_vars

    @property
    def l2_regularization(self):
        return tf.cast(tf.reduce_sum(tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)), dtype=self.float_dtype)
        