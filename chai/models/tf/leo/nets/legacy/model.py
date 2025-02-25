import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from context import Context
from tensorflow import losses
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add, Concatenate, Conv2D, Dense, Flatten, 
    Input, Lambda, LeakyReLU, Dropout, 
    AveragePooling2D, MaxPool2D, UpSampling2D, ZeroPadding2D)

def clip_gradients(grads, grad_threshold, grad_norm_threshold):
    """ Clips grads by value and then by norm """
    if grad_threshold > 0:
        grads = [tf.clip_by_value(g, -grad_threshold, grad_threshold) for g in grads]
    if grad_norm_threshold > 0:
        grads = [tf.clip_by_norm(g, grad_norm_threshold) for g in grads]
    return grads


"""
    LEO TF 2.1 ver 구현
    
>>> Future <<<
- 몇몇 classification 문제에 적합한 구현들은 시선 추적 문제에 맞게 변경하였으나
  차후 oridnal loss 도입하면 the omitted tricks for classfication must be introduced
- Learned learning-rate : lr-per-latent
- orientation, head-pose, extrinsic-parameter 등의 요소에 대해서도 클래스처럼 취급하여 
  매 순간 시계열 형식의 parameter gen 

>>> Terminology <<<
- We define the N-way K-shot problem
- num_classes: (5) N in N-way classification
  - 시선 추적 문제에 맞는 N-way 값은?

- 우리의 경우 num_examples_per_class 를 k_shot 변수로 쓰겠음
- num_tr_examples_per_class  : ( 1) Number of training samples per class (K in K-shot)
- num_val_examples_per_class : (15) Number of validation samples per class in a task instance
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
        tf.random.set_seed(seed)  # tfp도 같이 적용된다 함
        
    def setup_props(self):
        self.num_classes = 200 # 생성 파라미터의 임베딩 분포의 편차 크기에 영향
        self.result_bag = []
        self.is_meta_training = True

    def make_loss_fn(self):
        def euc_gaze_loss(y_true, y_pred):
            square = tf.math.square(y_pred - y_true)
            reduce_sum = tf.math.reduce_sum(square, axis=1)
            dists = tf.math.sqrt(reduce_sum)
            return tf.math.reduce_mean(dists)
        def mse_gaze_loss(y_true, y_pred):
            return tf.reduce_mean(tf.losses.mse(y_true, y_pred))
        
        if self.ctx.loss_type == 'mse':
            return mse_gaze_loss
        return euc_gaze_loss

    def _build(self, ctx):
        """ WARN: 함수명을 build로 쓰면 override 되서 파라미터 초기화 안 됨 """
        self.ctx = ctx
        self.int_dtype = tf.int64 if ctx.use_64bits else tf.int32
        self.float_dtype = tf.float64 if ctx.use_64bits else tf.float32
        if ctx.use_64bits:
            tf.keras.backend.set_floatx('float64')
            
        self.loss_fn = self.make_loss_fn()
        meta_lr = tf.keras.experimental.CosineDecayRestarts(self.ctx.meta_lr, self.ctx.first_decay_steps)
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
        self.fc1_encoder = Dense(self.ctx.num_latents, 
                                 use_bias=False, 
                                 name='encoder_final_fc1', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        self.fc1_encoder_dropout = Dropout(self.ctx.dropout_rate)
        
        self.fc2_encoder = Dense(self.ctx.num_latents_final, 
                                 use_bias=False, 
                                 name='encoder_final_fc2', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        
        """ latents / finetuning gradient learning-rate """
        latent_lr = np.tile(self.ctx.latent_lr, [1, self.ctx.num_latents])
        self.latent_lr = tf.Variable(latent_lr, name='latent_lr', dtype=self.float_dtype)
        
        latent_final_lr = np.tile(self.ctx.latent_lr, [1, self.ctx.num_latents_final])
        self.latent_final_lr = tf.Variable(latent_final_lr, 
                                           name='latent_final_lr', 
                                           dtype=self.float_dtype)
        
        finetuning_lr = np.tile(self.ctx.finetuning_lr, [1, self.ctx.gen_theta_dim])
        self.finetuning_lr = tf.Variable(finetuning_lr, 
                                         name='finetuning_lr', 
                                         dtype=self.float_dtype)
        
        finetuning_final_lr = np.tile(self.ctx.finetuning_lr, [1, self.ctx.gen_final_theta_dim])
        self.finetuning_final_lr = tf.Variable(finetuning_final_lr,  
                                               name='finetuning_final_lr', 
                                               dtype=self.float_dtype)
        
    def encode(self, task):
        """  EXP: what is the best way to extract prior knowledge from inputs
        - A. pair(image, label) 
        - B. tr_input으로 predict, tr_output 오차 사용
           - squared diff는 이후 sampling 할 때 KL 값이 너무 큼
           - TODO: final_fc1 변경 이후에서는 이것도 해볼 수 있을 듯
        - C. use features from predict model using tr_input
        - D. 오차들 + support-info (camera-parameter, cam to screen mm, orientation)
        """
        
        """ prior knowledge """
        preds, fv = self.predict(task)
        diffs = preds - task.tr_output
        support = task.tr_support     # TODO: split orientation & csd
        
        """ for final_fc1 """
        prior_x1 = tf.concat([support, diffs], axis=1, name='prior_fc1')
        z1 = self.fc1_encoder(prior_x1)
        if self.is_meta_training:
            z1 = self.fc1_encoder_dropout(z1)
            
        """ for final_fc2 """
        mse = tf.losses.mse(preds, task.tr_output)
        mse = tf.expand_dims(mse, 1) 
        prior_x2 = tf.concat([support, mse], axis=1, name='prior_fc2')
        z2 = self.fc2_encoder(prior_x2)
            
        return z1, z2
    
    
    """
        Relation Network 
    """
    def _build_relation(self):
        """ Modified to considering only in k-shot correlation but not in a batch,
            unlike original impl that dealing classification problem.
        - origin (N * K * dims...)
        relation_network_module = snt.nets.MLP(
                [2 * self._num_latents] * 3, use_bias=False, 
                regularizers={"w": regularizer}, initializers={"w": initializer})
            
        total_num_examples = self.ctx.num_k_shot * self.num_classes
        inputs = tf.reshape(inputs, [total_num_examples, self._num_latents])

        left  = tf.tile(tf.expand_dims(inputs, 1), [1, total_num_examples, 1])
        right = tf.tile(tf.expand_dims(inputs, 0), [total_num_examples, 1, 1])
        concat_codes = tf.concat([left, right], axis=-1)
        outputs = snt.BatchApply(relation_network_module)(concat_codes)
        outputs = tf.reduce_mean(outputs, axis=1)
        # 2 * latents, because we are returning means and variances of a Gaussian
        outputs = tf.reshape(
            outputs, [self.num_classes, self.num_examples_per_class, 2 * self._num_latents]) 
        """ 
        
        num_units = 2 * self.ctx.num_latents
        initializer = tf.keras.initializers.glorot_uniform(seed=self.ctx.seed)
        regularizer = tf.keras.regularizers.l2(self.ctx.l2_penalty_weight)

        num_input_shape = self.ctx.num_latents * self.ctx.num_k_shot
        self.relation_l1 = Dense(num_units, input_shape=(num_input_shape,), 
                                 use_bias=False,  activation='relu', name='relation_l1', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        self.relation_l2 = Dense(num_units, use_bias=False, 
                                 activation='relu', name='relation_l2', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        self.relation_l3 = Dense(num_units, use_bias=False, 
                                 activation='relu', name='relation_l3', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        
        num_units = 2 * self.ctx.num_latents_final
        num_input_shape = self.ctx.num_latents_final * self.ctx.num_k_shot
        self.relation_fl1 = Dense(num_units, input_shape=(num_input_shape,), 
                                 use_bias=False,  activation='relu', name='relation_fl1', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        self.relation_fl2 = Dense(num_units, use_bias=False, 
                                 activation='relu', name='relation_fl2', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        self.relation_fl3 = Dense(num_units, use_bias=False, 
                                 activation='relu', name='relation_fl3', 
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=regularizer)
        
    def relation(self, z1, z2):
        x1 = self.relation_l1(z1)
        x1 = self.relation_l2(x1)
        x1 = self.relation_l3(x1)
                                  
        x2 = self.relation_fl1(z2)
        x2 = self.relation_fl2(x2)
        x2 = self.relation_fl3(x2)
        return x1, x2
    
    
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
    
    @tf.function
    def _build_decoder(self):
        initializer = tf.keras.initializers.glorot_uniform(seed=self.ctx.seed)
        l2_regularizer = tf.keras.regularizers.l2(self.ctx.l2_penalty_weight)
        # 2 * dim, because we are returning means and variances
        num_units = 2 * self.ctx.gen_theta_dim
        self.decoder_fc1 = Dense(num_units, use_bias=False, name='decoder_fc1',
                             kernel_initializer=initializer, 
                             kernel_regularizer=l2_regularizer)
        
        initializer = tf.keras.initializers.glorot_uniform(seed=self.ctx.seed)
        l2_regularizer = tf.keras.regularizers.l2(self.ctx.l2_penalty_weight)
        num_units = 2 * self.ctx.gen_final_theta_dim
        self.decoder_fc2 = Dense(num_units, use_bias=False, name='decoder_fc2',
                             kernel_initializer=initializer, 
                             kernel_regularizer=l2_regularizer)
        
        """ 직교 규제화항: 이 시점에는 dead body 상태라 웨이트를 얻어올 수 없으므로 런타임에서 계산 """
    
    """
        Sampler
    """
    def possibly_sample(self, dist_params, stddev_offset=0., use_fixed=False):
        means, unnormalized_stddev = tf.split(dist_params, 2, axis=-1)
        if not self.is_meta_training or use_fixed:
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
    def _build_gaze_model(self, activation='relu'):
        # Shared-eye-Net
        self.avg_pool1 = AveragePooling2D(pool_size=(7, 7))
        self.avg_pool2 = AveragePooling2D(pool_size=(5, 5))
        
        self.conv1 = Conv2D(filters=32, kernel_size=(7,7), padding='same', activation=activation)
        self.conv2 = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation=activation)
        self.conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=activation)
        
        # landmark 
        self.ec_fc1 = Dense(100, activation=activation, name='ec_fc1')
        self.ec_fc2 = Dense(16,  activation=activation, name='ec_fc2')
        self.ec_fc3 = Dense(16,  activation=activation, name='ec_fc3')
        
        self.final_fc1 = Dense(32, activation=activation, name='final_fc1')
        self.final_fc12 = Dense(8,  activation=activation, name='final_fc12')
        self.final_fc2 = Dense(2,  activation='linear',   name='final_fc2')
        
    def extract_eye(self, eye_patch):
        x = self.conv1(eye_patch)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        x = self.conv3(x)
        x = tf.squeeze(x)
        return x

    def predict(self, task):
        le, re, ec = task.tr_in1, task.tr_in2, task.tr_in3

        ecx = self.ec_fc1(ec)
        ecx = self.ec_fc2(ecx)
        ecx = self.ec_fc3(ecx)
        lex = self.extract_eye(le)
        rex = self.extract_eye(re)
        
        x = Concatenate(name='bottleneck')([lex, rex, ecx])
        z = self.final_fc1(x)
        z = self.final_fc12(z)
        z = self.final_fc2(z)
        return z, x
    
    def predict_with(self, task, theta, theta2, use_train=True):
        """ NOTE: 타겟 레이어의 파라미터 theta 대치 (gradient 계산 연결을 위해)
        A. (failed) keras set_weights 방식 + get_value(theta)  
           --> Non-eager Tensor 타입 대입 자체가 불가
           fi_fc1, fi_fc2 = tf.split(theta, [16, 2], axis=1)
           self.final_fc1.set_weights(tf.keras.backend.get_value(fi_fc1))
           self.final_fc2.set_weights(fi_fc2) 
           !-> kernel, bias 2쌍
        B. (success) theta를 통한 직접 레이어 구현 
           - compute value directly from theta (o -> gradient 흐르는 것 확인함)
           - TODO) 아예 custom module로 만들어서 바로 쓸 수 있도록 하기
           - TODO) gen_theta_dim은 타겟 레이어들의 shape들의 합으로 자동 계산하도록
        """
        if use_train:
            le, re, ec = task.tr_in1, task.tr_in2, task.tr_in3
        else:
            le, re, ec = task.val_in1, task.val_in2, task.val_in3
            
        ecx = self.ec_fc1(ec)
        ecx = self.ec_fc2(ecx)
        ecx = self.ec_fc3(ecx)
        lex = self.extract_eye(le)
        rex = self.extract_eye(re)

        fi_fc1_k, fi_fc1_b = tf.split(theta, [80*32, 32*1], axis=1)
        fi_fc1_k = tf.reshape(fi_fc1_k, (80, 32))   # final_fc1 kernel shape
        fi_fc1_b = tf.reshape(fi_fc1_b, (32,  ))    # final_fc2 bias   shape
        
        z = Concatenate(name='bottleneck')([lex, rex, ecx])
        z = tf.einsum("ij,jk->ik", z, fi_fc1_k)
        z = z + fi_fc1_b
        z = tf.nn.relu(z)
        
        fi_fc2_k, fi_fc2_b = tf.split(theta2, [32*8, 8*1], axis=1)
        fi_fc2_k = tf.reshape(fi_fc2_k, (32, 8))   
        fi_fc2_b = tf.reshape(fi_fc2_b, (8,  ))    
        z = tf.einsum("ij,jk->ik", z, fi_fc2_k)
        z = z + fi_fc2_b
        z = tf.nn.relu(z)
        
        z = self.final_fc2(z)
        
        return z
    
    """
        High Level
    """
    def forward_encoder(self, task):
        z1, z2 = self.encode(task)
        z1 = tf.reshape(z1, [1, np.prod(z1.shape)])    # k-shot mixing
        z2 = tf.reshape(z2, [1, np.prod(z2.shape)])    # k-shot mixing
                                  
        z1, z2 = self.relation(z1, z2)
        latents, kl = self.possibly_sample(z1)
        latents2, kl2 = self.possibly_sample(z2, use_fixed=True)
        
        return latents, kl+kl2, latents2
    
    def forward_decoder(self, latents, latents2):
        weights_dist_params = self.decoder_fc1(latents)
        # Default to glorot_initialization and not stddev=1.
        fan_in = self.ctx.gen_theta_dim
        fan_out = self.num_classes
        stddev_offset = np.sqrt(2. / (fan_out + fan_in))
        gen_theta, kl = self.possibly_sample(weights_dist_params, stddev_offset)
        
        weights_dist_params = self.decoder_fc2(latents2)
        gen_final_theta, kl2 = self.possibly_sample(weights_dist_params, 0.0)
        
        return gen_theta, gen_final_theta, kl + kl2
    
    def calculate_inner_loss(self, task, gen_theta, gen_theta2, use_train=True):
        model_outputs = self.predict_with(task, gen_theta, gen_theta2, use_train)
        truth_outputs = task.tr_output if use_train else task.val_output
        if not use_train:
            self.result_bag.append({'pred':model_outputs, 'true':truth_outputs})
        return self.loss_fn(model_outputs, truth_outputs)
    
    def leo_inner_loop(self, task, latents, latents2):
        starting_latents = latents
        starting_latents2 = latents2
        theta, theta2, _ = self.forward_decoder(latents, latents2)
        loss = self.calculate_inner_loss(task, theta, theta2)
        
        for _ in range(self.ctx.num_latent_grad_steps):
            loss_grad = tf.gradients(loss, latents)
            latents -= self.latent_lr * loss_grad[0]
            
            loss_grad = tf.gradients(loss, latents2)
            latents2 -= self.latent_final_lr * loss_grad[0]
            
            theta, theta2, _ = self.forward_decoder(latents, latents2)
            loss = self.calculate_inner_loss(task, theta, theta2)
            
        penalty = 0.0 
        if self.is_meta_training:
            penalty += losses.mse(tf.stop_gradient(latents), starting_latents)
            penalty += losses.mse(tf.stop_gradient(latents2), starting_latents2)
        encoder_penalty = tf.cast(penalty, self.float_dtype)
        return loss, theta, theta2, encoder_penalty
    
    def finetuning_inner_loop(self, task, leo_loss, theta, theta2):
        tr_loss = leo_loss
        for _ in range(self.ctx.num_finetune_grad_steps):
            loss_grad = tf.gradients(tr_loss, theta)
            theta -= self.finetuning_lr * loss_grad[0]
            
            loss_grad = tf.gradients(tr_loss, theta2)
            theta2 -= self.finetuning_final_lr * loss_grad[0]
            
            tr_loss = self.calculate_inner_loss(task, theta, theta2)
        return tr_loss, theta, theta2
    
    def relay_theta(self, theta):
        """ 성능이 나빠지지만 다른 상황에서 사용되면 개선될 수도 있을 것 같아서 """
        kernel, bias = tf.split(theta, [80*32, 32*1], axis=1)
        kernel = tf.reshape(kernel, (80, 32))
        bias = tf.reshape(bias, (32,))
        self.final_fc1.set_weights([kernel, bias])
    
    def run_with_test(self, num_epochs, data, metrics, test_task_id=0, prior_task_id=0):
        test_tasks = data[test_task_id]
        for e in range(num_epochs):
            for i, tasks in enumerate(data):
                if i != test_task_id:
                    res, loss, theta = self(tasks)
                    metrics.add(res, loss)
                    metrics.next_batch()
                    # self.relay_theta(theta)
            metrics.add_test(self.test(test_tasks, prior_task_id))
            metrics.next_epoch()
    
    @tf.function
    def test(self, tasks, prior_task_id=0):
        self.is_meta_training = False
        
        self.result_bag = []
        task_for_gen = tasks[prior_task_id]
        theta, theta2, kl, encoder_penalty = self.generate_theta(task_for_gen)
        for task in tasks:
            self.calculate_inner_loss(task, theta, theta2, False)   
        return self.result_bag
        
    @tf.function
    def __call__(self, tasks):
        val_loss = []
        self.result_bag = []
        self.is_meta_training = True
        for task in tasks:
            loss, theta = self.run_per_task(task)
            val_loss.append(loss)
        batch_val_loss = tf.reduce_mean(val_loss)
        self.learn(batch_val_loss)
        
        return self.result_bag, val_loss, theta
    
    def generate_theta(self, task):
        # sample init latents from input prior
        latents, kl, z2 = self.forward_encoder(task)
        
        # adapt in latent embedding space
        loss, theta, theta2, encoder_penalty = self.leo_inner_loop(task, latents, z2)
        
        # adapt directly parameter space
        if self.ctx.num_finetune_grad_steps > 0:
            _, theta, theta2 = self.finetuning_inner_loop(task, loss, theta, theta2)
            
        return theta, theta2, kl, encoder_penalty
    
    def run_per_task(self, task):
        theta, theta2, kl, encoder_penalty = self.generate_theta(task)
        
        # compute validate loss by final theta
        val_loss = self.calculate_inner_loss(task, theta, theta2, False)   
        
        # append regularizer term in loss 
        val_loss += self.ctx.kl_weight * kl
        val_loss += self.ctx.encoder_penalty_weight * encoder_penalty
        # decoder_orthogonality_reg = self.compute_orthogonality_reg(self.decoder_fc1.weights)
        regularization_penalty = self.l2_regularization # + decoder_orthogonality_reg
        
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
        meta_vars = [v for v in meta_vars if 'final_fc1' not in v.name]
        
        if self.ctx.num_finetune_grad_steps > 0:
            return meta_vars
        else: # NOTE: gradinet of theta finetune-lr will be None.
            meta_vars = [v for v in meta_vars if 'finetuning_lr' not in v.name]
        return meta_vars
        
    def grads_and_vars(self, meta_loss):
        meta_vars = self.get_trainable_variables()
        meta_grads = tf.gradients(meta_loss, meta_vars)
        
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
        