import os
import logging

import numpy as np
import tensorflow as tf

"""
    Custom Optimizer
    
    용어:
    - inner_loop_lr <-- task_learning_rate (MAML++에서는 task_lr과 inner_lr이 같은 의미)
    - gradient_steps <-- number_of_training_steps_per_iter (++용어보다 원본이 좋은 듯)
    - use_learnable_learning_rates (lr 학습 기법 사용 여부)
        대응 설정명: learnable_per_layer_per_step_inner_loop_learning_rate 
    
    - MAML++의 Learned Learning Rate는 PyTorch AutoGrad로 암묵적으로 처리하기 때문에 
      meta-SGD 논문 TF 구현을 참고하였음 (github.com/ash3n/Meta-SGD-TF/blob/master/meta_sgd.py)
"""

class LSLRGradientDescentOptimizer(tf.Module):
    
    def __init__(self, network, gradient_steps, init_learning_rate=1e-3, use_learnable_lr=True):
        super(LSLRGradientDescentOptimizer, self).__init__(name='LSLROptimizer')
        self.gradient_steps = gradient_steps
        self.use_learnable_lr = use_learnable_lr
        self.init_learning_rate = init_learning_rate
        self.build(network)
        
    def build(self, network):
        network_dtype = network.dtype
        base_lr = self.init_learning_rate
        num_grad_steps = self.gradient_steps
        trainable_vars = network.trainable_variables
        
        """  TODO: BN Layer 파라미터 형태 """
        kernel_name_to_index = {}
        for idx, tf_var in enumerate(self._iter_layer_from(trainable_vars)):
            layer_kernel_name = tf_var.name         # key는 kernel 이름 기준
            kernel_name_to_index[layer_kernel_name] = idx
        
        num_layer = len(self._iter_layer_from(trainable_vars))
        lr_per_step_per_layer = np.tile(base_lr, (num_layer, num_grad_steps)).astype(network_dtype)
        self.learned_lr = tf.Variable(lr_per_step_per_layer, name="learned_lr") 
        self.kernel_name_to_idx = kernel_name_to_index
    
    def _iter_layer_from(self, trainable_vars):
        """ WARN: layer {kernel, bias} 전제 깨지면 형태별 처리 필요 """ 
        return trainable_vars[::2]    
        
        
    """
        Optimizer API
    """
    def apply(self, grads_and_vars, num_step):
        updated_params = []
        for i, (grad, var) in enumerate(grads_and_vars):
            learned_lr = self.get_learning_rate_by_idx(int(i/2), num_step)
            updated_var = tf.subtract(var, tf.multiply(learned_lr, grad))
            var.assign(updated_var)
            updated_params.append(updated_var)
        return updated_params
            
    def apply_gradients(self, trainable_vars, gradients, num_step):
        """
        TF GradientTape에서 넘겨주는 gradient는 named 처리가 되어있지 않은 리스트 형태
        단, gradient 계산 당시의 입력된 trainable_vars 내부 순서를 그대로 따름
        """
        theta_lr = self.trainable_variables
        for i, tf_var in enumerate(self._iter_layer_from(trainable_vars)):
            layer_name = tf_var.name
            k_id, b_id = 2*i+0, 2*i+1     
            learned_lr = self.get_learning_rate_by_idx(i, num_step)
            theta_prev_k, theta_prev_b = trainable_vars[k_id:b_id+1]        # (k)ernel, (b)ias
            theta_next_k = tf.subtract(theta_prev_k, tf.multiply(learned_lr, gradients[k_id]))
            theta_next_b = tf.subtract(theta_prev_b, tf.multiply(learned_lr, gradients[b_id]))
            theta_prev_k.assign(theta_next_k)
            theta_prev_b.assign(theta_next_b)   
            
    def get_learning_rate(self, layer_name, num_step):
        idx = self.kernel_name_to_idx[layer_name]
        return self.learned_lr[idx][num_step]
    
    def get_learning_rate_by_idx(self, idx, num_step):
        return self.learned_lr[idx][num_step]

    @tf.custom_gradient
    def gradientable_assign(x, y):
        """ www.tensorflow.org/api_docs/python/tf/custom_gradient """
        """ www.tensorflow.org/api_docs/python/tf/gradients"""
        """ github.com/tensorflow/tensorflow/issues/17735 """
        x = tf.compat.v1.assign(x, y)
        def grad(dy):
            return dy
        return x, grad 