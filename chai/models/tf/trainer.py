from __future__ import division, print_function, absolute_import

import logging
import numpy as np
import tensorflow as tf

from context import Context


"""
    Trainer
"""
class Trainer():
    
    @classmethod
    def create(cls, ctx, network):
        return Trainer(ctx, network, ctx.seed)
    
    def __init__(self, config, network, seed,  **kwargs):
        self.prepare(config, network, seed, **kwargs)
        
    def prepare(self, config, network, seed, **kwargs):
        self.ctx = Context.create(config)
        self.network = network
        self.init_random_seed(seed)
        self.compile_model()
        self.setup_callbacks()
        
    def init_random_seed(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        
    """
        Dataset
    """
    def setup_sequence(self, train, valid, test, to_tf_data=False):
        self.train_seq, self.valid_seq, self.test_seq = train, valid, test
        if to_tf_data:
            train, valid, test = self.setup_tf_data_from_seq(train, valid, test)
        self.train_data_tf, self.valid_data_tf, self.test_data_tf = train, valid, test

    def setup_tf_data_from_seq(self, train, valid, test):
        # reference : www.tensorflow.org/guide/data
        # github.com/tensorflow/tensorflow/issues/20698 
        n = train.get_batch_size()  # self.ctx.batch_size
        from_gen = tf.data.Dataset.from_generator
        tensor_slices = tf.data.Dataset.from_tensor_slices
        output_shapes = (((n,64,64,3), (n,64,64,3), (n,8)), (n,2))
        output_types = ((tf.float32, tf.float32, tf.float32), tf.float32)
        
        train_ds = from_gen(self.get_generator(train),
                             output_types=output_types,
                             output_shapes=output_shapes)
        
        valid_ds = from_gen(self.get_generator(valid),
                             output_types=output_types,
                             output_shapes=output_shapes)

        return train_ds, valid_ds, test
                         
    def get_generator(self, sequence):
        def select_dict_keys(inputs):
            inputs = (inputs['left_eye_patch'], 
                      inputs['right_eye_patch'],  
                      inputs['eye_corner_landmark'])   
            return inputs    # WARN: tuple로 지정해야 함   
        
        def select_arr_seqs(inputs):
            return (inputs[0], inputs[1], inputs[2])   
        
        def generator():
            it = iter(sequence)
            while True:
                inputs, targets = next(it)
                # yield select_dict_keys(inputs), targets
                yield select_arr_seqs(inputs), targets
        return generator       
        
    """
        Loss, Optimizer, Metrics
    """
    def make_optimizer(self):
        lr = self.get_learning_rate()
        return tf.keras.optimizers.Adam(learning_rate=lr)
    
    def make_metrics(self, optimizer):
        return [ 
            # tf.keras.metrics.mean_absolute_error,  
            self.get_mean_distance_metric(),
            self.get_learning_rate_metric(optimizer)
        ]
    
    def get_mean_distance_metric(self):
        def mean_distance(y_true, y_pred):
            square = tf.math.square(y_pred - y_true)
            reduce_sum = tf.math.reduce_sum(square, axis=1)
            dists = tf.math.sqrt(reduce_sum)
            return tf.math.reduce_mean(dists)
        return mean_distance
    
    def get_learning_rate_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr('float32')
        return lr

    def get_learning_rate(self, use_warm_up=True):
        if use_warm_up:
            return tf.keras.experimental.CosineDecayRestarts(self.ctx.base_lr, 100)
        return self.ctx.base_lr

    def make_loss_fn(self):
        @tf.function
        def gaze_loss(label, pred):
            return tf.reduce_mean(tf.losses.mse(label, pred))
        return gaze_loss
        
    """
        Model
    """
    def compile_model(self):
        model = self.network
        if type(self.network) is 'type':
            model = self.network.create()
            
        self.model = model
        l = self.loss_fn = self.make_loss_fn()
        o = self.optimizer = self.make_optimizer()
        m = self.metrics = self.make_metrics(self.optimizer)
        model.compile(loss=l, optimizer=o, metrics=m)
    
    def get_model(self):
        return self.model
    
    def summary(self):
        self.model.summary()
        
    """
        Train
    """
    def train(self):
        assert self.model is not None, "model must be build"
        self.train_by_fit()  # self.train_by_call()
        
    @tf.function
    def train_step(self, batch_item):
        model = self.get_model()
        with tf.GradientTape() as tape:
            preds = model(batch_item)
            loss = compute_loss(label, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))    
        
    def train_by_call(self):
        for epoch in range(self.ctx.num_epochs):
            for batch_item in self.train_data_tf:
                self.train_step(batch_item)
        
    def train_by_fit(self):
        """ reference: /www.tensorflow.org/api_docs/python/tf/keras/Model """
        model = self.get_model()
        model.fit(
            x=self.train_data_tf,
            validation_data=self.valid_data_tf,
            epochs=self.ctx.num_epochs,
            callbacks=self.callbacks,
            shuffle=False)
            # max_queue_size=32,
            # workers=self.ctx.num_workers,
            # use_multiprocessing=True)

    """
        Callbacks
    """
    def setup_callbacks(self):
        logging.info("")
        logging.info(">>> setup callbacks")
        
        term_on_nan = tf.keras.callbacks.TerminateOnNaN
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint
        lr_adapt_cb = tf.keras.callbacks.ReduceLROnPlateau
        
        checkpoint_path = self.ctx.checkpoint_path
        logging.info("checkpoint save path %s" % checkpoint_path)
        cpc = checkpoint_cb(checkpoint_path, 
                            save_freq='epoch',
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=False,
                            save_weights_only=True)
        
        lr_adapt = lr_adapt_cb(monitor='mean_absolute_error', 
                               factor=0.2,
                               patience=5, 
                               min_lr=0.0001)
        
        stop_nan = term_on_nan()
        self.callbacks = [ cpc, lr_adapt, stop_nan ]
