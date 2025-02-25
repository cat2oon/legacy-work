import sys
import tqdm

from trainer import *
from maml.context import *
from maml.inner_loop_optimizers import *


"""
    MAML Trainer
"""
class MAMLTrainer(Trainer):
    
    @classmethod
    def create(cls, ctx, network_class):
        return MAMLTrainer(ctx, network_class, ctx.seed)

    def __init__(self, config, network_class, seed, **kwargs):
        super(MAMLTrainer, self).__init__(config, network_class, seed, **kwargs)
        
        
    """
        Build
    """
    def prepare(self, config, network_class, seed, **kwargs):
        self.ctx = MAMLContext.create(config)
        self.network_class = network_class
        self.init_random_seed(seed)
        self.compile_model()

    def make_optimizers(self, meta_network):
        inner_optimizer = LSLRGradientDescentOptimizer
        meta_lr = tf.keras.experimental.CosineDecayRestarts(self.ctx.meta_lr, 100)
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
        self.task_optimizer = inner_optimizer(meta_network, self.ctx.num_grad_steps, self.ctx.task_lr)

    def compile_model(self):
        self.meta_model = self.network_class.create()
        self.task_model = self.network_class.create()    # base-model
        self.make_optimizers(self.meta_model)
        self.loss_fn = self.make_loss_fn()
        
        
    """
        Datasets
    """
    def setup_tf_data_from_seq(self, train, valid, test):
        n = train.get_batch_size()  
        from_gen = tf.data.Dataset.from_generator
        tensor_slices = tf.data.Dataset.from_tensor_slices
        
        dtype = tf.float32
        types = ((dtype, dtype, dtype), dtype, (dtype, dtype, dtype), dtype)
        shapes = (((n,64,64,3), (n,64,64,3), (n,8)), (n,2), ((n,64,64,3), (n,64,64,3), (n,8)), (n,2))
        train_ds = from_gen(self.get_generator(train), output_types=types, output_shapes=shapes)
        valid_ds = from_gen(self.get_generator(valid), output_types=types, output_shapes=shapes)

        return train_ds, valid_ds, test

    def get_generator(self, sequence):
        def select_arr_seqs(inputs):
            return (inputs[0], inputs[1], inputs[2])   
        
        def generator():
            it = iter(sequence)
            while True:
                t_x, t_y = next(it)
                s_x, s_y = next(it)
                yield select_arr_seqs(s_x), s_y, select_arr_seqs(t_x), t_y
        return generator       
           
        
    """
        MAML
    """
    def transfer_weight(self, model_src, model_dest):
        for x, y in zip(model_dest.variables, model_src.variables):
            x.assign(y)  # copies the variables of meta_from into model_target
        return model_dest
    
    def transfer_trainable_vars(self, trainable_vars, x, model=None):
        if model is None:
            c = self.network_class.create()
        else:
            c = model
        c(x)
        c.set_weights(trainable_vars)
        return c

    def get_layer_matched(self, var_name, model):
        root_name = var_name[:var_name.find('/')]
        layer_names = [layer.name for layer in model.layers]
        
        matched_layer_id = -1
        for layer_id, layer_name in enumerate(layer_names):
            if layer_name == root_name:
                matched_layer_id = layer_id
        
        layer = model.layers[matched_layer_id]
        if not hasattr(layer, 'fit'):  # It's just layer not model
            return layer  
        
        suffix_name = var_name[var_name.find('/')+1:]
        return self.get_layer_matched(suffix_name, layer)
            
    def clone_model(self, model_from, x):
        cloned_model = self.network_class.create()
        cloned_model(x)         # shock dead body
        cloned_model.set_weights(model_from.get_weights())
        return cloned_model

    def compute_loss(self, model, inputs, targets):
        preds = model(inputs)
        loss = self.loss_fn(targets, preds)
        return loss, preds
    
    def compute_loss_with(self, model, theta, inputs, targets):
        preds = model(inputs, theta)
        loss = self.loss_fn(targets, preds)
        return loss, preds    
    
    def shock_dead_body(self):
        blood_pack = iter(self.valid_data_tf)
        s_x, s_y, t_x, t_y = next(blood_pack)
        self.meta_model(s_x)
        self.task_model(s_x)
        del blood_pack
        
        
    """
        MAML++ Enhancements Imples
    """
    def get_per_step_loss_importance_vector(self):
        """ 원본 구현을 그대로 따름 """
        curr_epoch = self.ctx.current_epoch
        num_grad_steps = self.ctx.num_grad_steps
        
        loss_weights = np.ones(shape=(num_grad_steps)) * (1.0 / num_grad_steps)
        decay_rate = 1.0 / num_grad_steps / self.ctx.multi_step_loss_num_epochs
        min_threshold = 0.03 / num_grad_steps
        
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (curr_epoch * decay_rate), min_threshold)
            loss_weights[i] = curr_value

        curr_value = np.minimum(loss_weights[-1] + (curr_epoch*(num_grad_steps-1)*decay_rate),
                                1.0 - ((num_grad_steps-1)*min_threshold))
        loss_weights[-1] = curr_value
        return loss_weights     # NOTE: numpy * tensor ==> tensor

        
    """
        MAML++ Trainer TF 2.x ver
    """
    def init_run_context(self):
        ctx = self.ctx
        ctx.current_iter = 0
        ctx.current_epoch = 0
        ctx.total_iter = len(self.train_seq) * self.ctx.num_epochs
        return ctx       
         
    def on_epoch_start(self, epoch_id):
        ctx = self.ctx
        ctx.total_loss = 0
        ctx.current_epoch = epoch_id
    
    def on_epoch_end(self, epoch_id):
        ctx.total_loss = 0
    
    def run_experiment(self):
        self.shock_dead_body()
        ctx = self.init_run_context()
        
        # While Convergence or Epochs End
        for e in range(self.ctx.num_epochs):
            self.on_epoch_start(e)

            # train iteration 
            for i, train_batch in enumerate(self.train_data_tf):
                self.run_train_iteration(ctx, train_batch)

            # validation iteration
            for i, valid_batch in enumerate(self.valid_data_tf):
                self.run_valid_iteration(ctx, valid_batch)

            self.on_epoch_end(e)

    """ TODO: flatten layers """
    def compute_updated_theta(self, model, grads, llr, grad_step, apply=False):
        tvar_idx = 0
        updated_layers = []
        for i, layer in enumerate(model.layers):
            if len(layer.trainable_weights) == 0:  # non trainable layer
                continue
                
            # when just layer module
            if not hasattr(layer, 'layers'): 
                lr = llr[int(tvar_idx/2)][grad_step]
                k = tf.subtract(layer.kernel, tf.multiply(lr, grads[tvar_idx+0]))
                b = tf.subtract(layer.bias,   tf.multiply(lr, grads[tvar_idx+1]))
                if apply:
                    layer.kernel = k
                    layer.bias = b
                updated_layers.append(k.numpy())
                updated_layers.append(b.numpy())
                tvar_idx +=2
                continue
                
            # when nested model    
            for j, n_layer in enumerate(layer.layers):
                if len(n_layer.trainable_weights) != 0:  
                    lr = llr[int(tvar_idx/2)][grad_step]
                    k = tf.subtract(n_layer.kernel, tf.multiply(lr, grads[tvar_idx+0]))
                    b = tf.subtract(n_layer.bias,   tf.multiply(lr, grads[tvar_idx+1]))
                    if apply:
                        layer.kernel = k
                        layer.bias = b
                    updated_layers.append(k.numpy())
                    updated_layers.append(b.numpy())
                    tvar_idx += 2
        return updated_layers

    """ template model version """
    def run_train_iteration(self, ctx, batch):
        support_x, support_y, target_x, target_y = batch
        meta_optimizer, task_optimizer = self.meta_optimizer, self.task_optimizer
        
        # multi-step loss optimization (MSL)
        target_losses, meta_losses = [], []
        meta_loss_weights = self.get_per_step_loss_importance_vector()
        
        """ COPY를 하는 이유는 meta_model을 zombie로 만들지 않기 위해서 """
        meta_model = self.meta_model
        origin_meta_backup = self.clone_model(meta_model, support_x)
        
        # print(origin_meta_backup.trainable_variables[0][0].numpy())
        
        """ trainable learning rate """
        lr = self.ctx.task_lr
        llr = np.tile(lr, (len(meta_model.layers), ctx.num_grad_steps)).astype(np.float32)
        llr = tf.Variable(llr, name='learned_lr')
        losses_for_lr = []
        
        # outer loop
        with tf.GradientTape(persistent=True) as target_tape:
            theta_next = [var.numpy() for var in meta_model.trainable_variables]
                          
            # inner loop (N gradient steps)
            for grad_step in range(ctx.num_grad_steps):   
                task_model = self.clone_model(meta_model, support_x)
                task_model = self.transfer_trainable_vars(theta_next, support_x, meta_model)
                
                with tf.GradientTape() as support_tape:
                    support_loss_i, _ = self.compute_loss(task_model, support_x, support_y)
                    # print(support_loss_i)
                    
                # compute theta_next by gradients descent
                task_grads = support_tape.gradient(support_loss_i, task_model.trainable_variables)
                theta_next = self.compute_updated_theta(task_model, task_grads, llr, grad_step)
                
                # loss for meta(:theta) learning of step_i
                phi_model = self.transfer_trainable_vars(theta_next, support_x, meta_model)
                target_loss, preds = self.compute_loss(phi_model, target_x, target_y) 
                target_losses.append(target_loss)
                
                # loss for meta(:learned-lr) learning of step_i
                rho_model = self.clone_model(meta_model, support_x)
                self.compute_updated_theta(rho_model, task_grads, llr, grad_step, apply=True)
                loss_for_lr, _ = self.compute_loss(rho_model, target_x, target_y)
                losses_for_lr.append(loss_for_lr)
                
            # multi step loss (msl) with per step loss importance
            for grad_step, target_loss in enumerate(target_losses):
                meta_losses.append(meta_loss_weights[grad_step] * target_loss)
            meta_loss = tf.reduce_sum(meta_losses)
            
        # meta(learned-lr) learning 
        for loss in losses_for_lr:
            lr_grads = target_tape.gradient(loss, [llr])
            meta_optimizer.apply_gradients(zip(lr_grads, [llr]))
        
        # print(llr.numpy())
        
        # meta(theta) learning
        theta_meta = meta_model.trainable_variables
        meta_grads = target_tape.gradient(meta_loss, theta_meta)
        meta_optimizer.apply_gradients(zip(meta_grads, theta_meta))
        del target_tape
        
        # print("next:", theta_meta[0][0].numpy())
        
        self.on_iter_end(target_losses)       
        
    def on_iter_end(self, target_losses):
        ctx = self.ctx
        loss = tf.reduce_mean(target_losses).numpy()
        self.compute_metrics(loss)
        ctx.current_iter += 1

    def compute_metrics(self, loss, preds=None, labels=None):
        self.ctx.total_loss += loss
        self.ctx.loss = self.ctx.total_loss / (self.ctx.current_iter+1.0)
        print(self.ctx.loss)
        
    def print_learning(self, i, preds, targets):
        loss = self.ctx.loss.numpy()
        e = self.ctx.current_epoch
        
        if 0 < i <= 250:
            logging.info('Step %d: loss %s' % (i, loss))
            return
        
        if i < 1000 and i % 20 == 0:
            logging.info('Step %d: loss %s' % (i, loss))
            return
        
        if i > 5000 and i % 100 == 0:
            logging.info('epoch[%d] step %d: loss %s' % (e, i, loss))
            return
