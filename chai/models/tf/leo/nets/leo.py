from context import *
from cheat.tfs import *
from nets.eyenet import EyeModule



"""
    LEO Gens GTA
"""
class Leo(tf.Module):

    @classmethod
    def create(cls, ctx, *args, **kwargs):
        set_random_seed(ctx.seed)
        return Leo(ctx, *args, **kwargs)
        
    """
        Building
    """
    def __init__(self, ctx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_props(ctx)
        with self.name_scope:
            self.make(ctx)
        
    def setup_props(self, ctx):
        self.ctx = ctx
        self.result_bag = []
        
        self.is_leo = False
        self.is_meta_training = True
        self.last_genc_theta = None
        
        self.seed = ctx.seed
        self.num_classes = 512 
        self.num_latents = ctx.num_latents
        self.num_k_shots = ctx.num_k_shots
        self.gen_theta_dims = ctx.gen_theta_dims

    """
        Build Network
    """
    def make(self, ctx):
        self.make_eyenet()
        self.make_genc()
        
        self.cosine_loss_fn = cos_loss
        self.target_loss_fn = gaze_loss
        self.mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
        lr = CosineDecayRestarts(self.ctx.meta_lr, self.ctx.first_decay_steps)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
    def make_eyenet(self):
        self.eyenet = EyeModule.create_bec(out_dim=8)
        self.depth_regressor = EyeModule.create_depth_regressor()
        
        """ TODO: valid_shot, k_shot 크기가 다른 경우 분기 해줘야 함 """
        batch_zero = tf.tile(tf.constant([0.0]), [self.num_k_shots])
        self.zero = tf.constant(0.0, name='const_zero')
        self.batch_zero = tf.reshape(batch_zero, (-1, 1))
        
        zero_xy = tf.tile(tf.constant([0.0, 0.0]), [self.num_k_shots])
        self.delta_zero_xy = tf.reshape(zero_xy, (-1, 2))
        
        """ TODO: regularizer for preventing optimize wrong way by noisy samples """
        self.kappa_alpha_beta_both = variable([5.0, 2.0, -5.0, 2.0], name='kappa_both')   
        
    def make_genc(self):
        num_latents = self.num_latents
        theta_dims = self.gen_theta_dims
        var, seed = var_with_shape, self.seed
        
        # latent learning rate 
        self.theta_lr  = var_tile(self.ctx.theta_lr,  [1, theta_dims],  name='theta_lr')
        self.latent_lr = var_tile(self.ctx.latent_lr, [1, num_latents], name='latent_lr')
        
        """ EXP: bias + mlp """
        # encoder
        num_genc_input = 44           # quick: make_calibration_samples()
        self.enc1_w = var((num_genc_input, 16), name="enc1_w", seed=seed)
        self.enc1_b = var((16,),                name="enc1_b", seed=seed)
        self.enc2_w = var((16, 8),              name="enc2_w", seed=seed)
        self.enc2_b = var((8,),                 name="enc2_b", seed=seed)
        self.enc3_w = var((8, num_latents),     name="enc3_w", seed=seed)
        self.enc3_b = var((num_latents,),       name="enc3_b", seed=seed)
        
        # relation 
        num_wb_latents = 2 * num_latents
        relation_io = (num_wb_latents * self.num_k_shots, num_wb_latents)
        self.rel1_w = var(relation_io,         "rel1_w", seed=seed)
        self.rel1_b = var((num_wb_latents,),   "rel1_b", seed=seed)
        self.rel2_w = var((8, 64),             "rel2_w", seed=seed)
        self.rel2_b = var((64,),               "rel2_b", seed=seed)
        self.rel3_w = var((64, 2*num_latents), "rel3_w", seed=seed)
        self.rel3_b = var((2*num_latents,),    "rel3_b", seed=seed)
        
        """ EXP: decoder bias """
        # decoder 
        if self.ctx.gta_mode is 'kappa_angle':
            self.dec1_w = var((num_latents, 8), name="dec1_w", seed=seed)
            self.dec1_b = var((8,),             name="dec1_b", seed=seed)
            self.dec2_w = var((8, 8),           name="dec2_w", seed=seed)
            self.dec2_b = var((8,),             name="dec2_b", seed=seed)
            self.dec3_w = var((8, 2*theta_dims),name="dec3_w", seed=seed)
            self.dec3_b = var((2*theta_dims,),  name="dec3_b", seed=seed)
        else:
            self.dec1_w = var((num_latents, 16),  name="dec1_w", seed=seed)
            self.dec2_w = var((16, 64),           name="dec2_w", seed=seed)
            self.dec3_w = var((64, 2*theta_dims), name="dec3_w", seed=seed)

    
    """
        Gen System Pipeline
    """
    def forward_encoder(self, inputs):
        x = inputs
        x = dense(x, self.enc1_w, self.enc1_b)
        x = tf.nn.relu(x)
        x = dense(x, self.enc2_w, self.enc2_b)
        x = tf.nn.relu(x)
        x = dense(x, self.enc3_w, self.enc3_b)
        x = tf.nn.relu(x)            # history: tanh
        return x
    
    def forward_relation(self, encodes):
        # EXP. how about LSTM than this approach?
        x = tf.reshape(encodes, [self.num_k_shots, self.num_latents])
        left  = tf.tile(tf.expand_dims(x, 1), [1, self.num_k_shots, 1])
        right = tf.tile(tf.expand_dims(x, 0), [self.num_k_shots, 1, 1])
        codes = tf.concat([left, right], axis=-1)
        codes = tf.reshape(codes, [self.num_k_shots, 2*self.num_k_shots*self.num_latents])
        
        x = dense(codes, self.rel1_w, self.rel1_b) 
        x = tf.nn.relu(x)
        x = dense(x, self.rel2_w, self.rel2_b) 
        x = tf.nn.relu(x)
        x = dense(x, self.rel3_w, self.rel3_b) 
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, axis=0)
        x = tf.reshape(x, [-1, 2*self.num_latents])
        
        return x
    
    def forward_decoder(self, latents):
        # statistics of parameter 
        x = latents
        x = dense(x, self.dec1_w, self.dec1_b)
        x = tf.nn.relu(x)
        x = dense(x, self.dec2_w, self.dec2_b)
        x = tf.nn.relu(x)
        x = dense(x, self.dec3_w, self.dec3_b)
        statistics = tf.nn.tanh(x)
        
        # decide stddev offset (not fully understand but it seems 
        # to control inter-class distance in parameter space)
        fan_in = self.gen_theta_dims
        fan_out = self.num_classes
        stddev_offset = np.sqrt(2. / (fan_out + fan_in))
        
        # sample parameter from statistics
        mean_scale = 10.0 if self.ctx.gta_mode is 'kappa_angle' else None
        theta, kl = sample_dist(statistics, use_norm_std=False, mean_scale=mean_scale)
        return theta, kl
    
    
    """
        API
    """
    @tf.function(experimental_compile=True)
    def __call__(self, tasks, is_leo=True):
        self.is_leo = is_leo
        self.result_bag.clear()
        
        losses, genc_losses = [], []
        for task in tasks:
            if is_leo:
                genc_theta, genc_loss, kl, enc_penalty = self.gen_optimized_calibrator(task)
                loss = self.meta_train(genc_theta, task, kl, enc_penalty)
            else:
                loss = self.meta_train(None, task, 0, 0)
            losses.append(loss)
        return losses, self.result_bag       
    
    def train(self, data_provider, matrix, num_epochs, is_leo):
        for _ in range(num_epochs):
            for tasks in data_provider:
                loss, result = self(tasks, is_leo=is_leo)
                matrix.add(result, loss)
                matrix.next_batch()
            matrix.next_epoch()
        
    def evaluate(self, data_provider, use_cali_set, use_last_gen):
        self.result_bag.clear()
        gen_theta = self.last_genc_theta if use_last_gen else None
        for i, tasks in enumerate(data_provider):
            for task in tasks:
                self.predict_and_loss(task, gen_theta, use_cali_set=use_cali_set)
        return self.result_bag
         
        
    """
        Predictor
    """
    def predict_and_loss(self, task, gen_theta, use_cali_set):
        rctx = self.make_runtime_context(task, use_cali_set)
        self.predict_with(rctx, gen_theta)
        
        true_v_l, true_v_r = rctx.true_gaze_both
        pred_v_l, pred_v_r = rctx.pred_gaze_both
        pred_v_l, pred_v_r = tf.reshape(pred_v_l, (-1, 3)), tf.reshape(pred_v_l, (-1, 3))
        
        vec_loss_l = self.cosine_loss_fn(true_v_l, pred_v_l)
        vec_loss_r = self.cosine_loss_fn(true_v_r, pred_v_r)
        
        t_xy_loss_m = self.mae_loss_fn(rctx.true_pog_xy, rctx.pred_pog_xy)
        t_xy_loss_c = self.target_loss_fn(rctx.true_pog_xy, rctx.pred_pog_xy)
        t_xy_loss_c = tf.math.sqrt(t_xy_loss_c)
        t_xy_loss_l = self.target_loss_fn(rctx.true_pog_xy, rctx.pred_pog_xy_l)
        t_xy_loss_l = tf.math.sqrt(t_xy_loss_l)
        t_xy_loss_r = self.target_loss_fn(rctx.true_pog_xy, rctx.pred_pog_xy_r)
        t_xy_loss_r = tf.math.sqrt(t_xy_loss_r)
        
        """ TODO: NamedTuple """
        self.result_bag.append({'pred':rctx.pred_pog_xy, 'true':rctx.true_pog_xy, 'id': rctx.item_id})
        
        gaze_vec_losses = (vec_loss_l + vec_loss_r)
        target_xy_losses = ((t_xy_loss_c + t_xy_loss_l + t_xy_loss_r) / 2) + t_xy_loss_m
        
        return target_xy_losses + (10 * gaze_vec_losses)
    
    def predict_with(self, runtime_context, gen_theta=None):
        self.execute_eye_module(runtime_context)
        self.gen_theta_adaptation(gen_theta, runtime_context)
        self.compute_gaze_origin(runtime_context)
        self.calc_visual_axis(runtime_context)
        self.transform_face_rotation(runtime_context)
        self.compute_point_of_gaze(runtime_context)
        return runtime_context

    
    """
        Interpretable Algorithm
    """
    def execute_eye_module(self, rc):
        z, fv, penult_fv = self.eyenet(rc.eye_patch_dual)
        theta_phi_l, theta_phi_r, delta_xy_l, delta_xy_r = self.split_eye_z(z)
        
        rc.eye_module_z = z
        rc.eye_feature = fv
        rc.eye_feature_vector_penult = penult_fv
        rc.eye_pos_delta_xy_both = (delta_xy_l, delta_xy_r)
        rc.optical_theta_phi_both = (theta_phi_l, theta_phi_r)

    def gen_theta_adaptation(self, genc_theta, rc):   
        """ TODO Depth Regressor """
        angle_adapter = self.get_adaptation_strategy()
        rc.adapted_opt_angles = angle_adapter(genc_theta, rc)
            
    def random_effect_adapter(self, theta_mixed, rc):
        fixed_effect_z = rc.eye_module_z
        
        if theta_mixed is not None:
            input_fv = rc.eye_feature_vector_penult
            random_effect_z = dense_from(input_fv, theta_mixed, 16, 8, act=None, bias=False)
            mixed_z = fixed_effect_z + random_effect_z
        else:
            mixed_z = fixed_effect_z
            
        theta_phi_l, theta_phi_r, delta_xy_l, delta_xy_r = self.split_eye_z(mixed_z)
        theta_l, phi_l, theta_r, phi_r = self.split_theta_phi((theta_phi_l, theta_phi_r))
        
        rc.eye_pos_delta_xy_both = (delta_xy_l, delta_xy_r)
        adapted_opt_angles_l = (theta_l, phi_l, self.zero, self.zero)
        adapted_opt_angles_r = (theta_r, phi_r, self.zero, self.zero)
        return adapted_opt_angles_l, adapted_opt_angles_r
        
    def kappa_angle_adapter(self, genc_theta, rc):
        kappa_both = self.get_kappa_angle(genc_theta, rc, use_zeros=self.ctx.zero_kappa)
        alpha_l, beta_l, alpha_r, beta_r = kappa_both
        theta_l, phi_l,  theta_r, phi_r = self.split_theta_phi(rc.optical_theta_phi_both)
        
        adapted_opt_angles_l = (theta_l, phi_l, alpha_l, beta_l)
        adapted_opt_angles_r = (theta_r, phi_r, alpha_r, beta_r)
        return adapted_opt_angles_l, adapted_opt_angles_r
         
    def get_kappa_angle(self, theta_kappa, rc, use_zeros):
        if use_zeros:
            return (self.zero, self.zero, self.zero, self.zero)
        
        contextual_adaptation = False
        if theta_kappa is None:
            kappa_both = tf.reshape(self.kappa_alpha_beta_both, (-1,4))
            kappa_both = tf.tile(self.kappa_alpha_beta_both, [self.ctx.num_k_shots])
            kappa_both = tf.reshape(kappa_both, (-1, 4))
        elif contextual_adaptation:
            # face_R_flatten = tf.reshape(rc.face_R, (-1, 9))
            # eye_pos_l, eye_pos_r = rc.in_eye_pos_3d_both
            # opt_theta_phi_l, opt_theta_phi_r = rc.optical_theta_phi_both
            # opt_theta_l, opt_phi_l = tf.split(opt_theta_phi_l, [1,1], axis=1)
            # opt_theta_r, opt_phi_r = tf.split(opt_theta_phi_r, [1,1], axis=1)
            # input_k = [eye_pos_l, eye_pos_r, face_R_flatten, opt_theta_phi_l, opt_theta_phi_r]
            # inputs = tf.concat(input_tensors, axis=1)
            # kappa_both = dense_from(inputs, theta_kappa, 19, 4, tf.nn.tanh)
            pass
        else:
            kappa_both = theta_kappa    # Fixed Kappa 
            
        alpha_l, beta_l, alpha_r, beta_r = tf.split(kappa_both, [1,1,1,1], axis=1)
        return (deg2rad(alpha_l), deg2rad(beta_l), deg2rad(alpha_r), deg2rad(beta_r))

    def compute_gaze_origin(self, rc):
        face_R = tf.reshape(rc.face_R, (-1, 3, 3))
        t_vec = rc.t_vec
        eye_pos_l, eye_pos_r = rc.in_eye_pos_3d_both
        delta_xy_l, delta_xy_r = rc.eye_pos_delta_xy_both
        
        delta_pos_xy_l = face_R @ tf.reshape(delta_xy_l, (-1, 3, 1)) 
        delta_pos_xy_r = face_R @ tf.reshape(delta_xy_r, (-1, 3, 1))
        delta_pos_xy_l = tf.reshape(delta_pos_xy_l, (-1, 3))
        delta_pos_xy_r = tf.reshape(delta_pos_xy_r, (-1, 3))
        
        in_depth = tf.concat([t_vec, eye_pos_l, eye_pos_r], axis=1)
        delta_z_both = self.depth_regressor(in_depth)
        delta_z_l, delta_z_r = tf.split(delta_z_both, [1,1], axis=1)
        delta_pos_z_l = tf.concat([self.delta_zero_xy, delta_z_l], axis=1)
        delta_pos_z_r = tf.concat([self.delta_zero_xy, delta_z_r], axis=1)
        
        eye_stable_pos_l = eye_pos_l + delta_pos_xy_l + delta_pos_z_l
        eye_stable_pos_r = eye_pos_r + delta_pos_xy_r + delta_pos_z_r
        rc.stable_gaze_origin_both = (eye_stable_pos_l, eye_stable_pos_r)        
    
    def calc_visual_axis(self, rc):
        opt_angles_l, opt_angles_r = rc.adapted_opt_angles
        visual_axis_l = to_visual_axis(*opt_angles_l)
        visual_axis_r = to_visual_axis(*opt_angles_r)
        rc.visual_axis_both = (visual_axis_l, visual_axis_r)
    
    def transform_face_rotation(self, rc):
        """ visual_vec -> visual_gaze_vec """
        face_R = tf.reshape(rc.face_R, (-1, 3, 3))
        visual_axis_l, visual_axis_r = rc.visual_axis_both
        
        visual_axis_l = tf.reshape(visual_axis_l, (-1, 3, 1))
        visual_axis_r = tf.reshape(visual_axis_r, (-1, 3, 1))
        gaze_vec_l = face_R @ visual_axis_l
        gaze_vec_r = face_R @ visual_axis_r
        
        rc.gaze_vec_both = (gaze_vec_l, gaze_vec_r)

    def compute_point_of_gaze(self, rc):
        eye_pos_l, eye_pos_r = split_pair_vec(rc.stable_gaze_origin_both)
        gaze_vec_l, gaze_vec_r = split_pair_vec(rc.gaze_vec_both)
        
        target_l = calc_target(gaze_vec_l, eye_pos_l)
        target_r = calc_target(gaze_vec_r, eye_pos_r)
        target_mean = tf.reduce_mean([target_l, target_r], axis=0)
        
        rc.pred_gaze_both = (gaze_vec_l, gaze_vec_r)
        rc.pred_pog_xy   = drop_z_in_vec(target_mean, batch=True)
        rc.pred_pog_xy_l = drop_z_in_vec(target_l, batch=True)
        rc.pred_pog_xy_r = drop_z_in_vec(target_r, batch=True)
    
    
    """
        Helpers
    """
    def get_adaptation_strategy(self):
        mode = self.ctx.gta_mode
        if mode is 'mixed_effect':
            return self.random_effect_adapter
        if mode is 'kappa_angle':
            return self.kappa_angle_adapter
        return self.kappa_angle_adapter
        
    def split_eye_z(self, z):
        zz = tf.split(z, [2,2,2,2], axis=1)
        theta_phi_l, theta_phi_r, delta_xy_l, delta_xy_r = zz
        delta_xy_l = tf.concat([delta_xy_l, self.batch_zero], axis=1)
        delta_xy_r = tf.concat([delta_xy_r, self.batch_zero], axis=1)
        
        return theta_phi_l, theta_phi_r, delta_xy_l, delta_xy_r
    
    def split_theta_phi(self, opt_theta_phi_both):
        opt_theta_phi_l, opt_theta_phi_r = opt_theta_phi_both
        opt_theta_l, opt_phi_l = tf.split(opt_theta_phi_l, [1,1], axis=1)
        opt_theta_r, opt_phi_r = tf.split(opt_theta_phi_r, [1,1], axis=1)
        return opt_theta_l, opt_phi_l, opt_theta_r, opt_phi_r
    
    def make_runtime_context(self, task, use_cali_set=True):
        t = task
        ctx = Bunch()
        if use_cali_set:
            # ctx.frame = t.cal_frame
            ctx.eye_patch_both = (t.cal_le, t.cal_re)
            ctx.eye_patch_dual = t.cal_eye_dual
            ctx.t_vec = t.cal_t_vec
            ctx.in_eye_pos_3d_both = (t.cal_eye_pos_l, t.cal_eye_pos_r)
            ctx.face_R = t.cal_face_faze_R
            ctx.inv_nR = t.cal_inv_nR
            ctx.true_pog_xy = t.cal_t_xy_mm
            ctx.true_gaze_both = (t.cal_gaze_l, t.cal_gaze_r)
            ctx.true_ngaze_both = (t.cal_ngaze_l, t.cal_ngaze_r)
            ctx.pid = t.cal_pid
            ctx.item_id = t.cal_id
        else:
            # ctx.frame = t.val_frame
            ctx.eye_patch_both = (t.val_le, t.val_re)
            ctx.eye_patch_dual = t.val_eye_dual
            ctx.t_vec = t.val_t_vec
            ctx.in_eye_pos_3d_both = (t.val_eye_pos_l, t.val_eye_pos_r)
            ctx.face_R = t.val_face_faze_R
            ctx.inv_nR = t.val_inv_nR
            ctx.true_pog_xy = t.val_t_xy_mm
            ctx.true_gaze_both = (t.val_gaze_l, t.val_gaze_r)
            ctx.true_ngaze_both = (t.val_ngaze_l, t.val_ngaze_r)
            ctx.pid = t.val_pid
            ctx.item_id = t.val_id
        return ctx 


    """
        Latent Embedding Optimization (LEO)
    """
    def generate_latents(self, genc_inputs):
        # embeddings of each calibration samples
        embeddings = self.forward_encoder(genc_inputs)
        
        # statistics of latents 
        # (considering correlation over each calibration sample embeddings)
        statistics_for_latent = self.forward_relation(embeddings)
        
        # sample latents from statistics
        latents, kl = sample_dist(statistics_for_latent)
        
        return latents, kl
    
    def generate_theta(self, latents):
        gen_theta, kl = self.forward_decoder(latents)
        return gen_theta, kl

    def optimize_latents(self, task, base_latents):
        latents = base_latents
        theta, _ = self.generate_theta(latents)
        loss = self.predict_and_loss(task, theta, use_cali_set=True)
        
        for _ in range(self.ctx.num_latent_grad_steps):
            loss_grad = tf.gradients(loss, latents)
            latents -= self.latent_lr * loss_grad[0]
            theta, _ = self.generate_theta(latents)
            loss = self.predict_and_loss(task, theta, use_cali_set=True)
            
        penalty = 0.0 
        if self.is_meta_training:
            penalty = losses.mse(tf.stop_gradient(latents), base_latents)
        return loss, theta, tf.cast(penalty, tf.float32)
    
    def optimize_theta(self, task, loss, theta):
        for _ in range(self.ctx.num_finetune_grad_steps):
            loss_grad = tf.gradients(loss, theta)
            theta -= self.theta_lr * loss_grad[0]
            loss = self.predict_and_loss(task, theta, use_cali_set=True)
        return loss, theta

    def gen_optimized_calibrator(self, task):
        css = self.make_calibration_samples(task)
            
        # generate latents for calibrator functional
        latents_zero, kl = self.generate_latents(css)
        
        # optimize in latent embedding space 
        loss, theta, encoder_penalty = self.optimize_latents(task, latents_zero)
        
        # optimize directory in parmeter(θ) space 
        loss, gen_theta = self.optimize_theta(task, loss, theta)
        
        return gen_theta, loss, kl, encoder_penalty
    
    def make_calibration_samples(self, task):
        r = self.make_runtime_context(task, True)
        self.predict_with(r, None)
        
        fv = r.eye_feature   # r.eye_feature_vector_penult
        theta_phi_both = tf.reshape(r.optical_theta_phi_both, (-1, 4))
        err_target = r.true_pog_xy - r.pred_pog_xy
        pred_gaze_both = tf.reshape(r.pred_gaze_both, (-1, 6))
        true_gaze_both = tf.reshape(r.true_gaze_both, (-1, 6))
        err_vec = true_gaze_both - pred_gaze_both
        
        genc_input = tf.concat([fv, err_vec, err_target, theta_phi_both], axis=1)
        # print("genc_input shape:", genc_input.shape)
        
        return genc_input    

    def meta_train(self, genc_theta, task, kl, enc_penalty):
        task_loss = self.predict_and_loss(task, genc_theta, use_cali_set=False)
        if self.is_leo:
            # TODO: genc decoder orthogonality regularizer 
            genc_kl_loss = kl * self.ctx.kl_weight
            genc_enc_loss = enc_penalty * self.ctx.encoder_penalty_weight
            # tf.print(task_loss, genc_kl_loss, genc_enc_loss)
            loss = self.learn(task_loss + genc_kl_loss + genc_enc_loss)
        else:
            loss = self.learn(task_loss)
        return loss

    def learn(self, loss):
        grads, vars = self.grads_and_vars(loss)
        grads = clip_gradients(grads, 
                               self.ctx.gradient_threshold, 
                               self.ctx.gradient_norm_threshold) 
        self.optimizer.apply_gradients(list(zip(grads, vars)))
        return loss

        
    """
        Property
    """ 
    @property
    def l2_regularization(self):
        return tf.cast(tf.reduce_sum(tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)), dtype=tf.float32)

    def get_trainable_variables(self):
        eye_vars = self.trainable_variables
        # gen_vars = self.genc.trainable_variables
        t_vars = eye_vars # + gen_vars
        
        if self.ctx.num_finetune_grad_steps == 0:
            t_vars = filter_tensors(t_vars, 'finetuning_lr')
        if not self.is_leo:
            genc_names = [
                'leo/dec1_w:0', 'leo/dec2_w:0', 'leo/dec3_w:0', 
                'leo/enc1_w:0', 'leo/enc2_w:0', 'leo/enc3_w:0', 
                'leo/rel1_w:0', 'leo/rel2_w:0', 'leo/rel3_w:0',
                'leo/dec1_b:0', 'leo/dec2_b:0', 'leo/dec3_b:0', 
                'leo/enc1_b:0', 'leo/enc2_b:0', 'leo/enc3_b:0', 
                'leo/rel1_b:0', 'leo/rel2_b:0', 'leo/rel3_b:0',
                'leo/latent_lr:0', 'leo/theta_lr:0'
            ]
            
            t_vars = filter_tensors(t_vars, genc_names)
        if self.ctx.zero_kappa or self.ctx.gta_mode is 'mixed_effect':
            t_vars = filter_tensors(t_vars, ['leo/kappa_both:0'])
            
        return t_vars
        
    def grads_and_vars(self, loss):
        t_vars = self.get_trainable_variables()
        grads = tf.gradients(loss, t_vars)
        
        # print([(g,v) for g,v in zip(grads, t_vars)])

        broken_grads = [z[1].name for z in list(zip(grads, t_vars)) if z[0] is None]
        if len(broken_grads) != 0: 
            print("Check broken gradients: ", broken_grads)
        
        is_nan_loss = tf.math.is_nan(loss)
        is_nan_grad = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in grads])
        nan_loss_or_grad = tf.logical_or(is_nan_loss, is_nan_grad)
        
        reg_penalty = (1e-4 / self.ctx.l2_penalty_weight * self.l2_regularization)
        zero_or_regularization_gradients = [g if g is not None else tf.zeros_like(v)
            for v, g in zip(tf.gradients(reg_penalty, t_vars), t_vars)]

        grads = tf.cond(nan_loss_or_grad, lambda: zero_or_regularization_gradients, lambda: grads)
        return grads, t_vars
    