from cheat.tfs import *


"""
    GenC System
    Generate parameter(θ) for personal real-time context adjust calibrator
    
    TODO:
    - l2 / ortho regularizer loss orthogonality_regularize_term
"""
class GenC(tf.Module):
    
    def __init__(self, ctx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_props(ctx)
        with self.name_scope:
            self.make()
        
    """
        Buidling
    """
    def setup_props(self, ctx):
        self.ctx = ctx
        self.seed = ctx.seed
        self.init_lr = ctx.latent_lr
        self.l2_penalty = 1e-2
        self.num_classes = 50
        self.num_latents = ctx.num_latents
        self.num_k_shots = ctx.num_k_shots
        self.gen_theta_dims = ctx.gen_theta_dims
        
    def make(self):
        num_latents = self.num_latents
        theta_dims = self.gen_theta_dims
        var, seed = var_with_shape, self.seed
        
        """ WARN: cast(trainable variable) -> lose trainability """
        """ TODO: separate config theta_lr, latent_lr """
        # latent learning rate 
        self.theta_lr  = var_tile(self.init_lr, [1, theta_dims],  name='theta_lr')
        self.latent_lr = var_tile(self.init_lr, [1, num_latents], name='latent_lr')
        
        """ EXP: bias + mlp """
        # encoder
        num_genc_input = 31   # quick: make_calibration_samples()
        self.enc1_w = var((num_genc_input, 128), name="enc1_w", seed=seed)
        self.enc2_w = var((128, 128),            name="enc2_w", seed=seed)
        self.enc3_w = var((128, num_latents),    name="enc3_w", seed=seed)
        
        # relation 
        relation_in = (num_latents * self.num_k_shots, 128)
        self.rel1_w = var(relation_in, "rel1_w", seed=seed)
        self.rel2_w = var((128, 128),  "rel2_w", seed=seed)
        self.rel3_w = var((128, 2*num_latents),  "rel3_w", seed=seed)
        
        """ EXP: decoder bias """
        # decoder 
        self.dec1_w = var((num_latents, 128),  name="dec1_w", seed=seed)
        self.dec2_w = var((128, 512),          name="dec2_w", seed=seed)
        self.dec3_w = var((512, 2*theta_dims), name="dec3_w", seed=seed)
    
    """
        APIs
    """
    def __call__(self, x):
        """ encode+relation의 결과로 생성된 theta 공간에서 최적화가 필요하기 때문에
        call을 호출할 수가 없음, 일반 함수를 호출하여도 module이 operation을 추적하는지 확인할 것 """
        pass
    
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
    
    
    """
        Pipeline
    """
    def forward_encoder(self, inputs):
        x = inputs
        x = dense(x, self.enc1_w)
        x = tf.nn.dropout(x, 0.5)
        x = tf.nn.selu(x)
        x = dense(x, self.enc2_w)
        x = tf.nn.dropout(x, 0.5)
        x = tf.nn.selu(x)
        x = dense(x, self.enc3_w)
        x = tf.nn.dropout(x, 0.3)
        x = tf.nn.selu(x)
        return x
    
    def forward_relation(self, encodes):
        # EXP. how about LSTM than this approach?
        x = tf.reshape(encodes, [1, np.prod(encodes.shape)])
        x = dense(x, self.rel1_w) 
        x = tf.nn.dropout(x, 0.5)
        x = tf.nn.selu(x)
        x = dense(x, self.rel2_w) 
        x = tf.nn.dropout(x, 0.5)
        x = tf.nn.selu(x)
        x = dense(x, self.rel3_w) 
        x = tf.nn.dropout(x, 0.3)
        x = tf.nn.tanh(x) 
        return x
    
    def forward_decoder(self, latents):
        # statistics of parameter 
        x = latents
        x = dense(x, self.dec1_w)
        x = dense(x, self.dec2_w)
        s = dense(x, self.dec3_w)
        
        # decide stddev offset (not fully understand but it seems 
        # to control inter-class distance in parameter space)
        fan_in = self.gen_theta_dims
        fan_out = self.num_classes
        stddev_offset = np.sqrt(2. / (fan_out + fan_in))
        
        # sample parameter from statistics
        theta, kl = sample_dist(s, stddev_offset)
        return theta, kl
