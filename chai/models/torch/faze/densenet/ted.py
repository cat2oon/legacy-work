"""
 Transform Encoder Decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from modelsummary import summary
from collections import OrderedDict


"""
 Helpers
"""
def embedding_consistency_loss_weight_at_step(current_step, config):
    final_value = config.coeff_embedding_consistency_loss
    if config.embedding_consistency_loss_warmup_samples is None:
        return final_value
    
    warmup_steps = int(config.embedding_consistency_loss_warmup_samples / config.batch_size)
    if current_step <= warmup_steps:
        return (final_value / warmup_steps) * current_step
    return final_value

"""
DenseNetInitialLayers
"""
class DenseNetInitialLayers(nn.Module):
    def __init__(self, 
                 growth_rate=8, 
                 activation_fn=nn.ReLU, 
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNetInitialLayers, self).__init__()
        
        c_next = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, c_next, bias=False, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        # self.norm = normalization_fn(c_next, track_running_stats=False).to(device)
        self.norm = normalization_fn(c_next, track_running_stats=False)
        self.act = activation_fn(inplace=True)

        c_out = 4 * growth_rate
        self.conv2 = nn.Conv2d(2 * growth_rate, c_out, bias=False, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        self.c_now = c_out
        self.c_list = [c_next, c_out]

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        prev_scale_x = x
        x = self.conv2(x)
        return x, prev_scale_x
    
    
"""
DenseNetCompositeLayer
"""
class DenseNetCompositeLayer(nn.Module):
    def __init__(self, 
                 c_in, 
                 c_out, 
                 kernel_size=3, 
                 growth_rate=8, 
                 p_dropout=0.1,
                 activation_fn=nn.ReLU, 
                 normalization_fn=nn.BatchNorm2d,
                 transposed=False):
        super(DenseNetCompositeLayer, self).__init__()
        
        # self.norm = normalization_fn(c_in, track_running_stats=False).to(device)
        self.norm = normalization_fn(c_in, track_running_stats=False)
        self.act = activation_fn(inplace=True)
        
        if transposed:
            assert kernel_size > 1
            conv_layer = nn.ConvTranspose2d
        else:
            conv_layer = nn.Conv2d
        
        self.conv = conv_layer(c_in, 
                               c_out, 
                               kernel_size=kernel_size,
                               padding=1 if kernel_size > 1 else 0,
                               stride=1, 
                               bias=False)
                               # bias=False).to(device)
            
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.drop = nn.Dropout2d(p=p_dropout) if p_dropout > 1e-5 else None
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        if self.drop is not None:
            x = self.drop(x)
        return x
   
"""
DenseNetBlock
"""
class DenseNetBlock(nn.Module):
    def __init__(self, 
                 c_in, 
                 num_layers=4, 
                 growth_rate=8, 
                 p_dropout=0.1,
                 use_bottleneck=False, 
                 activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d, 
                 transposed=False):
        super(DenseNetBlock, self).__init__()
        
        self.use_bottleneck = use_bottleneck
        c_now = c_in
        
        for i in range(num_layers):
            i_ = i + 1
            if use_bottleneck:
                bottleneck = DenseNetCompositeLayer(c_now, 
                                                    4 * growth_rate, 
                                                    kernel_size=1, 
                                                    p_dropout=p_dropout,
                                                    activation_fn=activation_fn,
                                                    normalization_fn=normalization_fn)
                self.add_module('bneck%d' % i_, bottleneck)
            
            c_in_compos = 4 * growth_rate if use_bottleneck else c_now
            composit_layer = DenseNetCompositeLayer(c_in_compos, 
                                                    growth_rate,
                                                    kernel_size=3, 
                                                    p_dropout=p_dropout,
                                                    activation_fn=activation_fn,
                                                    normalization_fn=normalization_fn,
                                                    transposed=transposed)
            self.add_module('compo%d' % i_, composit_layer)
            c_now += list(self.children())[-1].c_now
        self.c_now = c_now

    def forward(self, x):
        x_before = x
        for i, (name, module) in enumerate(self.named_children()):
            if ((self.use_bottleneck and name.startswith('bneck')) or name.startswith('compo')):
                x_before = x
            x = module(x)
            if name.startswith('compo'):
                x = torch.cat([x_before, x], dim=1)
        return x

"""
DenseNetTransitionDown
"""
class DenseNetTransitionDown(nn.Module):
    def __init__(self, 
                 c_in, 
                 compression_factor=0.1, 
                 p_dropout=0.1,
                 activation_fn=nn.ReLU, 
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNetTransitionDown, self).__init__()
        
        c_out = int(compression_factor * c_in)
        self.composite = DenseNetCompositeLayer(c_in, 
                                                c_out,         
                                                kernel_size=1, 
                                                p_dropout=p_dropout,            
                                                activation_fn=activation_fn,
                                                normalization_fn=normalization_fn)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c_now = c_out

    def forward(self, x):
        x = self.composite(x)
        x = self.pool(x)
        return x

"""
DenseNetTransitionUp
"""
class DenseNetTransitionUp(nn.Module):
    def __init__(self, 
                 c_in, 
                 compression_factor=0.1, 
                 p_dropout=0.1,
                 activation_fn=nn.ReLU, 
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNetTransitionUp, self).__init__()
        
        c_out = int(compression_factor * c_in)
        # self.norm = normalization_fn(c_in, track_running_stats=False).to(device)
        self.norm = normalization_fn(c_in, track_running_stats=False)
        self.act = activation_fn(inplace=True)
        self.conv = nn.ConvTranspose2d(c_in, 
                                       c_out, 
                                       kernel_size=3,
                                       stride=2, 
                                       padding=1, 
                                       output_padding=1,
                                       bias=False)
                                       # bias=False).to(device)
        
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.drop = nn.Dropout2d(p=p_dropout) if p_dropout > 1e-5 else None
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

"""
DenseNetDecoderLastLayers
"""
class DenseNetDecoderLastLayers(nn.Module):
    def __init__(self, 
                 c_in, 
                 growth_rate=8, 
                 activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d,
                 skip_connection_growth=0):
        super(DenseNetDecoderLastLayers, self).__init__()
        
        # First deconv
        self.conv1 = nn.ConvTranspose2d(c_in, 
                                        4 * growth_rate, 
                                        bias=False,
                                        kernel_size=3, 
                                        stride=2, 
                                        padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        # Second deconv
        c_in = 4 * growth_rate + skip_connection_growth
        # self.norm2 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.norm2 = normalization_fn(c_in, track_running_stats=False)
        self.act = activation_fn(inplace=True)
        self.conv2 = nn.ConvTranspose2d(c_in, 
                                        2 * growth_rate, 
                                        bias=False,
                                        kernel_size=3, 
                                        stride=2, 
                                        padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        # Final conv
        c_in = 2 * growth_rate
        c_out = 3
        # self.norm3 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.norm3 = normalization_fn(c_in, track_running_stats=False)
        self.conv3 = nn.Conv2d(c_in, 
                               c_out, 
                               bias=False,
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        self.c_now = c_out

    def forward(self, x):
        x = self.conv1(x)
        #
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        #
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv3(x)
        return x

"""
DenseNetEncoder
"""
class DenseNetEncoder(nn.Module):
    def __init__(self, 
                 growth_rate=8, 
                 num_blocks=4, 
                 num_layers_per_block=4,
                 p_dropout=0.0, 
                 compression_factor=1.0,
                 activation_fn=nn.ReLU, 
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNetEncoder, self).__init__()
        
        self.c_at_end_of_each_scale = []

        # Initial down-sampling conv layers
        self.initial = DenseNetInitialLayers(growth_rate=growth_rate,
                                             activation_fn=activation_fn,
                                             normalization_fn=normalization_fn)
        c_now = list(self.children())[-1].c_now
        self.c_at_end_of_each_scale += list(self.children())[-1].c_list

        assert (num_layers_per_block % 2) == 0
        for i in range(num_blocks):
            i_ = i + 1
            dense_block = DenseNetBlock(c_now,
                                       num_layers=num_layers_per_block,
                                       growth_rate=growth_rate,                       
                                       p_dropout=p_dropout,
                                       activation_fn=activation_fn,
                                       normalization_fn=normalization_fn)
            self.add_module('block%d' % i_, dense_block)
            c_now = list(self.children())[-1].c_now
            self.c_at_end_of_each_scale.append(c_now)

            if i < (num_blocks - 1):  # transition block if not last layer
                dense_trans_down = DenseNetTransitionDown(c_now,
                                                          p_dropout=p_dropout,
                                                          compression_factor=compression_factor,
                                                          activation_fn=activation_fn,
                                                          normalization_fn=normalization_fn)
                self.add_module('trans%d' % i_, dense_trans_down)
                c_now = list(self.children())[-1].c_now
            self.c_now = c_now

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            if name == 'initial':
                x, prev_scale_x = module(x)
            else:
                x = module(x)
        return x

    
"""
DenseNetDecoder
"""
class DenseNetDecoder(nn.Module):
    def __init__(self, 
                 c_in, 
                 growth_rate=8, 
                 num_blocks=4, 
                 num_layers_per_block=4,
                 p_dropout=0.0, 
                 compression_factor=1.0,
                 activation_fn=nn.ReLU, 
                 normalization_fn=nn.BatchNorm2d,
                 use_skip_connections_from=None):
        super(DenseNetDecoder, self).__init__()

        self.use_skip_connections = (use_skip_connections_from is not None)
        if self.use_skip_connections:
            c_to_concat = use_skip_connections_from.c_at_end_of_each_scale
            c_to_concat = list(reversed(c_to_concat))[1:]
        else:
            c_to_concat = [0] * (num_blocks + 2)

        assert (num_layers_per_block % 2) == 0
        c_now = c_in
        for i in range(num_blocks):
            i_ = i + 1
            dense_block = DenseNetBlock(c_now,
                                        num_layers=num_layers_per_block,
                                        growth_rate=growth_rate,
                                        p_dropout=p_dropout,
                                        activation_fn=activation_fn,
                                        normalization_fn=normalization_fn,
                                        transposed=True)
            self.add_module('block%d' % i_, dense_block)
            c_now = list(self.children())[-1].c_now
            
            if i < (num_blocks - 1):    # transn block if not last layer
                dense_trans_up = DenseNetTransitionUp(c_now, 
                                                      p_dropout=p_dropout,
                                                      compression_factor=compression_factor,
                                                      activation_fn=activation_fn,
                                                      normalization_fn=normalization_fn)
                self.add_module('trans%d' % i_, dense_trans_up)
                c_now = list(self.children())[-1].c_now
                c_now += c_to_concat[i]

        # Last up-sampling conv layers
        self.last = DenseNetDecoderLastLayers(c_now,
                                              growth_rate=growth_rate,
                                              activation_fn=activation_fn,
                                              normalization_fn=normalization_fn,
                                              skip_connection_growth=c_to_concat[-1])
        self.c_now = 1

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            x = module(x)
        return x

"""
TED
"""
class TED(nn.Module):
    def __init__(self, 
                 z_dim_app=64, 
                 z_dim_gaze=2, 
                 z_dim_head=16,
                 growth_rate=32, 
                 activation_fn=nn.LeakyReLU,
                 normalization_fn=nn.InstanceNorm2d,
                 decoder_input_c=16,
                 normalize_3d_codes=False,
                 normalize_3d_codes_axis=None,
                 use_triplet=False,
                 gaze_hidden_layer_neurons=64,
                 backprop_gaze_to_encoder=False):
        super(TED, self).__init__()

        self.use_triplet = use_triplet
        self.decoder_input_c = decoder_input_c
        self.normalize_3d_codes = normalize_3d_codes
        self.normalize_3d_codes_axis = normalize_3d_codes_axis
        self.backprop_gaze_to_encoder = backprop_gaze_to_encoder
        if self.normalize_3d_codes:
            assert self.normalize_3d_codes_axis is not None

        # Define feature map dimensions at bottleneck
        bottleneck_shape = (2, 8)
        self.bottleneck_shape = bottleneck_shape

        # Encoder
        self.encoder = DenseNetEncoder(num_blocks=4,
                                       growth_rate=growth_rate,
                                       activation_fn=activation_fn,
                                       normalization_fn=normalization_fn)
        
        c_now = list(self.children())[-1].c_now
        enc_num_all = np.prod(bottleneck_shape) * decoder_input_c
        
        # Decoder
        self.decoder = DenseNetDecoder(decoder_input_c,
                                       num_blocks=4,
                                       growth_rate=growth_rate,
                                       activation_fn=activation_fn,
                                       normalization_fn=normalization_fn,
                                       compression_factor=1.0)

        # The latent code parts
        self.z_dim_app = z_dim_app
        self.z_dim_gaze = z_dim_gaze
        self.z_dim_head = z_dim_head
        
        """
        1) fc_enc 
        인코더 내부에 마지막에 fc_enc 호출하여 잠재 변수를 입력으로 받아 z_num_all 차원으로 fc 레이어 출력
        
        2) gaze layer
        3 * 2를 입력받아서 64개 dense 후 길이 3짜리 하나의 시선 벡터를 만듬
        => 이것은 양안이 아닌 하나의 양안중심 시선을 추적하겠다는 것
           어쩌면 이런 선택은 NVIDA가 양안 시선 맺힘에 대한 지식이 있어서 이런 결정을 했을지도 (안구 시선 조사해보기)
          고민1) 단안의 안축 계산 후 중심 시축 계산이 좋을 지? 
          고민2) 양안 중심 시축이 CNN 동공 피쳐를 명확하게 잡는데 도움이 될까?
              -> 얼굴을 고정한채 눈을 돌려보면 오히려 우세안 가중치도 잡고 더 도움이 될 수도...
              -> 하지만 이러면 명시적 카파 변환을 넣기 어려워짐. 
              -> 다만 여기에 head pose를 추가로 넣어주면 암묵적 카파 변환을 학습할 수 있으려나?
              
        """
        z_num_all = 3 * (z_dim_gaze + z_dim_head) + z_dim_app
        self.fc_enc = self.linear(c_now, z_num_all)
        self.fc_dec = self.linear(z_num_all, enc_num_all)
        self.build_gaze_layers(3 * z_dim_gaze, gaze_hidden_layer_neurons)

    def build_gaze_layers(self, num_input_neurons, num_hidden_neurons=64):
        self.gaze1 = self.linear(num_input_neurons, num_hidden_neurons)  # 1층
        self.gaze2 = self.linear(num_hidden_neurons, 3)                  # 2층 

    def linear(self, f_in, f_out):
        fc = nn.Linear(f_in, f_out)
        nn.init.kaiming_normal(fc.weight.data)
        nn.init.constant(fc.bias.data, val=0)
        return fc

    def rotate_code(self, data, code, mode, fr=None, to=None):
        """
        - Mode는 gaze, head 선택자
        - pitch-yaw (theta, phi)
        def R_x(theta):
            return np.array([
                [1., 0.,    0.],
                [0., cos, -sin],
                [0., sin,  cos]
            ]). 

        def R_y(phi):
            return np.array([
                [cos,  0., sin],
                [0.,   1., 0.],
                [-sin, 0., cos]
            ])
            
        def calculate_rotation_matrix(e):
            return np.matmul(R_y(e[1]), R_x(e[0]))
        """
        
        """ Must calculate transposed rotation matrices to be able to post-multiply to 3D codes """
        key_stem = 'rot_' + mode
        if fr is not None and to is not None:
            rotate_mat = torch.matmul(data[key_stem + '_' + fr], torch.transpose(data[key_stem + '_' + to], 1, 2))
        elif to is not None:
            rotate_mat = torch.transpose(data[key_stem + '_' + to], 1, 2)
        elif fr is not None:
            # transpose-of-inverse is itself
            rotate_mat = data[key_stem + '_' + fr]
        return torch.matmul(code, rotate_mat)

    def encode_to_z(self, data, suffix):
        """
        - 이미지 페어에서 쌍 순서를 Suffix로 지정하고 인코더를 통해 latent 추출
        - Suffix { a | b } -> 그냥 동일 프로파일 내의 임의의 이미지 x1, x2로 하면 안되나?
           - 데이터셋 생성 과정에서 a, b 할당 정책 조사:
           - park's preprocessing 과정과 관계없고 pytorch dataset (data.py)에서 할당
        """
        x = self.encoder(data['image_' + suffix])
        enc_output_shape = x.shape
        x = x.mean(-1).mean(-1)  # Global-Average Pooling   # CHW에서 W, H에 대해 평균 내버림. 결과 (batch, channels)

        """
        1) z_all = fc_enc(x) 
        - GAP 결과 채널수를 FC로 3 * (z_dim_gaze + z_dim_head) + z_dim_app 크기로 펼침 (batch, z_all)
        
        """
        # Create latent codes
        z_all = self.fc_enc(x)
        z_app = z_all[:, :self.z_dim_app]
        z_all = z_all[:, self.z_dim_app:]
        z_all = z_all.view(self.batch_size, -1, 3)   # 3배수 한 것대로 묶어주기
        z_gaze_enc = z_all[:, :self.z_dim_gaze, :]   # GAZE
        z_head_enc = z_all[:, self.z_dim_gaze:, :]   # HEAD

        z_gaze_enc = z_gaze_enc.view(self.batch_size, -1, 3)
        z_head_enc = z_head_enc.view(self.batch_size, -1, 3)
        return [z_app, z_gaze_enc, z_head_enc, x, enc_output_shape]  # (z_a_a, ze1_g_a, ze1_h_a, ze1_before_z_a, _)

    def decode_to_image(self, codes):
        z_all = torch.cat([code.view(self.batch_size, -1) for code in codes], dim=1)
        x = self.fc_dec(z_all)
        x = x.view(self.batch_size, self.decoder_input_c, *self.bottleneck_shape)
        x = self.decoder(x)
        return x

    def maybe_do_norm(self, code):
        if self.normalize_3d_codes:
            norm_axis = self.normalize_3d_codes_axis
            assert code.dim() == 3
            assert code.shape[-1] == 3
            if norm_axis == 3:
                b, f, _ = code.shape
                code = code.view(b, -1)
                normalized_code = F.normalize(code, dim=-1)
                return normalized_code.view(b, f, -1)
            else:
                return F.normalize(code, dim=norm_axis)
        return code

    def forward(self, data, loss_functions=None):
        is_inference_time = ('image_b' not in data)
        self.batch_size = data['image_a'].shape[0]

        """
        -> return of encode_to_z =>[z_app, z_gaze_enc, z_head_enc, x, enc_output_shape]
        ze1_before_z_a 안쓰는 구만 왜 뽑냐 헷갈리게
        """
        # Encode input from a 
        (z_a_a, ze1_g_a, ze1_h_a, ze1_before_z_a, _) = self.encode_to_z(data, 'a')
        if not is_inference_time:
            z_a_b, ze1_g_b, ze1_h_b, _, _ = self.encode_to_z(data, 'b')

        """
        Latent 벡터 정규화 구에 임베딩 (x, y, z)이라 이렇게 해주면 당연히 좋을 거라 생각은 들지만
        -> 논문에 이거에 대한 실험 결과 차이를 써주면 좋겠는데 없는 듯
        """
        # Make each row a unit vector through L2 normalization to constrain
        # embeddings to the surface of a hypersphere
        if self.normalize_3d_codes:
            assert ze1_g_a.dim() == ze1_h_a.dim() == 3
            assert ze1_g_a.shape[-1] == ze1_h_a.shape[-1] == 3
            ze1_g_a = self.maybe_do_norm(ze1_g_a)
            ze1_h_a = self.maybe_do_norm(ze1_h_a)
            if not is_inference_time:
                ze1_g_b = self.maybe_do_norm(ze1_g_b)
                ze1_h_b = self.maybe_do_norm(ze1_h_b)

        """
        그래디언트 흘려 보낼지 결정
        -> Loss에서 활용되는지 조사할 것:
        -> 왜 클론 하는거야?? 
        """
        # Gaze estimation output for image a
        if self.backprop_gaze_to_encoder:
            gaze_features = ze1_g_a.clone().view(self.batch_size, -1)
        else:
            # Detach input embeddings from graph!
            gaze_features = ze1_g_a.detach().view(self.batch_size, -1)    
        gaze_a_hat = self.gaze2(F.relu_(self.gaze1(gaze_features)))       # gaze_1층 RELU gaze_2층임
        gaze_a_hat = F.normalize(gaze_a_hat, dim=-1)

        
        """
        모델 출력 결과
        """
        output_dict = {
            'gaze_a_hat': gaze_a_hat,
            'z_app': z_a_a,
            'z_gaze_enc': ze1_g_a,
            'z_head_enc': ze1_h_a,
            'canon_z_gaze_a': self.rotate_code(data, ze1_g_a, 'gaze', fr='a'),
            'canon_z_head_a': self.rotate_code(data, ze1_h_a, 'head', fr='a'),
        }
        
        """
        함수로 만들어 두자 -> 그리고 entry 요소들의 이름을 enum 형태로 해야 겠다
        """
        if 'rot_gaze_b' not in data:
            return output_dict

        if not is_inference_time:
            output_dict['canon_z_gaze_b'] = self.rotate_code(data, ze1_g_b, 'gaze', fr='b')
            output_dict['canon_z_head_b'] = self.rotate_code(data, ze1_h_b, 'head', fr='b')

        # Rotate codes
        zd1_g_b = self.rotate_code(data, ze1_g_a, 'gaze', fr='a', to='b')
        zd1_h_b = self.rotate_code(data, ze1_h_a, 'head', fr='a', to='b')
        output_dict['z_gaze_dec'] = zd1_g_b
        output_dict['z_head_dec'] = zd1_h_b

        # Reconstruct
        x_b_hat = self.decode_to_image([z_a_a, zd1_g_b, zd1_h_b])
        output_dict['image_b_hat'] = x_b_hat

        # If loss functions specified, apply them
        if loss_functions is not None:
            losses_dict = OrderedDict()
            for key, func in loss_functions.items():
                losses = func(data, output_dict)  # may be dict or single value
                if isinstance(losses, dict):
                    for sub_key, loss in losses.items():
                        losses_dict[key + '_' + sub_key] = loss
                else:
                    losses_dict[key] = losses
            return output_dict, losses_dict

        return output_dict