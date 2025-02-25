import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
    SGNet
"""    
class SGNet(torch.nn.Module):
    
    def __init__(self, input_size, network_key):
        super(SGNet, self).__init__()
        self.input_size = input_size
        self.network_key = network_key
        self.build()
        
    def build(self):
        if self.network_key == 'ss':
            self._build_ss()
        elif self.network_key == 'gg':
            self._build_gg()   
    
    def forward(self, input_dict, loss_fn):
        if self.network_key == 'ss':
            return self._forward_ss(input_dict, loss_fn)
        elif self.network_key == 'gg':
            return self._forward_gg(input_dict, loss_fn)
        return None
            
    """
        SS
    """
    def _build_ss(self):
        self.le_net = SimpleNet(self.input_size)
        self.re_net = SimpleNet(self.input_size)
        self.fe_net = SimpleNet(self.input_size)
        
        self.fc1 = nn.Linear(384, 128)
    
    def _forward_ss(self, input_dict, loss_fn=None):
        fe = input_dict['face']
        le = input_dict['left_eye']
        re = input_dict['right_eye']
        
        le_fv = self.le_net(le, loss_fn)
        re_fv = self.re_net(re, loss_fn)
        fe_fv = self.fe_net(fe, loss_fn)
        
        x = torch.cat([le_fv, re_fv, fe_fv], dim=1)
        x = self.fc1(x)
        
        return x
         
    """
        GG
    """
    def _loss_gg(self, x, input_dict):
        loss_fn = nn.MSELoss(reduction='mean')
        return {
            'target_loss': loss_fn(x, input_dict['target_xy'])
        }
    
    def pixel_to_uv(self, x, y, fx, fy, cx, cy):
        u = (x - cx) / fx
        v = (y - cy) / fy
        return u, v
    
    def ec_txmn(self, ec, cm):
        # eye corner landmark [ x1 x2 x3 x4 y1 y2 y3 y4 ], camera-mat
        fx, fy = cm[:, 0, 0], cm[:, 1, 1]
        cx, cy = cm[:, 0, 2], cm[:, 1, 2]
        fx, fy = fx.view(-1, 1), fy.view(-1, 1)
        cx, cy = cx.view(-1, 1), cy.view(-1, 1)
        fx, fy = fx.repeat(1,4), fy.repeat(1,4)   # batch_size x [fx, fx, fx, fx], ...
        cx, cy = cx.repeat(1,4), cy.repeat(1,4)   # ...
        xs = ec[:, 0:4]    
        ys = ec[:, 4:]
        xs = (xs - cx) / fx    # uv normalize
        ys = (ys - cy) / fy    # uv normalize
        
        return torch.cat([xs, ys], dim=1)

    def final_txmn(self, xy, loc, ori):
        # loc [ DeviceCameraToScreenXMm(수평), DeviceCameraToScreenYMm(수직), DeviceScreenWidthMm, DeviceScreenHeightMm ]
        # bxy, lxy = loc[:, 0:2], loc[:, 2:4]
        # xy = torch.add(xy, bxy)
        
        # 오리엔테이션에 다른 비대칭 깨짐 현상 복원해주기
        # print("---ESTI---")
        # xy = torch.add(xy, bxy)
        
        return xy
        
    def _build_gg(self):
        self.eyenet = GimpleNet(self.input_size)   
        
        # Eye Corner Landmark 
        self.ec_fc1 = nn.Linear(8, 100)
        self.ec_fc2 = nn.Linear(100, 16)
        self.ec_fc3 = nn.Linear(16, 16)
        
        # Final
        self.final_fc4 = nn.Linear(32*2+16, 16)
        self.final_fc5 = nn.Linear(16, 2)

    def _forward_gg(self, input_dict, loss_fn=None):
        le = input_dict['left_eye']
        re = input_dict['right_eye']
        ec = input_dict['eye_corner']
        cm = input_dict['cam_mat']
        loc = input_dict['loc']
        ori = input_dict['orientation']
        
        ecx = self.ec_txmn(ec, cm)
        ecx = self.ec_fc1(ecx)
        ecx = self.ec_fc2(ecx)
        ecx = self.ec_fc3(ecx)
        
        lex = self.eyenet(le)
        rex = self.eyenet(re)
        
        x = torch.cat([lex, rex, ecx], dim=1)
        x = self.final_fc4(x)
        x = self.final_fc5(x)
        x = self.final_txmn(x, loc, ori)
        
        # print("---TARGET---")
        # print(input_dict['target_xy'])
        
        return x, self._loss_gg(x, input_dict)
   
"""
    GimpleNet
"""
class GimpleNet(torch.nn.Module):
    
    def __init__(self, input_size):
        super(GimpleNet, self).__init__()
        self.input_sizse = input_size
        self.build(input_size)

    def build(self, input_size):
        inc = input_size[0]
        self.activation = nn.PReLU()
        self.avg_pool1= nn.AvgPool2d(kernel_size=7)
        self.avg_pool2= nn.AvgPool2d(kernel_size=5)
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=32, kernel_size=7, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=5, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, padding=1) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = x.view(-1, 32)
        return x   
    
    
"""
    SimpleNet
"""    
class SimpleNet(torch.nn.Module):
    
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.input_sizse = input_size
        self.build(input_size)
        
    def _conv(self, inc, outc, kern, pad):
        return nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kern, padding=pad) 

    def build(self, input_size):
        num_in_channel = input_size[0]
        
        # Default Activation
        self.activation = nn.PReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=3)

        # Layer 1
        self.conv11 = self._conv(inc=num_in_channel, outc=32, kern=3, pad=1)
        self.conv12 = self._conv(inc=32,             outc=64, kern=3, pad=1)

        # Layer 2
        self.conv21 = self._conv(inc=64, outc=64,  kern=3, pad=1)
        self.conv22 = self._conv(inc=64, outc=128, kern=3, pad=1)

        # Layer 3
        self.conv31 = self._conv(inc=128, outc=96,  kern=3, pad=1)
        self.conv32 = self._conv(inc=96,  outc=192, kern=3, pad=1)

        # Lyaer 4
        self.conv41 = self._conv(inc=192, outc=128,  kern=3, pad=1)
        self.conv42 = self._conv(inc=128, outc=256,  kern=3, pad=1)

        # Layer 5
        self.conv51 = self._conv(inc=256, outc=160,  kern=3, pad=1)
        self.conv52 = self._conv(inc=160, outc=320,  kern=3, pad=1)
        self.conv52_dropout = nn.Dropout2d(p=0.40)

        # FC Layer 
        self.fc6 = nn.Linear(320, 128) 
        
    def forward(self, x, loss_fn=None):
        # Layer 1
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.max_pool(x)
        x = self.activation(x)
        
        # Layer 2
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.max_pool(x)
        x = self.activation(x)
        
        # Layer 3
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.max_pool(x)
        x = self.activation(x)
        
        # Layer 4
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.max_pool(x)
        x = self.activation(x)
        
        # Layer 5
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.avg_pool(x)
        x = self.conv52_dropout(x)
                        
        # Layer 6 (FC)
        x = x.view(-1, 320)
        x = self.fc6(x)
        x = F.log_softmax(x, dim=1)
        
        return x

    
    
    
"""
    Trash
"""
"""
def final_txmn_ex01(self, xy, loc):
    loc [ DeviceCameraToScreenXMm(수평), DeviceCameraToScreenYMm(수직), DeviceScreenWidthMm, DeviceScreenHeightMm ]
        # 근데 이거를 bias + range 보간처럼 쓰려면 orientation 레이블을 사용했어야 했을 텐데?
        # 아 이걸로 orientation을 추론하려고 이렇게 한거였구나? -> 그게 되나???
        # 아 이전 레이어에서 각도와 far로 축약 유도하려고 이렇게 한거였구나? 마치 해석가능구조에서 transform층 역할인거군
        # -> 근데 예전에 이렇게 했을 때 별로 였었는데... AutoML 모델이라서 다를 수도 
        #    -> win them all PR 영상보면 과도하게 큰 모델이 성능이 낮은 현상으로 설명되려나...
        # 아마도 이 방식이라면 전체 훈련 성능은 좋지 않고 few shot adaptation에서는 좋을 것으로 보임
        r, theta = xy[:, 0], xy[:, 1]
        bx, by, lx, ly = loc[:, 0], loc[:, 1], loc[:, 2], loc[:, 3]
        
#        print("----")
#        print(r, theta)
#        print(bx, by)
#        print(lx, ly)
#        print("----")
        
        dist_xy = torch.stack([bx+r*lx, by+r*ly], dim=0)
        dist_xy = torch.t(dist_xy).view(-1, 2, 1)
        
#        print("----")
#        print(dist_xy.shape)
#        print(dist_xy)
#        print("----")
        
        # 한번에 생성하는 함수 없으려나... 
        # rot_mat = torch.stack([
        #     [torch.cos(theta), -torch.sin(theta)], 
        #     [torch.sin(theta),  torch.cos(theta)]
        # ])
        
        r00 =  torch.cos(theta).view(-1, 1)
        r10 =  torch.sin(theta).view(-1, 1)
        r01 = -r10
        r11 =  r00
        r0 = torch.cat([r00, r01], dim=1)
        r1 = torch.cat([r10, r11], dim=1)
        rot_mat = torch.stack([r0, r1], dim=1)
        
#        print("----")
#        print(rot_mat.shape)
#        print(rot_mat)
#        print("----")
        
        xy = torch.bmm(rot_mat, dist_xy).view(-1, 2)
#        print(xy)
"""
