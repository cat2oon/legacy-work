#!/usr/bin/env python
# coding: utf-8
import torch
import matplotlib.pyplot as pp

from train import *
from sgnet import *
from generator.generator import *
from torchsummary import summary

import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO) 


"""
    Configuration
"""
def get_mock_config():
    return {
        'base_lr': 0.0005,
        'num_data_loaders' : 1,
        'num_training_epochs' : 20,
        'batch_size' : 64,
        'pin_memory': False,
        'use_apex' : False,
        'warmup_period_for_lr' : 1000000, 
        'decay_interval' : 0,
        'decay' : 0.8,
        'l2_reg' : 1e-4,
        'resource_path': '../../../../../data-archive/faze-resources/',
        'npz_root_path': '../../../../../data-archive/faze-recode-profile-npz/',
        # 'resource_path': 'J:\\datasets\\faze-resources',
        # 'npz_root_path': 'J:\\datasets\\everyone-recode',
    }


"""
 TODO:
- eye rect augmentation (GG 논문 기반)
- PGD (SS 논문 기반 - adversarial attack)
- MAML, TAT, meta+metric 최신기법들 중 하나 선정해서 실행
- Ordinal Loss (이것보다는 한 점 응시가 더 나을 듯)
- 모델 아키텍쳐 변경
- 배치 사이즈 점진적 증가
- CAM 활성화 맵 시각화 하기
- Eye 영역 크기 변경해보기
- Residual Connection
- mse, me 1.0에서 경계 

- final txmn 레이어 실험
- valid, test keras 처럼 돌리게 하기
- epoch loss 
- checkpoint save
- metric 출력 (실제 거리로)
- tensorboard 일정 주기로 기록하기

 Dataset 기반
- 왜 데이터 로더 오래 걸리는지 체크
- Eye 영역 못 찾는 것 개선 및 중심 위치 개선
- vc-One 데이터셋 적용
- vc-one scene3 temporal 적용
"""

if __name__ == "__main__":

    """
    Build Context
    """
    config = get_mock_config()
    Context.build(config)
    ctx = Context.get()

    """
    Build Network
    """
    network_key = "gg"
    INPUT_SIZE = (3, 64, 64)
    network = SGNet(INPUT_SIZE, network_key)
    
    ctx.set_network(network_key, network)
    ctx.load_network_to_device()

    """
    Loss And Optimizer
    """
    ctx.setup_base_loss_policy(verbose=True)
    optimizer = ctx.build_optimizer(network_key)
    ctx.set_main_optimizer(optimizer)

    loss_functions = ctx.make_loss_functions()
    ctx.set_loss_functions(loss_functions)

    """
    Data Generator
    """
    # gen = NPZDatasetGenerator(ctx, shuffle_train=False, item_selector_fn=selector)
    gen = NPZDatasetGenerator(ctx, shuffle_train=False)
    databag = gen.generate(verbose=True)
    ctx.set_databag(databag)
    train_ds, loader = ctx.get_dataset_and_loader(Context.TAG_TRAIN)

    """
    Train
    """
    ctx.use_multiple_gpu_if_available()
    ctx.run_train(network_key)
    
    
