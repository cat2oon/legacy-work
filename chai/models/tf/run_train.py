#!/usr/bin/env python
# coding: utf-8
import warnings

from sgnet import *
from trainer import *
from trainer_maml import *
from gazecapture_sequence import *
from gazecapture_k_shots_sequence import *

setup_logger()
warnings.filterwarnings('ignore')


"""
CURRENT:
- MAML
    - K-Shot Generator 
        - www.tensorflow.org/tutorials/customization/autodiff?hl=ko
    - MAML Trainer
        - 배치 적용하기
        - 5 shots 학습 후 3 shots 학습으로 나누면 더 좋을까?
        - 실험 비교 
            - 이어지는 프로파일 vs 랜덤 프로파일
            - 프로파일 내 랜덤 셔플 vs 프로파일 내에서는 순차

TODO:
    - adaptation 
        - MAML, TAT, meta+metric 최신기법들 중 하나 선정해서 실행
    - robust
        - PGD (SS 논문 기반 - adversarial attack) 
        - Ordinal Loss (이것보다는 한 점 응시가 더 나을 듯)
        - eye rect augmentation (GG 논문 기반)
        - TAT 
    - 모델 아키텍쳐 변경
        - Eye 영역 크기 변경해보기
        - Residual Connection (WRN) - meta learning에 더 상승 효과가 좋음
        - final txmn 레이어 실험
    - 부스팅
        - 폰 / 태블릿 분리하여 해보기 (원본도 분리해서 했음)
        - 배치 사이즈 점진적 변경
        + tf.data 병렬
    - tensorboard 일정 주기로 기록하기
        - CAM 활성화 맵 시각화 하기
        - 급격한 loss 증가일 때의 샘플 (폰/태블릿 체크)
        - 초기 startup code 정리 (브라질 노트북)
    - 실험
        - 한 프로파일 별로 학습했을 때 stats 
        - Light LSTM

DATASET:
    - vc-one 데이터셋 적용
    - vc-one scene3 temporal 적용
"""



def train_by_basic(ctx):
    # Model
    sg_model = SGNet.create()
    
    # Data Generator
    train_seq = GazeCaptureNpzSequence.create(ctx, 'train')
    valid_seq = GazeCaptureNpzSequence.create(ctx, 'val')
    test_seq = GazeCaptureNpzSequence.create(ctx, 'test')

    # Trainer & Model
    trainer = Trainer.create(ctx, SGNet.create())
    trainer.setup_sequence(train_seq, valid_seq, test_seq)
    
    # Run Train
    trainer.summary()
    trainer.train()


def train_by_maml(ctx):
    # Model
    sg_model = SGNet.create()

    # Dataset for K-Shots
    train_seq = GazeCaptureKShotSequence.create(ctx, 'train', K=5)
    valid_seq = GazeCaptureKShotSequence.create(ctx, 'val',   K=5)
    test_seq  = GazeCaptureKShotSequence.create(ctx, 'test',  K=5)

    # Trainer
    trainer = MAMLTrainer.create(ctx, sg_model, SGNet)
    trainer.setup_sequence(train_seq, valid_seq, test_seq, to_tf_data=True)

    # Train
    trainer.summary()
    trainer.train(epochs=20)

    
    
"""
    MAIN
"""    
if __name__ == "__main__":

    config = {
        'seed' : 860515,
        'num_epochs' : 20,
        'num_workers' : 8,
        'batch_size' : 256,
        'base_lr': 0.0005,
        'l2_reg' : 1e-4,
        'resource_path': '../../../../data-archive/faze-resources/',
        'npz_root_path': '../../../../data-archive/faze-recode-profile-npz/',
    }

    ctx = Context.create(config)
    train_by_maml(ctx)
