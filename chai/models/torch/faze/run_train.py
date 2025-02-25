#!/usr/bin/env python
# coding: utf-8

from train import *
from generator.generator import *

import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO) 


def get_mock_config():
    return {
        'normalize_3d_codes' : True,     'normalize_3d_codes_axis': 1,
        'densenet_growthrate' : 32, 'z_dim_app' : 64, 'z_dim_gaze' : 2, 'z_dim_head' : 16, 'decoder_input_c' : 32,

        'triplet_loss_type' : 'angular',  'triplet_loss_margin' : 0.0, 'triplet_regularize_d_within' : True,   # or euclidean
        'embedding_consistency_loss_warmup_samples' : 1000000, 'all_equal_embeddings': True,
        'embedding_consistency_loss_type' : None, # angular, euclidean

        'backprop_gaze_to_encoder' : True,
        'coeff_l1_recon_loss' : 1.0, 'coeff_gaze_loss' : 0.1, 'coeff_embedding_consistency_loss' : 2.0, 

        'pin_memory': False,
        # 'num_data_loaders' : 0,     
        'num_data_loaders' : 16,     
        # 'batch_size' : 1,
        'batch_size' : 32,
        # 'batch_size' : 64,
        # 'batch_size' : 256,
        'use_apex' : False,
        'base_lr': 0.0005,
        'warmup_period_for_lr' : 1000000, 
        
        'decay_interval' : 0, 'decay' : 0.8, 'num_training_epochs' : 20, 'l2_reg' : 1e-4, 'print_freq_train' : 20, 'print_freq_test' : 5000,
        'resource_path': '../../../../../data-archive/faze-resources/',
        'npz_root_path': '../../../../../data-archive/faze-recode-profile-npz/',
    }


if __name__ == "__main__":


    """
        Configuration
    """
    config = get_mock_config()
    Context.build(config)
    ctx = Context.get()


    """
        TED
    """
    network = TED(
        growth_rate=ctx.densenet_growthrate,
        z_dim_app=ctx.z_dim_app,
        z_dim_gaze=ctx.z_dim_gaze,
        z_dim_head=ctx.z_dim_head,
        decoder_input_c=ctx.decoder_input_c,
        normalize_3d_codes=ctx.normalize_3d_codes,
        normalize_3d_codes_axis=ctx.normalize_3d_codes_axis,
        use_triplet=ctx.triplet_loss_type,
        backprop_gaze_to_encoder=ctx.backprop_gaze_to_encoder,
    )


    """
        Set Network 
    """
    ctx.set_network('ted', network)
    ctx.load_network_to_device()
    ctx.setup_base_loss_policy(verbose=True)


    """
        Optimizer
    """
    optimizer, gaze_optimizer = ctx.build_optimizer('ted')
    ctx.set_main_optimizer(optimizer)
    ctx.set_gaze_optimizer(gaze_optimizer)

    """
        Loss
    """
    loss_functions = ctx.make_loss_functions()
    ctx.set_loss_functions(loss_functions)


    """
        Data-Generator
    """
    gen = NPZDatasetGenerator(ctx, shuffle_train=False)
    databag = gen.generate(verbose=True)
    ctx.set_databag(databag)

    """
        Check DataLoader
        'pid', 'key_idx', 'image_a', 'gaze_a', 'head_a', 'item_idx_a', 'rot_gaze_a', 'rot_head_a', 'target_x_a', 'target_y_a', 
        'image_b', 'gaze_b', 'head_b', 'item_idx_b', 'rot_gaze_b', 'rot_head_b', 'target_x_b', 'target_y_b'
    """
    # train_ds, loader = ctx.get_dataset_and_loader(Context.TAG_TRAIN)
    # for input_dict in iter(loader):
        # print("ProfileId {}, \nItem_idx {}\n\n".format(input_dict['pid'], input_dict['item_idx_b']))
        # print("PID: {}, IDX_B {} \t type TX: {}".format(input_dict['pid'], input_dict['item_idx_b'], type(input_dict['target_x_a'])))
        # print("{}".format(input_dict['target_x_a']))


    """
        Run Train
    """
    ctx.use_multiple_gpu_if_available()
    ctx.run_train('ted')






