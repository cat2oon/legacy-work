#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import time
import json
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict

from densenet.ted import *
from common.torch_common import *
from losses import (ReconstructionL1Loss, GazeAngularLoss, BatchHardTripletLoss,
                    AllFrontalsEqualLoss, EmbeddingConsistencyLoss)


"""
    Torch Context
"""
class Context():
    instance = None
    
    TAG_VAL = 'gc/val'
    TAG_TEST = 'gc/test'
    TAG_TRAIN = 'gc/train'
    SYM_DATASET = 'dataset'
    SYM_DATALOADER = 'dataloader'
    
    def __init__(self, config):
        self.config = config
        self.device = None
        self.databag = {}
        self.networks = {}
        self.optimizers = {}
        self.input_dict = None
        self.loss_functions = OrderedDict()
        self.initial_step = 0
        self.time_epoch_start = None
        self.num_elapsed_epochs = 0
        self.tensorboard = None
        
        self.prepare(config)
        
    @classmethod
    def build(cls, config):
        cls.instance = Context(config)
        
    @classmethod
    def get(cls):
        return cls.instance
    
    def prepare(self, config):
        self.make_config_prop()
        self.check_config_sanity(config)
        self.device = self.make_torch_device()
            
    
    """
        Epoch Handler
    """
    def start_epoch(self):
        if self.time_epoch_start is None:
            self.time_epoch_start = time.time()
        self.time_batch_fetch_start = time.time()

    def stop_epoch(self): 
        num_elapsed_epochs += 1
        time_epoch_end = time.time()
        time_epoch_diff = time_epoch_end - time_epoch_start
        if self.use_tensorboard:
            self.tensorboard.add_scalar('timing/epoch', time_epoch_diff, num_elapsed_epochs)
        # Done with an epoch now...!
        if num_elapsed_epochs % 5 == 0:
            self.saver.save_checkpoint(self.get_current_step())

    def reset_epoch(self):
        np.random.seed()  # Ensure randomness

        # Some cleanup
        train_data_iterator = None
        torch.cuda.empty_cache()
        gc.collect()

        # Restart!
        time_epoch_start = time.time()
        global train_dataloader
        train_data_iterator = iter(train_dataloader)
        time_batch_fetch_start = time.time()
        input_dict = next(train_data_iterator)
        
   
    """
        TVT Process
    """
    def load_input_to_device(self, input, device):
        for k, v in input.items():
            if isinstance(v, torch.Tensor):
                input[k] = v.detach().to(device, non_blocking=True)
        return input
        
    def setup_training_steps(self):
        if hasattr(self, 'skip_training') and self.skip_training:
            self.set_training_steps(0)
            return
        ds, _ = self.get_dataset_and_loader(Context.TAG_TRAIN)
        num_steps = int(self.num_training_epochs * len(ds) / self.batch_size)
        self.set_training_steps(num_steps)
  
    def run_train(self, network_key, verbose=True):
        self.setup_training_steps()
        self.setup_tensorboard()
        self.setup_running_statistics()
        self.setup_last_checkpoint(network_key)
        self.prepare_network_input(Context.TAG_TRAIN)
        
        logging.info('*** Run Training ***  Steps: [%d]' % self.num_training_steps) 
        for step in range(self.initial_step, self.num_training_steps):
            if verbose and step % 10 is 0:
                logging.info('>>> current step: %d' % step)
            self.set_current_step(step)
            self.execute_training_step(step, network_key)
        logging.info('>>> Complete training') 
       
    def execute_training_step(self, step, network_key, verbose=False):
        time_iteration_start = time.time()

        try:
            self.start_epoch()
            input = self.get_next_input_dict()
        except StopIteration:
            self.stop_epoch()
            self.reset_epoch()

        input_tensor = self.load_input_to_device(input, self.device)
        self.running_timings.add('batch_fetch', time.time() - self.time_batch_fetch_start)
        
        # Zero gradient
        tag = Context.TAG_TRAIN
        self.zero_optimizer_gradient(tag)
        time_forward_start = time.time()
        network = self.set_network_mode(network_key, tag)
        
        # Feed Foward 
        output_dict, loss_dict = network(input_tensor, self.loss_functions)

        # Compute Loss
        loss_dict = self.compute_multi_gpu_loss(loss_dict)
        loss_to_optimize = self.compute_loss_to_optimize(loss_dict)
        self.update_learning_rate(self.get_current_step())

        # Optimize Loss
        self.optimize_loss(loss_to_optimize)

        time_backward_end = time.time()
        self.running_timings.add('forward_and_backward', time_backward_end - time_forward_start)

        # Store values for logging later
        for key, value in loss_dict.items():
            loss_dict[key] = value.detach().cpu()
        for key, value in loss_dict.items():
            self.running_losses.add(key, value.numpy())

        self.running_timings.add('iteration', time.time() - time_iteration_start)            
        
        
        
    """
        Learning Rate Policy
    """
    def update_learning_rate(self, current_step):
        lr = self.learning_rate_at_step(current_step)
        all_param_groups = self.optimizer.param_groups
        for i, param_group in enumerate(all_param_groups):
            if i == 0:  # Don't do it for the gaze-related weights
                param_group['lr'] = lr
                
    def learning_rate_at_step(self, current_step):
        if current_step <= self.ramp_up_until_step:
            return self.ramp_up_a * current_step + self.ramp_up_b
        if self.decay_interval != 0:
            step_diff = current_step - self.ramp_up_until_step
            return np.power(self.decay, int(step_diff / self.decay_interval))
        return max_lr


    
    """
        Network, Data-Generator
    """    
    def set_network_mode(self, network_key, tag):
        net = self.get_network(network_key)
        if tag is Context.TAG_TRAIN:
            net.train()
        elif tag is Context.TAG_TEST:    
            net.eval()  
        return net
            
    def zero_optimizer_gradient(self, tag):
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()
            
    def prepare_network_input(self, data_tag):
        dataset, loader = self.get_dataset_and_loader(data_tag)
        """ TODO: T, V ,T """
        self.train_data_iterator = iter(loader)
    
    def get_next_input_dict(self):
        """ TODO: 데이터 문제 발생 시 처리"""
        input_dict = next(self.train_data_iterator)    
        return input_dict
        
    def set_network(self, key, network):
        assert network is not None
        self.networks[key] = network
        
    def get_network(self, key):
        return self.networks[key]
    
    def set_databag(self, databag):
        self.databag = databag
        logging.info('>>> data-bag is setup') 
        logging.info('=> { %s } ' % ', '.join(databag.keys()))
        
    def get_dataset_and_loader(self, data_tag):
        assert data_tag in self.databag
        dataset = self.databag[data_tag][Context.SYM_DATASET]
        dataloader = self.databag[data_tag][Context.SYM_DATALOADER]
        return dataset, dataloader
       
        
    
    """
        Optimizer
    """
    def optimize_loss(self, loss_to_optimize):
        if self.use_apex:
            with amp.scale_loss(loss_to_optimize, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_to_optimize.backward()
        self.optimizer.step()

        # optimize small gaze part too, separately (if required)
        if not self.backprop_gaze_to_encoder:
            if self.use_apex:
                with amp.scale_loss(loss_dict['gaze'], gaze_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_dict['gaze'].backward()
            self.gaze_optimizer.step()

    def build_optimizer(self, network_key):
        base_lr = self.base_lr
        gaze_lr = 1.0 * base_lr
        l2_w_decay = self.l2_reg

        net = self.get_network(network_key) 
        gaze_params = [p for n, p in net.named_parameters() if n.startswith('gaze')]
        non_gaze_params = [p for n, p in net.named_parameters() if not n.startswith('gaze')]

        gaze_optimizer = None
        if self.require_separate_gaze_loss():
            params = [{ 'params': non_gaze_params }, { 'params': gaze_params, 'lr': gaze_lr }]
        else:
            params = non_gaze_params
            gaze_optimizer = optim.SGD(gaze_params, lr=gaze_lr, momentum=0.9, 
                                       nesterov=True, weight_decay=l2_w_decay)    
            
        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, 
                              nesterov=True, weight_decay=l2_w_decay)
            
        return optimizer, gaze_optimizer
    
    def set_main_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.set_optimizer("optimizer", optimizer)
        
    def set_gaze_optimizer(self, optimizer):
        self.gaze_optimizer = optimizer
        self.set_optimizer("gaze_optimizer", optimizer)
    
    def set_optimizer(self, key, optimizer):
        if optimizer is not None:
            self.optimizers[key] = optimizer
            logging.info('Set optimizer as key: %s' % key)
        
    def get_optimizer(self, optimizer_key):
        return self.optimizers[optimizer_key]
        
        
        
    """
        Losses
    """
    def compute_multi_gpu_loss(self, loss_dict):
        # If doing multi-GPU training, just take an average
        for key, value in loss_dict.items():
            if value.dim() > 0:
                value = torch.mean(value)
                loss_dict[key] = value
        return loss_dict
    
    def compute_loss_to_optimize(self, loss_dict):
        loss_to_optimize = self.coeff_l1_recon_loss * loss_dict['recon_l1']

        if self.triplet_loss_type is not None:
            triplet_losses = []
            triplet_losses = [
                loss_dict['triplet_gaze_' + self.triplet_loss_type],
                loss_dict['triplet_head_' + self.triplet_loss_type],
            ]
            if self.triplet_regularize_d_within:
                triplet_losses += [
                    loss_dict['triplet_gaze_%s_d_within' % self.triplet_loss_type],
                    loss_dict['triplet_head_%s_d_within' % self.triplet_loss_type],
                ]
            loss_to_optimize += 1.0 * sum(triplet_losses)

        if self.embedding_consistency_loss_type is not None:
            gaze_loss_prefix = 'embedding_consistency_gaze_'   
            head_loss_prefix = 'embedding_consistency_head_'
            
            embedding_consistency_losses = [
                loss_dict[gaze_loss_prefix + self.embedding_consistency_loss_type],
              # loss_dict[head_loss_prefix + self.embedding_consistency_loss_type],
            ]

            coeff_ec_loss = self.ec_loss_weight_at_step(self.get_current_step())
            loss_to_optimize += coeff_ec_loss * sum(embedding_consistency_losses)

        if self.all_equal_embeddings:
            loss_to_optimize += sum([loss_dict['all_equal_gaze'], loss_dict['all_equal_head']])

        if self.backprop_gaze_to_encoder:
            loss_to_optimize += self.coeff_gaze_loss * loss_dict['gaze']

        return loss_to_optimize
    
    def ec_loss_weight_at_step(self, current_step):
        coef_ec_loss = cfg['coeff_embedding_consistency_loss']
        ec_loss_warmup_samples = cfg['embedding_consistency_loss_warmup_samples']

        final_value = coef_ec_loss
        if ec_loss_warmup_samples is None:
            return final_value

        warmup_steps = int(ec_loss_warmup_samples / cfg['batch_size'])
        if current_step <= warmup_steps:
            return (final_value / warmup_steps) * current_step
        return final_value
 
    def setup_base_loss_policy(self, verbose=False):
        self.max_lr = self.base_lr * self.batch_size
        self.ramp_up_until_step = int(self.warmup_period_for_lr / self.batch_size)
        self.ramp_up_a = (self.max_lr - self.base_lr) / self.ramp_up_until_step
        self.ramp_up_b = self.base_lr
        
        if verbose:
            logging.info('>>> base lose policy <<<')
            logging.info('max learning rate: %f' % self.max_lr)
            logging.info('ramp up a: %f, b: %f' % (self.ramp_up_a, self.ramp_up_b))
    
    def make_loss_functions(self):
        loss_functions = OrderedDict()
        loss_functions['gaze'] = GazeAngularLoss()
        loss_functions['recon_l1'] = ReconstructionL1Loss(suffix='b')

        if self.triplet_loss_type is not None:
            loss_functions['triplet'] = BatchHardTripletLoss(
                distance_type=self.triplet_loss_type, margin=self.triplet_loss_margin)

        if self.all_equal_embeddings:
            loss_functions['all_equal'] = AllFrontalsEqualLoss()

        if self.embedding_consistency_loss_type is not None:
            loss_functions['embedding_consistency'] = EmbeddingConsistencyLoss(
                distance_type=self.embedding_consistency_loss_type)

        return loss_functions
    
    def set_loss_functions(self, loss_functions):
        self.loss_functions = loss_functions
        logging.info('>>> Setup loss functions')
        logging.info('list: { %s } ' % ', '.join(loss_functions.keys()))
    
    
        
    """
        Config 
    """
    def check_config_sanity(self, config):
        assert config is not None
        
        use_norm_3d_codes = config['normalize_3d_codes']
        triplet_loss_type = config['triplet_loss_type']
        ec_loss_type = config['embedding_consistency_loss_type']

        if ec_loss_type is not None:
            assert triplet_loss_type is None
        if triplet_loss_type is not None:
            assert ec_loss_type is None

        if triplet_loss_type == 'angular' or ec_loss_type == 'angular':
            assert use_norm_3d_codes is True
        elif triplet_loss_type == 'euclidean' or ec_loss_type == 'euclidean':
            assert use_norm_3d_codes is False
            
        if 'use_tensorboard' in config and config['use_tensorboard']:
            assert 'save_path' in config 
            
        if 'use_checkpoint' in config and config['use_checkpoint']:
            assert 'save_path' in config 
        
    def make_config_prop(self):
        self.__dict__.update(self.config)
    
    
    
    """
        Device
    """
    def make_torch_device(self):
        machine = "cuda:0" if torch.cuda.is_available() else "cpu"
        return torch.device(machine)
    
    def load_network_to_device(self):
        for key, net in self.networks.items():
            net.to(self.device)
            
    def is_parallel_gpu_support(self):
        return torch.cuda.device_count() > 1
            
    def use_multiple_gpu_if_available(self):
        if self.is_parallel_gpu_support():
            logging.info('Using %d GPUs!' % torch.cuda.device_count())
            for key, net in self.networks.items():
                self.set_network(key, nn.DataParallel(net))
        else:
            logging.info('No multiple GPUs')
        
        
    
    """
        Accessors
    """
    def set_current_step(self, step):
        self.current_step = step
        
    def get_current_step(self):
        return self.current_step
        
    def use_apex(self):
        return False
        
    def use_parallel_device(self):
        return False
    
    def require_separate_gaze_loss(self):
        return not self.backprop_gaze_to_encoder 
    
    def set_initial_step(self, num_init_step):
        self.initial_step = num_init_step
        logging.info('>>> Set initial step %d' % num_init_step)
    
    def set_training_steps(self, num_steps):
        self.num_training_steps = num_steps
        self.last_training_step = num_steps - 1 
        
    def setup_tensorboard(self):
        if hasattr(self, 'use_tensorboard') and self.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.tensorboard = SummaryWriter(log_dir=self.save_path)
            logging.info('>>> use tensorboard. save_path: %s' % self.save_path)
   
    def setup_running_statistics(self):
        self.running_losses = RunningStatistics()
        self.running_timings = RunningStatistics()
    
    def setup_last_checkpoint(self, network_key):
        if hasattr(self, 'use_checkpoint') and self.use_checkpoint:
            net = self.get_network(network_key)
            saver, initial_step = self.load_checkpoint(net, self.save_path)
            self.saver = saver
            self.set_initial_step(initial_step)
        
    def load_checkpoint(self, network, save_path):
        from checkpoints_manager import CheckpointsManager
        saver = CheckpointsManager(network, save_path)
        initial_step = saver.load_last_checkpoint()
        return saver, initial_step
    
    def summary_network(self, network_key, input_dict):
        # summary(network, train_input_dict)
        pass
 

"""
   TEST 
"""
def execute_test(tag, data_dict):
    test_losses = RunningStatistics()
    with torch.no_grad():
        for input_dict in data_dict['dataloader']:
            network.eval()
            input_dict = send_data_dict_to_gpu(input_dict)
            output_dict, loss_dict = network(input_dict, loss_functions=loss_functions)
            for key, value in loss_dict.items():
                test_losses.add(key, value.detach().cpu().numpy())
    test_loss_means = test_losses.means()
    logging.info('Test Losses at [%7d] for %10s: %s' %
                 (current_step + 1, '[' + tag + ']',
                  ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))
    if self.use_tensorboard:
        for k, v in test_loss_means.items():
            tensorboard.add_scalar('test/%s/%s' % (tag, k), v, current_step + 1)

            
"""
    Context Utility
"""
class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([ (k, np.mean(v)) for k, v in self.losses.items() if len(v) > 0 ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []

