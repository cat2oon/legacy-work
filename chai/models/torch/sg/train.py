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

from tqdm import tqdm
from collections import OrderedDict

from losses import (ReconstructionL1Loss, GazeAngularLoss, BatchHardTripletLoss,
                    AllFrontalsEqualLoss, EmbeddingConsistencyLoss)


"""
    Torch Trainer
"""
class Context():
    instance = None
    
    # TODO: ENUM
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
   
    def make_config_prop(self):
        self.__dict__.update(self.config)
         
    def check_config_sanity(self, config):
        assert config is not None
        if 'use_tensorboard' in config and config['use_tensorboard']:
            assert 'save_path' in config 
        if 'use_checkpoint' in config and config['use_checkpoint']:
            assert 'save_path' in config 
        return
    

    """
        Train Process
    """
    def load_input_to_device(self, input, device):
        for k, v in input.items():
            if isinstance(v, torch.Tensor):
                input[k] = v.detach().to(device, non_blocking=True)
                input[k] = input[k].double()
        return input
  
    def run_train(self, network_key, verbose=True):
        # self.setup_tensorboard()
        # self.setup_last_checkpoint(network_key)
        self.prepare_network_input(Context.TAG_TRAIN)
        
        logging.info('') 
        logging.info('*** Training ***') 
        
        for this_epoch in range(self.num_training_epochs):
            logging.info('>>> Epoch {}/{}'.format(this_epoch, self.num_training_epochs))
            
            tk0 = tqdm(self.train_data_loader, total=int(len(self.train_data_loader)))
            for batch_idx, inputs in enumerate(tk0):
                # Zero gradient
                tag = Context.TAG_TRAIN
                self.zero_optimizer_gradient(tag)
                network = self.set_network_mode(network_key, tag)

                # Feed Foward 
                input_tensor = self.load_input_to_device(inputs, self.device)
                output_dict, loss_dict = network(input_tensor, self.loss_functions)

                # Compute Loss
                loss_dict = self.compute_multi_gpu_loss(loss_dict)
                loss_to_optimize = self.compute_loss_to_optimize(loss_dict)
                # self.update_learning_rate(self.get_current_step())

                # Optimize Loss
                self.optimize_loss(loss_to_optimize)
                tk0.set_postfix(loss=loss_to_optimize.item()) 
            
            # end epoch 
            np.random.seed()  # Ensure randomness
            torch.cuda.empty_cache()
            gc.collect()
        
        logging.info('') 
        logging.info('') 
        logging.info('>>> Complete training <<<') 
        
        
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
        elif tag is Context.TAG_TEST or tag is Context.TAG_VAL:
            net.eval()  
        return net
            
    def zero_optimizer_gradient(self, tag):
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()
            
    def prepare_network_input(self, data_tag):
        dataset, loader = self.get_dataset_and_loader(data_tag)
        self.train_data_loader = loader
        self.train_data_iterator = iter(loader)

    def set_databag(self, databag):
        self.databag = databag
        logging.info('') 
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

    def build_optimizer(self, network_key):
        net = self.get_network(network_key) 
        optimizer = optim.SGD(net.parameters(), lr=self.base_lr, momentum=0.9, 
                              nesterov=True, weight_decay=self.l2_reg)
        return optimizer
       
        
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
        loss_to_optimize = loss_dict['target_loss']
        return loss_to_optimize

    def setup_base_loss_policy(self, verbose=False):
        self.max_lr = self.base_lr * self.batch_size
        self.ramp_up_until_step = int(self.warmup_period_for_lr / self.batch_size)
        self.ramp_up_a = (self.max_lr - self.base_lr) / self.ramp_up_until_step
        self.ramp_up_b = self.base_lr
        
        if verbose:
            logging.info('')
            logging.info('>>> base lose policy <<<')
            logging.info('max learning rate: %f' % self.max_lr)
            logging.info('ramp up a: %f, b: %f' % (self.ramp_up_a, self.ramp_up_b))
    
    def make_loss_functions(self):
        loss_functions = OrderedDict()
        loss_functions['gaze'] = GazeAngularLoss()
        return loss_functions
    
    def set_loss_functions(self, loss_functions):
        self.loss_functions = loss_functions
        logging.info('')
        logging.info('>>> Setup loss functions <<<')
        logging.info('loss list: { %s } ' % ', '.join(loss_functions.keys()))
   
    
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
        for key, net in self.networks.items():
            self.set_network(key, nn.DataParallel(net))
#         if self.is_parallel_gpu_support():
#             logging.info('Using %d GPUs!' % torch.cuda.device_count())
#             for key, net in self.networks.items():
#                 self.set_network(key, nn.DataParallel(net))
#         else:
#             logging.info('>>> No multiple GPUs <<<')
        
        
    
    """
        Accessors
    """
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
        
    def set_network(self, key, network):
        assert network is not None
        network = network.double()
        self.networks[key] = network
        
    def get_network(self, key):
        return self.networks[key]
     
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

      