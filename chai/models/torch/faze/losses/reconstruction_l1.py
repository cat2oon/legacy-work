import torch.nn as nn


class ReconstructionL1Loss(object):

    def __init__(self, suffix='b'):
        self.suffix = suffix
        self.loss_fn = nn.L1Loss(reduction='mean')

    def __call__(self, input_dict, output_dict):
        x = input_dict['image_' + self.suffix].detach()
        x_hat = output_dict['image_' + self.suffix + '_hat']
        return self.loss_fn(x, x_hat)
