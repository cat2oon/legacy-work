from abc import *
from abc import ABCMeta

from ac.common.prints import console_print
from ai.nas.params.controller import ControllerParams


class Controller(ABC, metaclass=ABCMeta):
    def __init__(self, params: ControllerParams):
        console_print("Build controller {}".format(params.name))

        self.params = params
        self.name = params.name

        """ trainer outs """
        self.lr = None
        self.rl_loss = None
        self.optimizer = None
        self.reward = None
        self.train_op = None
        self.train_loss = None
        self.train_step = None
        self.baseline = None
        self.grad_norm = None

        """ macro only """
        self.skip_rate = None

        """ controller weights """
        self.w_lstm = []
        self.g_emb = None
        self.w_emb = None
        self.w_soft = None
        self.b_soft = None
        self.b_soft_no_learn = None
        self.w_attn_1 = None
        self.w_attn_2 = None
        self.v_attn = None

        """ sampler """
        self.sample_arc = None  # (conv, reduce)
        self.sample_entropy = None
        self.sample_log_prob = None
        self.build_arc_samplers()

    """ arc builder """

    @abstractmethod
    def build_arc_samplers(self):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def make_sampler(self, lstm_size, num_lstm_layers, prev_c=None, prev_h=None, use_bias=False):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def create_params(self, num_branches, lstm_size, num_lstm_layers):
        raise NotImplementedError("abstract method")

    def build_trainer(self, child_model):
        trainer_outs = self.create_trainer(child_model)

        self.lr = trainer_outs["lr"]
        self.rl_loss = trainer_outs["rl_loss"]
        self.optimizer = trainer_outs["optimizer"]
        self.reward = trainer_outs["reward"]
        self.train_op = trainer_outs["train_op"]
        self.train_loss = trainer_outs["train_loss"]
        self.train_step = trainer_outs["train_step"]
        self.baseline = trainer_outs["baseline"]
        self.grad_norm = trainer_outs["grad_norm"]
        self.skip_rate = trainer_outs["skip_rate"]

    @abstractmethod
    def create_trainer(self, child_model):
        raise NotImplementedError("abstract method")

    """ accessors """

    def get_sample_arc(self):
        return self.sample_arc

    def get_controller_ops(self):
        controller_ops = {
            "lr": self.lr,
            "loss": self.train_loss,
            "train_op": self.train_op,
            "train_step": self.train_step,
            "grad_norm": self.grad_norm,
            "optimizer": self.optimizer,
            "baseline": self.baseline,
            "entropy": self.sample_entropy,
            "sample_arc": self.sample_arc,
            "skip_rate": self.skip_rate,
            "sample_val_loss": self.rl_loss,
            "sample_reward": self.reward
        }
        return controller_ops
