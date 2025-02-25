from ac.common.utils import get_field_key_val_pairs
from ac.common.prints import console_print, pretty_print_pairs


class TrainOpParams:
    def __init__(self):
        """"""
        """ custom """
        self.purpose = "unknown"

        self.num_train_batches = 0          # for lr annealing
        self.lr_dec_every_batches = 100     # for lr decaying

        self.l2_reg = 1e-4
        self.clip_mode = "norm"  # { "global" "norm", None }
        self.grad_bound = None

        self.num_replicas = None
        self.sync_replicas = False
        self.num_aggregate = None

        self.lr_init = 1e-3
        self.lr_dec_start = 0
        self.lr_dec_rate = 0.9
        self.lr_max = None
        self.lr_min = None
        self.lr_T_0 = None
        self.lr_T_mul = None
        self.lr_cosine = False
        self.optimizer_name = "momentum"

        """ NO FLAGS """
        self.lr_dec_min = None       # NO FLAG
        self.lr_warmup_val = None    # NO FLAG
        self.lr_warmup_steps = 100   # NO FLAG
        self.get_grad_norms = False  # NO FLAG
        self.moving_average = None   # store the moving average of parameters NO FLAG

    def set_num_batches_and_decay(self, num_batches, decay_every_epochs):
        self.num_train_batches = num_batches
        self.lr_dec_every_batches = num_batches * decay_every_epochs

    @staticmethod
    def from_flags(flags, purpose="models"):
        p = TrainOpParams()

        if purpose == "models":
            p.for_model(flags)
        elif purpose == "controller":
            p.for_controller(flags)
        return p

    def for_model(self, flags):
        self.purpose = "models"
        self.l2_reg = flags.child_l2_reg
        self.clip_mode = flags.clip_mode
        self.grad_bound = flags.child_grad_bound

        self.lr_max = flags.child_lr_max
        self.lr_min = flags.child_lr_min
        self.lr_init = flags.child_lr
        self.lr_dec_start = flags.child_lr_dec_start
        self.lr_dec_rate = flags.child_lr_dec_rate
        self.lr_cosine = flags.child_lr_cosine
        self.lr_T_0 = flags.child_lr_T_0
        self.lr_T_mul = flags.child_lr_T_mul
        self.optimizer_name = flags.optimizer_name

        self.num_replicas = flags.child_num_replicas
        self.sync_replicas = flags.child_sync_replicas
        self.num_aggregate = flags.child_num_aggregate

    def for_controller(self, flags):
        self.purpose = "controller"
        self.l2_reg = flags.controller_l2_reg
        self.clip_mode = None
        self.grad_bound = None

        self.lr_dec_start = 0
        self.optimizer_name = "adam"
        self.lr_init = flags.controller_lr
        self.lr_dec_rate = flags.controller_lr_dec_rate

        self.sync_replicas = flags.controller_sync_replicas
        self.num_aggregate = flags.controller_num_aggregate
        self.num_replicas = flags.controller_num_replicas

    def pretty_print(self):
        console_print("Params for {}".format(self.purpose))
        pretty_print_pairs(get_field_key_val_pairs(self))
