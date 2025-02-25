from ai.nas.params.train_ops import TrainOpParams


class ControllerParams:
    def __init__(self, controller_name="no-name"):
        self.name = controller_name

        """ for train ops """
        self.for_train_op = None
        self.lr_dec_every_epochs = 2

        """ macro only """
        self.skip_target = 0.8
        self.skip_weight = 0.5

        """ controller's but NAS use """
        self.train_steps = 50
        self.train_every = 2
        self.forwards_limit = 2  # 쓰이지 않음
        self.training = True

        """ default value in child """
        self.search_for = "both"  # share       # micro x
        self.search_whole_channels = False      # micro x

        self.num_cells = 6
        self.num_layers = 4
        self.num_branches = 6
        self.out_filters = 48

        self.lstm_size = 32
        self.lstm_num_layers = 2
        self.lstm_keep_prob = 1.0               # micro 사용 안함

        self.tanh_constant = None
        self.op_tanh_reduce = 1.0
        self.temperature = None

        self.entropy_weight = None
        self.use_critic = False     # micro x
        self.baseline_decay = 0.999  # same above

    @staticmethod
    def from_flags(flags, name="exp1"):
        p = ControllerParams()

        """ train ops """
        p.for_train_op = TrainOpParams.from_flags(flags, "controller")
        p.lr_dec_every_epochs = flags.controller_lr_dec_every_epochs

        """ macro only """
        p.skip_target = flags.controller_skip_target
        p.skip_weight = flags.controller_skip_weight

        """ controller's but NAS use """
        p.train_steps = flags.controller_train_steps
        p.train_every = flags.controller_train_every
        p.forwards_limit = flags.controller_forwards_limit
        p.training = flags.controller_training

        """ general """
        p.name = name
        p.search_for = flags.search_for
        p.search_whole_channels = flags.controller_search_whole_channels

        p.num_cells = flags.child_num_cells  # 사용함
        p.num_layers = flags.child_num_layers  # macro sample 할 때 사용
        p.num_branches = flags.child_num_branches  # 사용함
        p.out_filters = flags.child_out_filters  # micro 사용 안함
        p.lstm_keep_prob = flags.controller_keep_prob       # micro 사용 안함

        p.tanh_constant = flags.controller_tanh_constant
        p.op_tanh_reduce = flags.controller_op_tanh_reduce
        p.temperature = flags.controller_temperature

        p.entropy_weight = flags.controller_entropy_weight
        p.baseline_decay = flags.controller_bl_dec
        p.use_critic = flags.controller_use_critic      # micro 사용안함

        return p

    def get_for_train_op(self) -> TrainOpParams:
        train_op_params = self.for_train_op
        train_op_params.set_num_batches_and_decay(self.train_steps, self.lr_dec_every_epochs)
        return train_op_params
