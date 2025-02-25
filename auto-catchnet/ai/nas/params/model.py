from ai.nas.params.train_ops import TrainOpParams


class ModelParams:
    def __init__(self, model_name="no-name"):
        """"""

        """ custom defaults """
        self.seed = "860515"
        self.data_path = None
        self.monitor_thresholds = 100.0
        self.dataset_mode = None
        self.pool_layers = None

        """ """
        self.name = model_name
        self.use_aux_heads = False

        self.fixed_arc = None
        self.num_epochs = None
        self.cutout_size = None
        self.batch_size = 32
        self.eval_batch_size = 100
        self.lr_dec_every_epochs = 1

        self.num_layers = 2
        self.num_cells = 5
        self.out_filters = 24
        self.keep_prob = 1.0
        self.drop_path_keep_prob = None

        """ train ops """
        self.for_train_op = None

        """ only for macro search """
        self.num_branches = 6
        self.out_filters_scale = 1
        self.whole_channels = False

    @staticmethod
    def from_flags(flags, name='v'):
        p = ModelParams()

        """ train ops """
        p.for_train_op = TrainOpParams.from_flags(flags)

        """ custom defaults """
        p.data_path = flags.data_path
        p.dataset_mode = flags.dataset_mode
        p.monitor_thresholds = flags.monitor_thresholds
        p.pool_layers = flags.pool_layers

        """ base models defaults """
        p.name = name
        p.num_epochs = flags.num_epochs
        p.fixed_arc = flags.child_fixed_arc
        p.cutout_size = flags.child_cutout_size
        p.use_aux_heads = flags.child_use_aux_heads

        p.num_layers = flags.child_num_layers
        p.num_cells = flags.child_num_cells
        p.out_filters = flags.child_out_filters
        p.keep_prob = flags.child_keep_prob
        p.drop_path_keep_prob = flags.child_drop_path_keep_prob

        p.batch_size = flags.batch_size
        p.eval_batch_size = flags.eval_batch_size
        p.grad_bound = flags.child_grad_bound
        p.lr_dec_every_epochs = flags.child_lr_dec_every

        """ only for macro search """
        p.num_branches = flags.child_num_branches
        p.whole_channels = flags.controller_search_whole_channels
        p.out_filters_scale = flags.child_out_filters_scale

        return p

    def get_for_train_op(self, num_batches) -> TrainOpParams:
        train_op_params = self.for_train_op
        train_op_params.set_num_batches_and_decay(num_batches, self.lr_dec_every_epochs)
        return train_op_params
