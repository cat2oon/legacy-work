class NASParams:
    def __init__(self):
        self.num_epochs = None
        self.model_log_every = 50           # n steps
        self.ctrl_log_every = 50            # n steps
        self.eval_every_epochs = 1
        self.exp_name = "exp_"

        self.output_dir = "./out"
        self.reset_output_dir = True

    @staticmethod
    def from_flags(flags):
        p = NASParams()

        p.exp_name = flags.exp_name
        p.num_epochs = flags.num_epochs
        p.model_log_every = flags.log_every
        p.ctrl_log_every = flags.controller_log_every
        p.eval_every_epochs = flags.eval_every_epochs

        return p
