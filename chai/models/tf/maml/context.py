from context import * 


"""
    MAML Trainer Context
"""
class MAMLContext(Context):
    instance = None

    def __init__(self, config):
        super().__init__(config)
        self.remove_unused()
    
    @classmethod
    def create(cls, obj):
        if type(obj) is MAMLContext:
            return obj
        ctx = MAMLContext(obj)
        cls.instance = ctx
        return ctx
    
    @classmethod
    def get(cls):
        return cls.instance
    
    def remove_unused(self):
        self.remove_attr('base_lr')
        self.remove_attr('batch_size')
    
    def check_config(self, config):
        check = self.assert_contains
        check(config, 'num_epochs')
        check(config, 'num_grad_steps')
    
    def setup_default_configs(self):
        sety = self.set_if_not_exist
        sety('task_lr', 0.001)
        sety('meta_lr', 0.001)
        sety('checkpoint_path', "weights.{epoch:03d}-{val_loss:.3f}.hdf5")