import logging
from bunch import Bunch


"""
    Trainer Context
"""
class Context():
    instance = None
    
    def __init__(self, config, **kwargs):
        self.check_config(config)
        self.__dict__.update(config)
        self.setup_default_configs()
    
    @classmethod
    def create(cls, obj):
        if type(obj) is Context:
            return obj
        ctx = Context(obj)
        cls.instance = ctx
        return ctx
    
    @classmethod
    def get(cls):
        return cls.instance
    
    def check_config(self, config):
        check = self.assert_contains
        check(config, 'num_epochs')
    
    def assert_contains(self, config, key):
        assert key in config, ('require [%s] in config' % key)
        
    def remove_attr(self, key):
        if hasattr(self, key):
            delattr(self, key)
    
    def set_if_not_exist(self, key, value):
        if not hasattr(self, key):
            setattr(self, key, value)
    
    def setup_default_configs(self):
        sety = self.set_if_not_exist
        sety('checkpoint_path', "weights.{epoch:03d}-{val_loss:.3f}.hdf5")
        sety('num_workers', 8)
        sety('prefetch_size', 32)