import json
from stats.evaluation import *


class ProfileReport:
    
    def __init__(self, pid):
        self.pid = pid
        self.epoch_to_eval = {}
        self.num_eval_frames = -1
        
        
    """
        API
    """ 
    def add(self, eval_json):
        e = Evaluation.from_json(eval_json)
        self.epoch_to_eval[e.uid] = e
        
    def evaluation_of(self, epoch):    
        return self.epoch_to_eval[str(epoch)]
    
    def get_best_evaluation(self):
        last_key = list(self.epoch_to_eval.keys())[-1]
        return self.epoch_to_eval[last_key]
        
        
    """ 
        template methods
    """
    def is_empty(self):
        return False
        
        
    """
        Import & Export
    """
    @classmethod
    def parse_json(cls, json):
        if json == {}:
            return EmptyProfileReport()
        
        pid = list(json.keys())[0]
        evaluations = json[pid]
        
        r = cls(pid)
        for e in evaluations:
            r.add(e)
        
        return r
        
    @classmethod
    def load_from(cls, report_path):
        with open(report_path, 'r') as f:
            data = json.load(f)
        return cls.parse_json(data)
    
    
    """
        Property
    """
    @property
    def profile_id(self):
        return self.pid
        
        
    
"""
    Null Object Pattern
"""
class EmptyProfileReport(ProfileReport):
    
    def __init__(self):
        super().__init__(-1)
        
    def is_empty(self):
        return True
     
                        