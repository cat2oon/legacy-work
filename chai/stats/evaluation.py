import uuid
import numpy as np
import pandas ad pd

class Evaluation:
    
    def __init__(self, uid=None):
        self.set_uid(uid)
        self.fid_to_error = {}
        self.num_frames = 0
        self.total_mean = None
        self.total_std  = None
        self.total_min  = None
        self.total_max  = None
        self.percentile = []
        self.percentile_to_frame_ids = []
        
    def set_uid(self, uid):
        self.uid = str(uuid.uuid4()) if uid is None else uid
        
    
    """
        API
    """
    def calc_stats(self):
        num_frames = len(self.fid_to_error.keys())
        errors = np.array(self.fid_to_error.values())
        
        df = pd.DataFrame(errors)[0]
        total_mean = df.mean()
        total_std = df.std()
        total_median = df.median()
        total_min = df.min()
        total_max = df.max()
        percentile = np.array(df.quantile(np.linspace(.1, 1, 9, 0)))
        
        
    
    """
        Import & Export
    """
    @classmethod
    def from_error(cls, fid_to_error):
        e = cls()
        e.fid_to_error = fid_to_error
        return e
     
    @classmethod
    def from_json(cls, json):
        stats, items = json
        stats = eval(stats)
        epoch = list(stats.keys())[0]
        stats = stats[epoch]
        
        e = cls(epoch)
        e.num_frames = int(stats['count'])
        e.total_mean = stats['mean']
        e.total_std  = stats['std']
        e.total_min  = stats['min']
        e.total_max  = stats['max']
        e.percentile = [stats['{}%'.format(i*10)] for i in range(1,10)]
        e.percentile_to_frame_ids = [[id[0] for id in sorted(fids)] for fids in items.values()]
        
        return e
        