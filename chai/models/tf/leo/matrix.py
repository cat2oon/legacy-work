import sys
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import namedtuple as nt

class Matrix():
    Item = nt("ITEM", ['id', 'true', 'pred', 'err'])
    
    def __init__(self, ctx):
        self.ctx = ctx
        self.current_step = 0
        self.current_epoch = 0
        self.current_batch_idx = 0
        self.mean_dist = 0.0
        self.accum_mean_dist = []
        
        self.current_loss = None
        self.history = []
        self.test_history = []
        
    """
        Metrics
    """
    @classmethod
    def compute_errors(cls, report): 
        fid_to_errors = {}
        for i in range(len(report)):
            items = report[i]
            t = items['true']
            p = items['pred']
            fid = items['id']
            fid = Matrix.to_numpy(fid)
            t, p = Matrix.to_numpy(t), Matrix.to_numpy(p)
            errors = Matrix.euclidean_dist(t, p)
            for fi, err in zip(fid, errors):
                fid_to_errors[int(fi)] = err
        return dict(sorted(fid_to_errors.items())) 

    @classmethod
    def compute_metrics(cls, result, refine=False): 
        np_result, errors = [], []
        for i in range(len(result)):
            items = result[i]
            t, p = items['true'], items['pred']
            t, p = Matrix.to_numpy(t), Matrix.to_numpy(p)
            error = Matrix.euclidean_dist(t, p)
            errors += error
            if refine:
                for j in range(items['id'].shape[0]):
                    n_item = cls.Item(items['id'][j], t[j], p[j], error[j])
                    np_result.append(n_item)
        errors = Matrix.remove_nan_or_inf(errors)
        return errors, np_result

    @classmethod
    def euclidean_dist(cls, true, pred):
        old_settings = np.seterr(all='ignore')  
        np.seterr(all='ignore')
        diffs = true - pred
        dists = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)
        dists = list(np.around(dists, decimals=3))
        np.seterr(**old_settings)  
        return dists

    
    """
        Helper 
    """
    @classmethod
    def to_numpy(cls, x): 
        type_x = type(x)
        if type_x is np.ndarray:
            return x
        return x.numpy()
    
    @classmethod
    def remove_nan_or_inf(cls, arrs):
        return [v for v in arrs if not math.isnan(v) and not math.isinf(v)]
    

    """
        Visualize
    """
    @classmethod
    def statistics(cls, result, func=None):
        errors, res = Matrix.compute_metrics(result, refine=True)
        if func is not None:
            return func(res)
        return errors, pd.DataFrame(np.array(errors))
    
    def report(self):
        msg = "[{:03d} {:6d}] 현재: {:3f} / 누적: {:3f} | loss:{:2f} {:60s}".format(
            self.current_epoch, self.current_step, self.mean_dist, 
            self.get_accum_mean_dist(), self.current_loss, "|")
        print(msg, end='\r')
        sys.stdout.flush()       
    
    
    """
        API
    """
    def add(self, result, loss):
        self.current_loss = np.around(np.mean(loss), decimals=2)
        dists, _ = Matrix.compute_metrics(result)
        self.mean_dist = float(np.mean(dists))
        self.accum_mean_dist += dists
        self.current_step += 1
        
    def add_test(self, result):
        dists, _ = Matrix.compute_metrics(result)
        self.mean_dist = float(np.mean(dists))
        print('e[{:03d}] test:'.format(self.current_epoch), 
              np.around(np.mean(dists), decimals=4), "|")
        
        if self.current_epoch % 30 is not 0 or self.current_epoch is 0:
            return 
        
        pts = []
        for i in range(self.ctx.batch_size):
            preds = np.around(result[i]['pred'].numpy(), decimals=3)  
            trues = np.around(result[i]['true'].numpy(), decimals=3)
            pts += zip(preds, trues)
            
        for pred, true in pts:
        # for i in range(0, len(pts), 3):
            # print(pts[i], pts[i+1], pts[i+2])
            # print("P:", pred, "T:", true)
            print("D:", np.around(pred - true, decimals=3))
        

    def get_accum_mean_dist(self):
        md = np.around(np.mean(self.accum_mean_dist), decimals=3)
        md = float(md)
        return md
        

    """
        Step
    """
    def next_epoch(self, report_after=3, report=True):
        if report and self.current_epoch > report_after:
            pass
#             print("epochs[{}]: {} {:60s}".format(
#                 self.current_epoch, self.get_accum_mean_dist(), "|"))
        self.current_epoch += 1
        self.accum_mean_dist = []
        
    def next_batch(self, report=True):
        if report: 
            self.report()
        self.current_batch_idx += 1
 