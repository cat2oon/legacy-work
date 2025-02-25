import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
 Tensor Ops
"""
def GlobalAveragePooling(x):
    return x.mean(-1).mean(-1)  

def swap_axis(x, axis_a, axis_b):
    return torch.transpose(x, axis_a, axis_b)

"""
 Random Seed
"""
def setup_seed(seed):
    """
    랜덤 시드를 세팅하더라도 worker가 0이 아닌 경우에(?) 다소 차이가 있다고 함
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # discuss.pytorch.org/t/random-seed-initialization/7854/16 suggest
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def worker_init_fn(worker_id):
    # Custom worker init to not repeat pairs
    np.random.seed(np.random.get_state()[1][0] + worker_id)