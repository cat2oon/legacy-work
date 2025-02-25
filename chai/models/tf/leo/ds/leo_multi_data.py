from ds.leo_data import *
# from ds.data_utils import *


"""
    Multi-DataProvider
"""
class MultiDataProvider():
    
    @timeit
    def __init__(self, ctx, pids, shuffle=True):
        self.ctx = ctx
        self.pids = pids
        self.build_index(shuffle)
        
    def build_index(self, shuffle):
        self.providers = [DataProvider(self.ctx, pid) for pid in self.pids]
        self.provider_lens = [len(p) for p in self.providers]
        
        access_index = [] 
        for provider_idx, length in enumerate(self.provider_lens):
            for batch_idx in range(length):
                access_index.append((provider_idx, batch_idx))
        self.access_index = access_index
        
        if shuffle:
            np.random.shuffle(access_index)
        
    def __len__(self):
        return len(self.access_index)

    def __getitem__(self, batch_idx):
        p_idx, p_batch_idx = self.access_index[batch_idx]
        provider = self.providers[p_idx]
        return provider[p_batch_idx]
