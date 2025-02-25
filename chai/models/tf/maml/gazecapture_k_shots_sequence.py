from gazecapture_sequence import *



"""
    GazeCaptureNpzSequence for K-Few Shots
"""
class GazeCaptureKShotSequence(GazeCaptureNpzSequence):
    
    @classmethod
    def create(cls, ctx, data_tag, K):
        return GazeCaptureKShotSequence(data_tag,
                                        ctx.npz_root_path,
                                        ctx.resource_path,
                                        K,
                                        ctx.seed,
                                        exclude_profiles) 
    
    def __init__(self, 
                 data_tag, 
                 npz_root_path, 
                 resource_path, 
                 K=5, 
                 seed=1234,
                 exclude_profiles=None, 
                 custom_getter=None):
        super(GazeCaptureKShotSequence, self).__init__(data_tag, 
                                                       npz_root_path,
                                                       resource_path,
                                                       K,
                                                       seed,
                                                       exclude_profiles,
                                                       custom_getter)
        
    """ Override : Preapre Index """
    def get_index_iterator(self):
        for pid in self.index:
            item_ids = self.index[pid] # use choice not shuffle
            for iid in np.random.choice(item_ids, len(item_ids)):  
                yield pid, iid
        return None, None

    def get_batch_iterator(self, K):
        gen = self.get_index_iterator()
        while True:
            bag = {}
            for i in range(K):
                pid, iid = next(gen)
                if pid is None:
                    return None
                if pid not in bag:
                    bag[pid] = []
                bag[pid].append(iid)
            if len(bag) > 1:     # Not allow mixing profiles 
                continue
            yield bag

    def prepare_batch_index(self, batch_size):
        batch_iter = self.get_batch_iterator(batch_size)
        while True:
            try:
                batch = next(batch_iter) 
                self.batch_index.append(batch)
            except:
                break
                
        """ TODO: 매개변수 인자 주기 - 성능 저하 요소 체크 """
        np.random.shuffle(self.batch_index)
        
        logging.info("Complete build %s batch index", self.tag)            
            
