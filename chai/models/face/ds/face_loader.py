from ds.morphable_model import *

class FaceDatasetLoader:
    
    def __init__(self, dataset_path):
        self.morphable_model = MorphableModel(dataset_path)
        
        
    def get_item(self, item_id):
        return self.morphable_model.process_item(item_id)
   
