import numpy as np

def cos_sim(x, y, eps=1e-08):
    norm_x = np.linalg.norm(x) + eps 
    norm_y = np.linalg.norm(y) + eps
    return np.dot(x, y) / (norm_x * norm_y)