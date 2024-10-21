import numpy as np

class DataFrame:
    def __init__(self):
        pass
    
def block_diag(a: np.ndarray, b: np.ndarray)->np.ndarray:
    y1,x1 = a.shape
    y2,x2 = b.shape
    res = np.zeros((y1+y2,x1+x2),dtype=a.dtype)
    res[:y1,:x1] = a
    res[y1:y1+y2,x1:x1+x2] = b
    return res