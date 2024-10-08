from scipy.signal import convolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_avg(x: np.ndarray, N: int=4, mode='same') -> np.ndarray:
    filter = np.ones(N)/N
    y = convolve(x,filter,mode=mode)
    return y

def normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype('float')
    x -= x.min()
    x /= x.max()
    return x
