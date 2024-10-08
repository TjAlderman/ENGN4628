from scipy.signal import convolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_avg(x: np.ndarray, N: int=4, mode='same') -> np.ndarray:
    filter = np.ones(N)/N
    y = convolve(x,filter,mode=mode)
    return y
