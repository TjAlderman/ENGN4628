from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_avg(x: np.ndarray, N: int = 4, mode="same") -> np.ndarray:
    filter = np.ones(N) / N
    y = convolve(x, filter, mode=mode)
    return y


def gaussian(
    x: np.ndarray, N: int = 7, sigma: int = 1, mode: str = "constant"
) -> np.ndarray:
    return gaussian_filter1d(x, sigma=sigma, radius=N // 2, mode=mode)


def normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype("float")
    x -= x.min()
    x /= x.max()
    return x
