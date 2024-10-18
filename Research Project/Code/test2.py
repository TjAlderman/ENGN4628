import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from src.state import HEV

hybrid = HEV()

w = np.linspace(0,60,600)
power_regen = hybrid.power_per_torque(w,"Regen")
power_ev = hybrid.power_per_torque(w,"EV")
plt.plot(w,power_regen)
plt.plot(w,power_ev)
plt.show()


data_name = 'short_5min'  # Consider making this a command-line argument
fake = True

try:
    if fake:
        alpha = np.load(f'data/fake-slope_{data_name}.npy')
        v = np.load(f'data/fake-velocity_{data_name}.npy')
        a = np.load(f'data/fake-acceleration_{data_name}.npy')
        ts = np.load(f'data/fake-time_{data_name}.npy')
    else:
        alpha = np.load(f'data/slope_{data_name}.npy')
        v = np.load(f'data/velocity_{data_name}.npy')
        a = np.load(f'data/acceleration_{data_name}.npy')
        ts = np.load(f'data/time_{data_name}.npy')
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    sys.exit(1)

dt = np.diff(ts)  # Assuming constant time step
alpha = alpha[1:]
v = v[1:]
a = a[1:]
ts = ts[1:]

plt.plot(ts,v)
plt.show()

