import numpy as np
import matplotlib.pyplot as plt

# Can run this to avoid importing filter.py
def rolling_avg(data, window_size):
    return np.convolve(data, np.ones(window_size), 'same') / window_size

# Define fixed interval length in seconds (modifiable)
fixed_interval = 1  # seconds

# Define fake data for a short drive to work (~5 minutes) with traffic and turns
# Total duration: 300 seconds (5 minutes)
# Time points with stops and variations to simulate traffic
t_short = np.array([
    0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150,
    165, 180, 195, 210, 225, 240, 255, 270, 285, 300
])  # seconds

# Corresponding speeds in km/h, including stops (0 km/h) to simulate traffic
v_short = np.array([
    0, 0, 20, 40, 60, 50, 30, 0, 10, 30, 50,
    0, 10, 30, 50, 40, 0, 20, 40, 60, 0
])  # km/h

v_short = v_short / 3.6  # Convert km/h to m/s

# Increase timestep resolution with fixed interval
t_interp_short = np.arange(t_short[0], t_short[-1] + fixed_interval, fixed_interval)
v_interp_short = np.interp(t_interp_short, t_short, v_short)

# Modify the rolling average calculation
window_size = 5  # Adjust this value as needed
v_interp_short = rolling_avg(v_interp_short, window_size)

# Ensure that t_interp_short and v_interp_short have the same length
t_interp_short = t_interp_short[:len(v_interp_short)]

# Derive acceleration
a_interp_short = np.append(0, np.diff(v_interp_short)) / fixed_interval

# Generate fake slope data to account for turns (more dynamic changes)
alpha_interp_short = (np.pi/18) * np.sin(2 * np.pi * 0.005 * t_interp_short)  # Increased frequency for turns

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(t_interp_short, v_interp_short, label='Velocity (m/s)')
plt.plot(t_interp_short, a_interp_short, label='Acceleration (m/s²)')
plt.plot(t_interp_short, alpha_interp_short, label='Slope (radians)')
plt.xlabel('Time (s)')
plt.title('Fake Data for Short Drive to Work (~5 Minutes) with Traffic and Turns')
plt.legend()
plt.grid(True)
plt.show()

# Save the data to .npy files
np.save('data/fake-velocity_short_5min.npy', v_interp_short)
np.save('data/fake-acceleration_short_5min.npy', a_interp_short)
np.save('data/fake-slope_short_5min.npy', alpha_interp_short)
np.save('data/fake-time_short_5min.npy', t_interp_short)
