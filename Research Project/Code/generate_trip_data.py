import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_realistic_velocity(num_points, interval_size, traffic_density, speed_limit):
    v = np.zeros(num_points)
    
    # Ensure the trip doesn't start with a traffic stop
    initial_segment = min(np.random.randint(np.ceil(40/interval_size), np.ceil(100/interval_size)), num_points)
    target_speed = speed_limit / 3.6  # Convert km/h to m/s
    v[:initial_segment] = np.linspace(0, target_speed, initial_segment)
    i = initial_segment

    max_acceleration = 3.0  # m/s^2, adjust as needed for realism

    while i < num_points:
        if np.random.rand() < 0.1:  # 10% chance to change speed limit
            speed_limit = np.random.choice([20, 40, 60, 80])  # km/h
        
        target_speed = speed_limit / 3.6  # Convert km/h to m/s
        
        # Determine segment length
        segment_length = np.random.randint(50, 100)  # segment length for maintained speed
        segment_end = min(i + segment_length, num_points)
        
        if np.random.rand() < traffic_density and i < num_points - 60:  # Introduce a stop, but not too close to the end
            max_stop_duration = 30  # Maximum stop duration in seconds
            stop_duration = np.random.randint(5, max_stop_duration + 1)
            stop_points = int(stop_duration / interval_size)
            
            deceleration_length = min(50, segment_end - i)
            v[i:i+deceleration_length] = np.linspace(v[i], 0, deceleration_length)
            
            stop_end = min(i + deceleration_length + stop_points, num_points)
            v[i+deceleration_length:stop_end] = 0
            
            i = stop_end
        else:  # Normal driving segment
            # Improved acceleration logic with maximum acceleration limit
            if v[i] < target_speed:
                acceleration_length = min(int((target_speed - v[i]) / 2) + 10, segment_end - i)
                acceleration_curve = np.sqrt(np.linspace(0, 1, acceleration_length))
                
                # Calculate the maximum allowed speed increase
                max_speed_increase = max_acceleration * interval_size * acceleration_length
                
                # Limit the speed increase
                speed_increase = min((target_speed - v[i]), max_speed_increase)
                
                v[i:i+acceleration_length] = v[i] + speed_increase * acceleration_curve
                i += acceleration_length
            
            # Maintain speed with small variations
            remaining_length = segment_end - i
            noise = np.random.normal(0, 0.5, remaining_length)
            v[i:segment_end] = target_speed + noise
            
            # Ensure speed doesn't exceed the limit
            v[i:segment_end] = np.clip(v[i:segment_end], 0, target_speed * 1.05)
            
            i = segment_end
    
    # Ensure end velocity is zero
    end_length = int(np.ceil(20/interval_size))  # Convert to integer
    v[-end_length:] = np.linspace(v[-end_length], 0, end_length)
    
    # Apply light smoothing
    window_size = 5
    v = np.convolve(v, np.ones(window_size)/window_size, mode='same')
    
    return v

def generate_trip_data(trip_length, interval_size, trip_name, slope_variation=0.1, traffic_density=0.5):
    # Ensure interval_size is not zero
    if interval_size <= 0:
        raise ValueError("interval_size must be greater than 0")

    num_points = int(trip_length / interval_size) + 1
    t = np.linspace(0, trip_length, num_points)
    
    # Generate multiple sections with different speed limits based on trip length
    min_section_length = 200  # minimum section length in seconds
    max_section_length = 500  # maximum section length in seconds
    avg_section_length = 300  # average section length in seconds
    
    # Determine the number of sections based on trip_length
    number_of_sections = max(int(trip_length / avg_section_length), 1)
    
    # Generate random section lengths between min and max, ensuring the total does not exceed trip_length
    section_lengths = np.random.randint(min_section_length, max_section_length + 1, size=number_of_sections)
    section_lengths = section_lengths * num_points // np.sum(section_lengths)
    
    # Define the range of possible speed limits (in km/h)
    speed_limit_min = 40
    speed_limit_max = 100  # Adjust the maximum speed limit as needed
    speed_limits = np.random.randint(speed_limit_min, speed_limit_max + 1, size=number_of_sections)
    
    v = np.array([])
    for length, speed_limit in zip(section_lengths, speed_limits):
        section_v = generate_realistic_velocity(length, interval_size, traffic_density, speed_limit)
        v = np.concatenate((v, section_v))
    
    # Trim or extend to match the desired trip length
    v = v[:num_points]
    if len(v) < num_points:
        v = np.pad(v, (0, num_points - len(v)), mode='edge')
    
    # Ensure start and end velocities are zero
    end_length = int(np.ceil(20/interval_size))  # 20s of trip length for end
    v[:end_length] = np.linspace(0, v[end_length], end_length)
    v[-end_length:] = np.linspace(v[-end_length], 0, end_length)
    
    a = np.gradient(v, t)
    
    # Generate slope data
    slope = np.cumsum(np.random.normal(0, slope_variation, num_points))
    slope = slope / np.max(np.abs(slope)) * 0.1  # Normalize to +/- 0.1 radians
    
    # Calculate distance
    d = np.cumsum(v * interval_size)
    
    # Save data to CSV
    data = np.column_stack((t, d, v, a, slope))
    np.savetxt(f"{trip_name}.csv", data, delimiter=",", header="t,d,v,a,slope", comments="")
    
    # Plot data
    plt.figure(figsize=(12, 6))
    plt.plot(t, v, label='Velocity (m/s)')
    plt.plot(t, a, label='Acceleration (m/s²)')
    plt.plot(t, slope, label='Slope (radians)')
    plt.xlabel('Time (s)')
    plt.title(f'Fake Data for Trip: {trip_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{trip_name}.png")
    plt.close()

def generate_realistic_acceleration(v, interval_size):
    a = np.diff(v) / interval_size
    a = np.insert(a, 0, 0)  # Add initial acceleration
    
    # Smooth acceleration
    window_size = 5
    a = np.convolve(a, np.ones(window_size)/window_size, mode='same')
    
    return a

def generate_realistic_slope(t, slope_variation):
    f1, f2, f3 = 0.005, 0.01, 0.02  # Different frequencies for variation
    alpha = slope_variation * (np.sin(2 * np.pi * f1 * t) + 
                               0.5 * np.sin(2 * np.pi * f2 * t) + 
                               0.3 * np.sin(2 * np.pi * f3 * t))
    
    # Add some random hills
    num_hills = int(len(t) / 500)  # Adjust as needed
    for _ in range(num_hills):
        center = np.random.randint(0, len(t))
        width = np.random.randint(50, 200)
        height = np.random.uniform(0.02, 0.05)
        alpha += height * np.exp(-((t - center) / width) ** 2)
    
    return alpha

def main():
    parser = argparse.ArgumentParser(description="Generate fake trip data")
    parser.add_argument("trip_length", type=int, help="Trip length in seconds")
    parser.add_argument("interval_size", type=float, help="Interval size in seconds")
    parser.add_argument("trip_name", type=str, help="Name of the trip")
    parser.add_argument("--slope_var", type=float, default=0.1, help="Slope variation")
    parser.add_argument("--traffic_den", type=float, default=0.5, help="Traffic density")
    args = parser.parse_args()

    generate_trip_data(args.trip_length, args.interval_size, args.trip_name, args.slope_var, args.traffic_den)

if __name__ == "__main__":
    main()

# Example usage:
# python generate_trip_data.py 300 1 trip1 --slope_var 0.1 --traffic_den 0.5
#
# This command generates trip data for a 300-second trip with 1-second intervals.
# The trip is named "trip1" with a slope variation of 0.1 and a traffic density of 0.5.
#
# Additional examples:
# python generate_trip_data.py 600 2 trip2 --slope_var 0.2 --traffic_den 0.3
# python generate_trip_data.py 900 5 trip3 --slope_var 0.15 --traffic_den 0.7
