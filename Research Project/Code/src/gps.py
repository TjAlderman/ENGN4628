import pandas as pd
import numpy as np

def calculate_velocity(df):
    # Convert the timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # Assuming the timestamp is in seconds since epoch

    # Function to calculate distance using the Haversine formula
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371e3  # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c  # Distance in meters

    # Calculate distances
    distances = []
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)
    distances.insert(0, 0)  # No distance for the first point

    df['distance_m'] = distances

    # Calculate time differences in seconds
    df['time_diff_s'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    # Calculate velocity in m/s
    df['velocity_m_s'] = df['distance_m'] / df['time_diff_s'].replace(0, np.nan)  # Avoid division by zero

    return df
