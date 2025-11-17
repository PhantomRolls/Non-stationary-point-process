import pandas as pd
import numpy as np
import csv

real_times = []

# Open file CSV
def simulate_data(file_data):
    with open(file_data, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # row[0] contains str like '2022-01-03 06:15:00.062309337'
            timestamp = row[0]
            
            # Remove data
            time_part = timestamp.split()[1]  # '06:15:00.062309337'
            
            # Eliminate ':'
            time_number_str = time_part.replace(':', '')  # '061500.062309337'
            
            # Eliminate '.' 
            time_number = int(time_number_str.replace('.', ''))  # 061500062309337
            
            # Add to array
            real_times.append(time_number) 

    # Rescale data to have them in the desired interval [0,T] 
    b = 5000
    a = 0

    # Max and Min of the sequence
    min_time = min(real_times)
    max_time = max(real_times)

    # Normalize
    times = np.array([int((t - min_time) / (max_time - min_time) * (b - a) + a) for t in real_times],  dtype=np.float64)
    return times