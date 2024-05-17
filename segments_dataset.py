##import numpy as np
##import pandas as pd
##
### Generate 3D random curve
##def generate_random_curve_3d(fixed_length, endpoint_distance, num_segments):
##    # Initial angle and segment length
##    segment_length = fixed_length / num_segments
##
##    # Initialize arrays to store the x, y, and z coordinates
##    x_points = np.random.rand(num_segments)
##    y_points = np.random.rand(num_segments)
##    z_points = np.random.rand(num_segments)
##
##    # Randomly generate direction angles for each segment
##    for i in range(1, num_segments):
##        theta = np.random.rand() * 2 * np.pi  # Azimuthal angle
##        phi = np.arccos(2*np.random.rand() - 1)  # Polar angle, ensures uniform distribution on the sphere
##
##        # Update the coordinates based on the segment's direction and length
##        x_points[i] = x_points[i-1] + segment_length * np.sin(phi) * np.cos(theta)
##        y_points[i] = y_points[i-1] + segment_length * np.sin(phi) * np.sin(theta)
##        z_points[i] = z_points[i-1] + segment_length * np.cos(phi)
##
##    # Calculate the actual endpoint
##    actual_distance = np.sqrt(x_points[-1]**2 + y_points[-1]**2 + z_points[-1]**2)
##
##    # Scale the curve to fit the endpoint distance
##    scaling_factor = endpoint_distance / actual_distance
##    x_points *= scaling_factor
##    y_points *= scaling_factor
##    z_points *= scaling_factor
##
##    # Moving average smoothing
##    window_size = 2
##    x_points = np.convolve(x_points, np.ones(window_size)/window_size, mode='valid')
##    y_points = np.convolve(y_points, np.ones(window_size)/window_size, mode='valid')
##    z_points = np.convolve(z_points, np.ones(window_size)/window_size, mode='valid')
##
##    points = np.column_stack((x_points, y_points, z_points))
##
##    return points
##
### Mask intervals in the curve
##def mask_intervals(curve, num_intervals=1):
##    num_points = len(curve)
##    masked_curve = np.copy(curve)  # Create a copy of the curve to apply masking
##    
##    for _ in range(num_intervals):
##        start_index = np.random.randint(0, num_points)
##        end_index = start_index + 1
##        masked_curve[start_index:end_index] = np.nan  # Mask the interval by setting points to NaN
##
##    return masked_curve
##
### Parameters
##num_curves = 10000
##fixed_length = 10
##endpoint_distance = 15
##num_segments = 100
##num_intervals_to_mask = 16
##
##all_unmasked_curves = []
##all_masked_curves = []
##
##for _ in range(num_curves):
##    curve = generate_random_curve_3d(fixed_length, endpoint_distance, num_segments)
##    formatted_curve = [f'({x},{y},{z})' for x, y, z in curve]
##    all_unmasked_curves.append(formatted_curve)
##    
##    masked_curve = mask_intervals(curve, num_intervals=num_intervals_to_mask)
##    formatted_masked_curve = [f'({x},{y},{z})' if not np.isnan(x) else '(nan,nan,nan)' for x, y, z in masked_curve]
##    all_masked_curves.append(formatted_masked_curve)
##
### Convert to DataFrames
##df_unmasked = pd.DataFrame(all_unmasked_curves)
##df_masked = pd.DataFrame(all_masked_curves)
##
### Save to CSV files
##df_unmasked.to_csv('random_3d_curves_unmasked.csv', index=False, header=False)
##df_masked.to_csv('random_3d_curves_masked.csv', index=False, header=False)
##
##print("CSV files with 10,000 random 3D curves saved as 'random_3d_curves_unmasked.csv' and 'random_3d_curves_masked.csv'.")

import numpy as np

# Generate 3D random curve
def generate_random_curve_3d(fixed_length, endpoint_distance, num_segments):
    # Initial angle and segment length
    segment_length = fixed_length / num_segments

    # Initialize arrays to store the x, y, and z coordinates
    x_points = np.zeros(num_segments)
    y_points = np.zeros(num_segments)
    z_points = np.zeros(num_segments)

    # Randomly generate direction angles for each segment
    for i in range(1, num_segments):
        theta = np.random.rand() * 2 * np.pi  # Azimuthal angle
        phi = np.arccos(2*np.random.rand() - 1)  # Polar angle, ensures uniform distribution on the sphere

        # Update the coordinates based on the segment's direction and length
        x_points[i] = x_points[i-1] + segment_length * np.sin(phi) * np.cos(theta)
        y_points[i] = y_points[i-1] + segment_length * np.sin(phi) * np.sin(theta)
        z_points[i] = z_points[i-1] + segment_length * np.cos(phi)

    # Calculate the actual endpoint
    actual_distance = np.sqrt(x_points[-1]**2 + y_points[-1]**2 + z_points[-1]**2)

    # Scale the curve to fit the endpoint distance
    scaling_factor = endpoint_distance / actual_distance
    x_points *= scaling_factor
    y_points *= scaling_factor
    z_points *= scaling_factor

    # Moving average smoothing
    window_size = 2
    x_points = np.convolve(x_points, np.ones(window_size)/window_size, mode='valid')
    y_points = np.convolve(y_points, np.ones(window_size)/window_size, mode='valid')
    z_points = np.convolve(z_points, np.ones(window_size)/window_size, mode='valid')

    points = np.column_stack((x_points, y_points, z_points))

    return points

# Mask intervals in the curve
def mask_intervals(curve, num_intervals=1):
    num_points = len(curve)
    masked_curve = np.copy(curve)  # Create a copy of the curve to apply masking
    
    for _ in range(num_intervals):
        start_index = np.random.randint(0, num_points)
        end_index = start_index + 1
        masked_curve[start_index:end_index] = np.nan  # Mask the interval by setting points to NaN

    return masked_curve

# Parameters
num_curves = 10000
fixed_length = 10
endpoint_distance = 15
num_segments = 100
num_intervals_to_mask = 16

all_unmasked_curves = []
all_masked_curves = []

for _ in range(num_curves):
    curve = generate_random_curve_3d(fixed_length, endpoint_distance, num_segments)
    formatted_curve = [f'({x},{y},{z})' for x, y, z in curve]
    all_unmasked_curves.append(formatted_curve)
    
    masked_curve = mask_intervals(curve, num_intervals=num_intervals_to_mask)
    formatted_masked_curve = [f'({x},{y},{z})' if not np.isnan(x) else '(nan,nan,nan)' for x, y, z in masked_curve]
    all_masked_curves.append(formatted_masked_curve)

# Convert to numpy arrays
unmasked_curves_array = np.array(all_unmasked_curves, dtype=object)
masked_curves_array = np.array(all_masked_curves, dtype=object)

# Save to CSV files
np.savetxt('random_3d_curves_unmasked.csv', unmasked_curves_array, fmt='%s', delimiter=',')
np.savetxt('random_3d_curves_masked.csv', masked_curves_array, fmt='%s', delimiter=',')

print("CSV files with 10,000 random 3D curves saved as 'random_3d_curves_unmasked.csv' and 'random_3d_curves_masked.csv'.")
