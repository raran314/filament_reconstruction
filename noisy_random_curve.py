import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 3D random curve
def generate_random_curve_3d(fixed_length, endpoint_distance, num_segments, visualize=True):
    # Initial angle and segment length
    segment_length = fixed_length / num_segments

    # Initialize arrays to store the x, y, and z coordinates
    x_points = np.random.rand(num_segments)
    y_points = np.random.rand(num_segments)
    z_points = np.random.rand(num_segments)

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

    # Visualize the curve
##    if visualize:
##        fig = plt.figure()
##        ax = fig.add_subplot(111, projection='3d')
##        ax.plot(x_points, y_points, z_points, 'o-')
##        ax.set_title('Random 3D Curve with Fixed Length {} and Endpoint Distance {}'.format(fixed_length, endpoint_distance))
##        ax.set_xlabel('X')
##        ax.set_ylabel('Y')
##        ax.set_zlabel('Z')
##        plt.show()

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

# Example usage
fixed_length = 10
endpoint_distance = 15
num_segments = 100
random_curve = generate_random_curve_3d(fixed_length, endpoint_distance, num_segments)
masked_curve = mask_intervals(random_curve, num_intervals=16)  # Specify the number of intervals to mask

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

# Plot unmasked curve
ax1.plot(random_curve[:, 0], random_curve[:, 1], random_curve[:, 2], label='Original Curve')
ax1.set_title('Original Curve')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot masked curve
ax2.plot(masked_curve[:, 0], masked_curve[:, 1], masked_curve[:, 2], label='Masked Curve')
ax2.set_title('Masked Curve')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.show()
