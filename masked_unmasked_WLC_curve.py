import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

def reflect_3d(x, y, z):
    norm = np.sqrt(x**2 + y**2 + z**2)
    if norm >= 1.0:
        x, y, z = x / norm, y / norm, z / norm
        x, y, z = x * (2 - norm), y * (2 - norm), z * (2 - norm)
    return x, y, z

def random_point_in_unit_sphere():
    while True:
        x, y, z = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        if x**2 + y**2 + z**2 <= 1:
            return x, y, z

def worm_like_chain_3d(num_steps=1000, persistence_length=0.1, step_size=0.05):
    x, y, z = random_point_in_unit_sphere()
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)
    path_x, path_y, path_z = [x], [y], [z]

    for _ in range(num_steps):
        theta += np.random.normal(0, step_size / persistence_length)
        phi += np.random.normal(0, step_size / persistence_length)
        dx = step_size * np.sin(theta) * np.cos(phi)
        dy = step_size * np.sin(theta) * np.sin(phi)
        dz = step_size * np.cos(theta)
        x, y, z = x + dx, y + dy, z + dz
        
        # Reflect if outside the unit sphere
        x, y, z = reflect_3d(x, y, z)

        path_x.append(x)
        path_y.append(y)
        path_z.append(z)

    return np.array(path_x), np.array(path_y), np.array(path_z)

def mask_intervals(curve, num_intervals=1):
    num_points = curve.shape[0]
    masked_curve = np.copy(curve)  # Create a copy of the curve to apply masking
    
    for _ in range(num_intervals):
        start_index = np.random.randint(0, num_points)
        end_index = start_index + 1
        masked_curve[start_index:end_index] = np.nan  # Mask the interval by setting points to NaN

    return masked_curve

def plot_wlc_3d_smoothed(num_chains=1, num_steps=1000, persistence_length=0.1, step_size=0.05, sigma=2, num_intervals=1):
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for _ in range(num_chains):
        path_x, path_y, path_z = worm_like_chain_3d(num_steps, persistence_length, step_size)
        
        # Apply Gaussian smoothing
        path_x_smooth = gaussian_filter1d(path_x, sigma=sigma)
        path_y_smooth = gaussian_filter1d(path_y, sigma=sigma)
        path_z_smooth = gaussian_filter1d(path_z, sigma=sigma)
        
        # Combine smoothed paths into a single array for masking
        path_smooth = np.column_stack((path_x_smooth, path_y_smooth, path_z_smooth))
        
        # Apply masking
        masked_path_smooth = mask_intervals(path_smooth, num_intervals)
        
        # Plot unmasked smoothed path
        ax1.plot(path_smooth[:, 0], path_smooth[:, 1], path_smooth[:, 2], label=f'Chain {_ + 1}')
        
        # Plot masked smoothed path
        ax2.plot(masked_path_smooth[:, 0], masked_path_smooth[:, 1], masked_path_smooth[:, 2], label=f'Chain {_ + 1}')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Unmasked Smoothed Worm-Like Chain')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Masked Smoothed Worm-Like Chain')

    plt.show()

# Plotting smoothed worm-like chain random walks in 3D with masking
plot_wlc_3d_smoothed(num_chains=1, num_steps=1000, persistence_length=0.1, step_size=0.05, sigma=2, num_intervals=20)
