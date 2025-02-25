"""
Tools file for plotting and data manipulation.

@author: ericjhkim
"""

import os
import re
import numpy as np
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D, art3d

#%% Plotting
def generate_frames(states, dt, preview=False):
    """
    Create frames for the GIF animation with heading indicators in 3D.
    """
    def create_cross_with_circles_3d(x, y, z, theta_x, theta_y, theta_z, arm_length=1.0, circle_radius=0.2, num_points=100):
        """Generate 3D cross with circles oriented based on the theta angles."""
        # Create cross points in the XY plane
        cross_base = np.array([[arm_length, 0, 0], [-arm_length, 0, 0], [0, arm_length, 0], [0, -arm_length, 0]])

        # Apply additional +45 degrees rotation to the cross base
        theta_z += np.pi/4

        # Create rotation matrices for each axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]])
        
        R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]])
        
        R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]])

        # Combine the rotation matrices
        rotation_matrix = R_z @ R_y @ R_x

        # Rotate the cross base
        rotated_cross = (rotation_matrix @ cross_base.T).T

        # Create circles at the ends of the cross
        angles = np.linspace(0, 2 * np.pi, num_points)
        circle_base = np.array([[circle_radius * np.cos(angle), circle_radius * np.sin(angle), 0] for angle in angles])
        circles = []
        for point in rotated_cross:
            rotated_circle = (rotation_matrix @ circle_base.T).T + point
            circles.append(rotated_circle + np.array([x, y, z]))

        return rotated_cross + np.array([x, y, z]), circles
    
    # Turn off interactive mode
    plt.ioff()

    min_x, max_x = np.min([states[k][:, 3] for k in states.keys()]), np.max([states[k][:, 3] for k in states.keys()])
    min_y, max_y = np.min([states[k][:, 4] for k in states.keys()]), np.max([states[k][:, 4] for k in states.keys()])
    min_z, max_z = np.min([states[k][:, 5] for k in states.keys()]), np.max([states[k][:, 5] for k in states.keys()])
    radius = 0.022 * np.linalg.norm([max_x - min_x, max_y - min_y, max_z - min_z])  # Adjust circle radius as needed
    arm_length = 0.022 * np.linalg.norm([max_x - min_x, max_y - min_y, max_z - min_z])  # Adjust arm length as needed
    circle_radius = arm_length * 0.2  # Adjust circle radius as needed

    lims = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    for key in states.keys():
        for l in range(3):
            min_value = min(states[key][:, l+3])
            max_value = max(states[key][:, l+3])
            if min_value < lims[l][0]:
                lims[l][0] = min_value
            if max_value > lims[l][1]:
                lims[l][1] = max_value
    buffer_x = 0.1 * (lims[0][1] - lims[0][0])
    buffer_y = 0.1 * (lims[1][1] - lims[1][0])
    buffer_z = 0.1 * (lims[2][1] - lims[2][0])
    lims[0][0] -= buffer_x
    lims[0][1] += buffer_x
    lims[1][0] -= buffer_y
    lims[1][1] += buffer_y
    lims[2][0] -= buffer_z
    lims[2][1] += buffer_z

    # Loop through time steps (1 frame per timestep)
    num_timesteps = len(states[[k for k in states.keys()][0]])
    if preview:
        myrange = range(num_timesteps-1, num_timesteps)
        dpi = 200
    else:
        myrange = range(0, num_timesteps)
        dpi = 300
    for i in tqdm(myrange, desc="Generating GIF Frames", unit="frame"):

        t = i * dt

        # Configure plotting area
        fig = plt.figure(figsize=(8, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'Simulation Time: {t:.2f}s')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')

        # Colours
        colours = ["b", "r", "g", "c", "m", "k"]
        handles, labels = [], []
        hi = {}

        # Loop through agents
        for k, key in enumerate(states.keys()):
            px = states[key][i, 3]
            py = states[key][i, 4]
            pz = states[key][i, 5]

            # Extract theta angles from state vector
            theta_x = states[key][i, 0]
            theta_y = states[key][i, 1]
            theta_z = states[key][i, 2]

            # Create cross with circles
            cross_points, circles = create_cross_with_circles_3d(px, py, pz, theta_x, theta_y, theta_z, arm_length=arm_length, circle_radius=circle_radius)

            # Draw the cross
            ax.plot([cross_points[0, 0], cross_points[1, 0]], [cross_points[0, 1], cross_points[1, 1]], [cross_points[0, 2], cross_points[1, 2]], color=colours[k], linewidth=2)
            ax.plot([cross_points[2, 0], cross_points[3, 0]], [cross_points[2, 1], cross_points[3, 1]], [cross_points[2, 2], cross_points[3, 2]], color=colours[k], linewidth=2)

            # Draw the circles
            for circle in circles:
                poly = art3d.Poly3DCollection([circle], color=colours[k])
                ax.add_collection3d(poly)

            # Highlight one arm to indicate heading
            ax.plot([cross_points[0, 0], cross_points[1, 0]], [cross_points[0, 1], cross_points[1, 1]], [cross_points[0, 2], cross_points[1, 2]], color='k', linewidth=2)

            # Draw trajectory
            for j in range(i + 1):
                if j > 0:
                    x = [states[key][j, 3], states[key][j - 1, 3]]
                    y = [states[key][j, 4], states[key][j - 1, 4]]
                    z = [states[key][j, 5], states[key][j - 1, 5]]
                    ax.plot(x, y, z, c=colours[k], linewidth=0.9)

            # Draw initial position
            x = states[key][0, 3]
            y = states[key][0, 4]
            z = states[key][0, 5]
            ax.scatter(x, y, z, edgecolors=colours[k], marker='o', s=18, facecolors='none')

            # Draw final position
            x = states[key][-1, 3]
            y = states[key][-1, 4]
            z = states[key][-1, 5]
            ax.scatter(x, y, z, c=colours[k], marker='x', s=16)

            # Assign handles for the legend
            hi[key] = ax.scatter([], [], [], c=colours[k], marker='s')

        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])

        handles.extend([hi[k] for k in hi.keys()])
        labels.extend(np.concatenate((["Leader"], [f"Follower {k+1}" for k in range(len(hi.keys()))])))

        # Save the current frame as an image
        ax.set_axisbelow(True)  # grid z-order
        ax.yaxis.grid(color='gray')

        plt.figlegend(handles, labels, loc="upper left", ncol=1, fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the legend
        plt.gca().set_aspect('equal')

        if preview:
            continue
        else:
            plt.savefig(f'frames/frame_{i:04d}.png')
            plt.close(fig)  # Close the figure to prevent it from displaying
    
    if preview:
        plt.show()

#%% Data Management
def save_to_h5py(data, gains, filename="sim_data", dataset_name="data"):
    """
    Saves a numpy array to an HDF5 file using h5py, with a timestamp in the filename.

    Parameters:
        data (dict): The data dictionary to save (has leader and follower trajectories).
        filename (str): Base name of the file (default: 'sim_data').
        dataset_name (str): Name of the dataset inside the HDF5 file (default: 'data').
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")
    
    # Generate a timestamp
    filename = f"{filename}.h5"

    with h5py.File(filename, "w") as h5file:
        for key, value in data.items():
            h5file.create_dataset(key, data=value)

        h5file.create_dataset("gains", data=gains)
        # for key, value in gains.items():
        #     h5file.create_dataset(key, data=value)
    
    # The data is saved in multiple datasets, one for each leader/follower.
    # Each dataset is an integer string from 0 (leader) to n (followers).
    # Each dataset contains an array of length T x 3, where T is the number of timesteps, and 3 is the states (theta, x, y) in the global frame.
    
    print(f"Data saved to {filename}.")

def load_latest_h5py(filename="sim_data", dataset_name="data", directory="data"):
    """
    Loads the most recent HDF5 file based on the timestamp in the filename.

    Parameters:
        filename (str): Base name of the file to look for (default: 'sim_data').
        dataset_name (str): Name of the dataset inside the HDF5 file (default: 'data').
        directory (str): Directory to search for the files (default: 'data').

    Returns:
        numpy.ndarray: The loaded data array from the latest file.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    # Compile a regex to match files with timestamps
    pattern = re.compile(f"{re.escape(filename)}_(\\d{{8}}_\\d{{6}})\\.h5$")
    
    # List all files in the specified directory
    files = os.listdir(directory)
    
    # Filter and extract valid timestamped files
    timestamped_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            timestamped_files.append((file, match.group(1)))
    
    if not timestamped_files:
        raise FileNotFoundError(f"No files matching the pattern {filename}_<timestamp>.h5 found in '{directory}'.")
    
    # Sort files by timestamp
    timestamped_files.sort(key=lambda x: datetime.strptime(x[1], "%Y%m%d_%H%M%S"), reverse=True)
    latest_file = timestamped_files[0][0]
    
    # Load the latest file
    file_path = os.path.join(directory, latest_file)
    data = {}
    with h5py.File(file_path, "r") as h5file:
        datasetnames = h5file.keys()
        for dataset_name in datasetnames:
            data[dataset_name] = h5file[dataset_name][:]
    
    print(f"Loaded data from {file_path}.")
    return data