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
    def create_prism_points_3d(x, y, z, vx, vy, vz, scale=1.0):
        """Generate 3D triangular prism points oriented based on the velocity vector."""
        # Create triangle shape in the XY plane with the nose pointing in the +x direction
        triangle_base = np.array([[scale, 0, 0], [-scale * 0.5, scale * 0.5, 0], [-scale * 0.5, -scale * 0.5, 0]])

        # Normalize the velocity vector
        velocity_vector = np.array([vx, vy, vz])
        norm = np.linalg.norm(velocity_vector)
        if norm == 0:
            return triangle_base + np.array([x, y, z])  # Return the base shape if velocity is zero

        velocity_vector /= norm

        # Create rotation matrix to align the x-axis with the velocity vector
        x_axis = np.array([1, 0, 0])
        v = np.cross(x_axis, velocity_vector)
        s = np.linalg.norm(v)
        c = np.dot(x_axis, velocity_vector)

        if s != 0:
            vx_matrix = np.array([[0, -v[2], v[1]],
                                  [v[2], 0, -v[0]],
                                  [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + vx_matrix + (vx_matrix @ vx_matrix) * ((1 - c) / (s ** 2))
            rotated_triangle = (rotation_matrix @ triangle_base.T).T
        else:
            rotated_triangle = triangle_base

        return rotated_triangle + np.array([x, y, z])
    
    # Turn off interactive mode
    plt.ioff()

    min_x, max_x = np.min([states[k][:, 3] for k in states.keys()]), np.max([states[k][:, 3] for k in states.keys()])
    min_y, max_y = np.min([states[k][:, 4] for k in states.keys()]), np.max([states[k][:, 4] for k in states.keys()])
    min_z, max_z = np.min([states[k][:, 5] for k in states.keys()]), np.max([states[k][:, 5] for k in states.keys()])
    size = 0.022 * np.linalg.norm([max_x - min_x, max_y - min_y, max_z - min_z])  # Adjust prism size as needed

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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Colours
        colours = ["b", "r", "g", "c", "m", "k"]
        handles, labels = [], []
        hi = {}

        # Loop through agents
        for k, key in enumerate(states.keys()):
            px = states[key][i, 3]
            py = states[key][i, 4]
            pz = states[key][i, 5]

            # Calculate velocity from position data
            if i == 0:
                vx, vy, vz = 0, 0, 0
            else:
                vx = (states[key][i, 3] - states[key][i - 1, 3]) / dt
                vy = (states[key][i, 4] - states[key][i - 1, 4]) / dt
                vz = (states[key][i, 5] - states[key][i - 1, 5]) / dt

            # Create prism points
            prism_points = create_prism_points_3d(px, py, pz, vx, vy, vz, scale=size)

            # Draw the prism
            poly = art3d.Poly3DCollection([prism_points], color=colours[k], alpha=0.8)
            ax.add_collection3d(poly)

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