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

#%% Plotting
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_frames(states, dt):
    """
    Create frames for the GIF animation with heading indicators.
    """
    # Turn off interactive mode
    plt.ioff()

    min_x, max_x = np.min([states[k][:, 1] for k in states.keys()]), np.max([states[k][:, 1] for k in states.keys()])
    min_y, max_y = np.min([states[k][:, 2] for k in states.keys()]), np.max([states[k][:, 2] for k in states.keys()])
    size = 0.022*np.linalg.norm([max_x - min_x, max_y - min_y])  # Adjust line length as needed

    lims = [[0.0, 0.0], [0.0, 0.0]]
    for key in states.keys():
        for l in range(2):
            min_value = min(states[key][:, l+1])
            max_value = max(states[key][:, l+1])
            if min_value < lims[l][0]:
                lims[l][0] = min_value
            if max_value > lims[l][1]:
                lims[l][1] = max_value
    buffer_x = 0.1 * (lims[0][1] - lims[0][0])
    buffer_y = 0.1 * (lims[1][1] - lims[1][0])
    lims[0][0] -= buffer_x
    lims[0][1] += buffer_x
    lims[1][0] -= buffer_y
    lims[1][1] += buffer_y

    # Loop through time steps (1 frame per timestep)
    num_timesteps = len(states[[k for k in states.keys()][0]])
    for i in tqdm(range(num_timesteps), desc="Generating GIF Frames", unit="frame"):

        t = i * dt

        # Configure plotting area
        fig = plt.figure(figsize=(5, 4), dpi=300)
        ax = fig.add_subplot(111)
        ax.set_title(f'Simulation Time: {t:.2f}s')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Colours
        colours = ["b", "r", "g", "c", "m", "k"]
        handles, labels = [], []
        hi = {}

        # Loop through agents
        for k, key in enumerate(states.keys()):
            px = states[key][i, 1]
            py = states[key][i, 2]

            # Extract heading from SE(2) matrix
            theta = states[key][i, 0]

            # # Compute heading line
            # hx = px + heading_length * np.cos(theta)
            # hy = py + heading_length * np.sin(theta)

            # # Plot point position
            # hi[key] = ax.scatter(px, py, c=colours[k], marker='.', s=20)

            # # Draw heading indicator
            # ax.plot([px, hx], [py, hy], c=colours[k], linewidth=1.8, alpha=0.8)
            
            # Triangle shape (elongated)
            p1 = (px + 1.5*size * np.cos(theta), py + size * np.sin(theta))  # Tip of the triangle
            p2 = (px + (-0.5 * size) * np.cos(theta + np.pi / 2), py + (-0.5 * size) * np.sin(theta + np.pi / 2))  # Bottom left
            p3 = (px + (-0.5 * size) * np.cos(theta - np.pi / 2), py + (-0.5 * size) * np.sin(theta - np.pi / 2))  # Bottom right

            # Draw triangle
            ax.fill([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color=colours[k], alpha=0.8)

            # Draw trajectory
            for j in range(i + 1):
                if j > 0:
                    x = [states[key][j, 1], states[key][j - 1, 1]]
                    y = [states[key][j, 2], states[key][j - 1, 2]]
                    ax.plot(x, y, c="k", linewidth=0.9)

            # Draw initial position
            x = states[key][0, 1]
            y = states[key][0, 2]
            ax.scatter(x, y, edgecolors=colours[k], marker='o', s=18, facecolors='none')

            # Draw final position
            x = states[key][-1, 1]
            y = states[key][-1, 2]
            ax.scatter(x, y, c=colours[k], marker='x', s=16)

            # Assign handles for the legend
            hi[key] = ax.scatter([], [], c=colours[k], marker='s')

        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])

        handles.extend([hi[k] for k in hi.keys()])
        labels.extend(np.concatenate((["Leader"], [f"Follower {k+1}" for k in range(len(hi.keys()))])))

        # Save the current frame as an image
        ax.set_axisbelow(True)  # grid z-order
        ax.yaxis.grid(color='gray')

        plt.figlegend(handles, labels, loc="upper left", ncol=1, fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the legend
        plt.gca().set_aspect('equal')

        plt.savefig(f'frames/frame_{i:04d}.png')
        plt.close(fig)  # Close the figure to prevent it from displaying

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