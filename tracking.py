"""
This project is a reproduction of ideas presented in He and Geng's paper:
Trajectory tracking of nonholonomic mobile robots by geometric control on special Euclidean group
https://doi.org/10.1002/rnc.5561

This code simulates nonholonomic mobile robot control.

@author: ericjhkim
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *
import tools as tools
from tqdm import tqdm
from PIL import Image

def main():
    #%% Simulation parameters
    PROBLEM = 0                                         # 0: Setpoint tracking, 1: Trajectory tracking
    SIM_TIME = 20                                       # Simulation time in seconds
    dt = 0.1                                            # Time step
    SAVE_DATA = False                                    # Save data to file
    CREATE_GIF = False                                   # Create GIF

    # Directories
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f"visualizations/anim_{TIMESTAMP}.gif"

    #%% Initialization
    k_p, k_d, k, k_e = 1.0, 2.0, 0.1, 1.0

    # Leader's configuration
    g0 = in_SE2([np.deg2rad(0.0), 0, 0])                # Initial pose
    xi0 = np.array([0.0, 0.0, 0.0])                     # Initial velocity
    def u_leader(t):
        return np.array([0.15*np.cos(0.4*t), 10, 0])    # Dynamic leader control input
        # return np.array([0.0, 0.0, 0.0])                # Leader's control input (static control)
    
    # Follower's initial configuration
    g1 = in_SE2([np.deg2rad(-90), -500, 500])           # Initial pose
    xi1 = np.array([np.deg2rad(2.0), 10.0, 0.0])        # Initial velocity

    # Trajectory data storage
    states = {
        "0": [out_SE2(g0)],                             # Leader
        "1": [out_SE2(g1)]                              # Follower
    }

    # Simulation
    for t in np.arange(0, SIM_TIME, dt):
        # Update system (leader)
        u0 = u_leader(t)
        g0, xi0 = update_system(g0, xi0, u0, dt)

        # Relative configuration and velocity
        g01 = np.linalg.inv(g0) @ g1

        # Compute control (follower)
        ui = controller(PROBLEM, g0, g1, g01, xi0, xi1, k_p, k_d, k, k_e, u0)

        # Update system (follower)
        g1, xi1 = update_system(g1, xi1, ui, dt)

        # Save trajectory
        states["0"].append(out_SE2(g0))
        states["1"].append(out_SE2(g1))

    for key in states.keys():
        states[key] = np.array(states[key])
    
    gains = [k_p, k_d, k, k_e]
    
    # Save data
    if SAVE_DATA:
        tools.save_to_h5py(states, gains, filename=f"data/data_{TIMESTAMP}", dataset_name="simulation")

    # Generate gif frames
    if CREATE_GIF:
        if not os.path.exists('frames'):
            os.makedirs('frames')

        # lims = tools.get_lims(states)
        tools.generate_frames(states, dt)

        # Stitch the frames into a single GIF
        frames = []
        frame_files = [f'frames/frame_{i:04d}.png' for i in range(len(os.listdir('frames')))]
        for frame_file in frame_files:
            frame = Image.open(frame_file)
            frames.append(frame)

        # Save the frames as a GIF
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=dt*1000, loop=0)

        # Clean up the frame images after the GIF is created
        import shutil
        shutil.rmtree('frames')

    plt.plot(states["0"][:,1], states["0"][:,2], 'b-', label='Leader', alpha=0.2)
    plt.plot(states["1"][:,1], states["1"][:,2], 'r-', label='Follower', alpha=0.2)
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.show()

#%% Functions
def update_system(gi, xi_i, ui, dt):
    """
    Updates the system's state based on the control input.

    Parameters:
    - gi: Current configuration matrix (SE(2)) of agent i.
    - xi_i: Current velocity vector (se(2)) of agent i.
    - ui: Control input vector [u_theta, u_x, u_y] for agent i.
    - dt: Time step.

    Returns:
    - Updated configuration and velocity.
    """
    xi_i_hat = hat_map(xi_i)                    # Algebra element
    gi_next = gi @ exp_map(xi_i_hat * dt)       # Update gi using matrix exponential
    xi_i_next = xi_i + ui * dt                  # Update xi_i using control input

    return gi_next, xi_i_next

# Trajectory tracking controller
def controller(PROBLEM, g0, g1, g01, xi0, xi1, k_p, k_d, k, k_e, u0):
    """
    Computes the follower control input for trajectory tracking.

    Parameters:
    - PROBLEM: Simulation type (0: Setpoint tracking, 1: Trajectory tracking).
    - g0: Leader's configuration matrix (SE(2)).
    - g1: Follower's configuration matrix (SE(2)).
    - g01: Relative configuration matrix (SE(2)).
    - xi0: Leader's velocity vector (se(2)).
    - xi1: Follower's velocity vector (se(2)).
    - k_p, k_d, k, k_e: Control gains.
    - u0: Leader's control inputs.

    Returns:
    - u: Control input vector [u_theta, u_x, u_y].
    """
    # Leader states
    theta0 = np.arctan2(g0[1, 0], g0[0, 0])

    # Follower states
    theta1 = np.arctan2(g1[1, 0], g1[0, 0])

    # Relative position
    X_hat = log_map(g01)  # Logarithm map for se(2) - convert group element to algebra element
    theta01, qx01, qy01 = vee_map(X_hat)  # Vectorize
    # theta01 = np.arctan2(g01[1, 0], g01[0, 0])
    rx01 = g01[0, 2]
    ry01 = g01[1, 2]
    # rx01 = (g1[0, 2] - g0[0, 2]) * np.cos(theta0) + (g1[1, 2] - g0[1, 2]) * np.sin(theta0)
    # ry01 = -(g1[0, 2] - g0[0, 2]) * np.sin(theta0) + (g1[1, 2] - g0[1, 2]) * np.cos(theta0)

    # Compute beta01 (to handle nonholonomic constraint)
    beta01 = np.arctan2(qy01, qx01) if qx01 != 0 or qy01 != 0 else 0 # flipped sign

    # Leader's control inputs
    u_theta0, u_x0, u_y0 = u0

    # Compute relative velocities
    omega0, vx0, vy0 = xi0  # Extract velocity components
    omega1, vx1, vy1 = xi1  # Extract velocity components
    xi01 = [omega1-omega0,
            vx1 - (vx0-omega0*ry01)*np.cos(theta01) - omega0*rx01*np.sin(theta01),
            (vx0-omega0*ry01)*np.sin(theta01) - omega0*rx01*np.cos(theta01)]
    omega01, vx01, vy01 = xi01

    # Compute the adjoint orientation correction: theta1_tilde = theta0 + theta01
    theta1_tilde = theta0 + theta01

    # Compute control inputs using Equation (47)
    u_theta = -k_e * (theta1 - theta1_tilde) - k_p * (theta01 + k * beta01) - k_d * omega01 + u_theta0
    u_x = -k_p * qx01 - k_d * vx01 + (u_x0 - u_theta0 * ry01) * np.cos(theta01) + u_theta0 * rx01 * np.sin(theta01)

    # Enforce nonholonomic constraint (no sideways motion)
    u_y = 0

    return np.array([u_theta, u_x, u_y])

if __name__ == "__main__":
    main()