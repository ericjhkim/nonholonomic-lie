"""
This project is an extension of ideas presented in He and Geng's paper:
Trajectory tracking of nonholonomic mobile robots by geometric control on special Euclidean group
https://doi.org/10.1002/rnc.5561

This code simulates (holonomic) multi-agent UAV control.

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
    SIM_TIME = 20                                       # Simulation time in seconds
    dt = 0.1                                            # Time step
    SAVE_DATA = False                                    # Save data to file
    CREATE_GIF = False                                   # Create GIF

    # Directories
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f"visualizations/anim_{TIMESTAMP}.gif"

    #%% Initialization
    k_p, k_d, k_e = 1.0, 8.0, 4.0
    gains = [k_p, k_d, k_e]

    # Leader's configuration
    g0 = in_SE2([np.deg2rad(0.0), 0, 0])                # Initial pose
    xi0 = np.array([0.0, 0.0, 0.0])                     # Initial velocity
    def u_leader(t):
        # return np.array([0.1*np.sin(0.4*t), 1, 0])      # Dynamic leader control input (for Fig. 12 in [1])
        # if t < 3:
        #     return np.array([0, 1, 0])
        # elif t < 10:
        #     return np.array([0.1*np.sin(2.5*t-1.5), 1, 0])
        # else:
        #     return np.array([0, 0, 0])
        return np.array([0, 1, 0]) if t < 10 else np.array([0, 0, 0])
        # return np.array([0.1*np.sin(0.5*t-1.5), 1, 0]) if t >= 3 else np.array([0, 1, 0])      # Dynamic leader control input (for Fig. 13 in [1])
    
    # Followers' initial configurations
    gf = [in_SE2([np.deg2rad(-np.pi/2), -40, 40]),
          in_SE2([np.deg2rad(np.pi/2), -20, -25]),
          in_SE2([np.deg2rad(np.pi), -70, -10])]
    
    xif = np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0]])
    
    n_followers = 3
    # Matrix that shows each robot (row) and which other robots they follow (cols)
    following = np.array([[0, 0, 0],                    # row of zeros: only following global leader
                          [0, 0, 0],                    # row of zeros: only following global leader
                          [1, 1, 0]])                   # robot 3 is following robots 1 and 2.
    
    # Relative desired positions
    p_c_i = np.array([[-15, 15], [-15, -15], [-15, 0]])
    # p_c_i = np.zeros((n_followers, 2))                  # Set to zero for setpoint tracking

    # Trajectory data storage
    states = {
        "0":  [out_SE2(g0)]                             # Leader
    }
    for i in range(n_followers):                        # Followers
        states[str(i+1)] = [out_SE2(gf[i])]

    # Simulation
    for t in np.arange(0, SIM_TIME, dt):
        # Update system (global leader)
        u0 = u_leader(t)
        g0, xi0 = update_system(g0, xi0, u0, dt)
        states["0"].append(out_SE2(g0))

        for i in range(n_followers):
            if sum(following[i]) > 0:   # If there are local leaders
                local_leaders_i = np.nonzero(following[i])[0].tolist()
                M = len(local_leaders_i)
                G_list = [gf[x] for x in local_leaders_i]
                Xi_list = [xif[x] for x in local_leaders_i]
                lambdas = np.ones(M-1)/(M-1) if M > 2 else [0.5]
                
                g_c = calc_gc(M, G_list, lambdas)
                xi_c = calc_xic(M, Xi_list, lambdas)
            else:   # If no local leaders, follow global leader
                g_c = g0
                xi_c = xi0

            gc_bar = get_gc_bar(xi_c, p_c_i[i])         # Compute relative configuration and velocity for desired formation (offset)
            g_a = g_c @ gc_bar                          # Compute desired configuration based on the mismatch between current and desired formations (formation offset)
            g_a_i = np.linalg.inv(g_a) @ gf[i]

            # Compute control (follower)
            ui = controller(g_a_i, xi_c, xif[i], k_p, k_d, k_e, u0)
    
            # Update system (follower)
            gf[i], xif[i] = update_system(gf[i], xif[i], ui, dt)
    
            # Save trajectory
            states[str(i+1)].append(out_SE2(gf[i]))

    for key in states.keys():
        states[key] = np.array(states[key])

    # Generate gif frames
    if CREATE_GIF:
        if not os.path.exists('frames'):
            os.makedirs('frames')

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
    
    # Save data
    if SAVE_DATA:
        tools.save_to_h5py(states, gains, filename=f"data/data_{TIMESTAMP}", dataset_name="simulation")

    tools.generate_frames(states, dt, preview=True)

#%% Functions
def get_gc_bar(xi_c, p_c_i):
    """
    Computes relative configuration and velocity for desired formation.

    Parameters:
    - xi_c: Virtual leader velocity.
    - p_c_i: Desired position wrt virtual leader.

    Returns:
    - gc_bar: Relative configuration and velocity for desired formation.
    """
    omega_c, v_xc, v_yc = xi_c
    theta_c_bar = np.arctan2(omega_c*p_c_i[0], v_xc - omega_c*p_c_i[1])
    gc_bar = in_SE2([theta_c_bar, p_c_i[0], p_c_i[1]])
    return gc_bar

def calc_xic(M, Xi_list, lambdas):
    """
    Calculates virtual leader velocity xic.

    Parameters:
    - M: Number of local leader.
    - Xi_list: List of local leader velocities.
    - lambdas: List of weights.

    Returns:
    - xi_c: Virtual leader velocity.
    """
    xi_c = Xi_list[0]
    for j in range(M-1):
        xi_c = (1-lambdas[j])*xi_c + lambdas[j]*Xi_list[j+1]
    return xi_c

def calc_gc(M, G_list, lambdas):
    """
    Calculates virtual leader configuration gc.

    Parameters:
    - M: Number of local leader.
    - G_list: List of local leader configurations.
    - lambdas: List of weights.

    Returns:
    - g_c: Virtual leader configuration.
    """
    g_c = G_list[0]
    for j in range(M-1):
        g_c = g_c @ exp_map(lambdas[j]*log_map(np.linalg.inv(g_c) @ G_list[j+1]))
    return g_c

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
def controller(g01, xi0, xi1, k_p, k_d, k_e, u0):
    """
    Computes the follower control input for trajectory tracking.

    Parameters:
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
    # Relative position
    X_hat = log_map(g01)  # Logarithm map for se(2) - convert group element to algebra element
    theta01, qx01, qy01 = vee_map(X_hat)  # Vectorize

    # Leader's control inputs
    u_theta0, u_x0, u_y0 = u0

    # Relative velocity
    xi01 = xi1 - xi0
    omega01, vx01, vy01 = xi01

    # Compute control inputs using Equation (47)
    u_theta = -k_e*theta01 - k_d*omega01 + u_theta0
    u_x = -k_p*qx01 - k_d*vx01 + u_x0
    u_y = -k_p*qy01 - k_d*vy01 + u_y0

    return np.array([u_theta, u_x, u_y])

if __name__ == "__main__":
    main()
