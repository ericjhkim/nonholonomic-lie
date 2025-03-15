import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tools as tools
from tqdm import tqdm
from PIL import Image
from scipy.linalg import expm, logm

# Update the main function to use SE(3)
def main():
    #%% Simulation parameters
    SIM_TIME = 20                                       # Simulation time in seconds
    dt = 0.1                                            # Time step
    SAVE_DATA = True                                    # Save data to file
    CREATE_GIF = True                                   # Create GIF

    # Directories
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f"visualizations/anim_{TIMESTAMP}.gif"

    #%% Initialization
    k_p, k_d, k_e = 1.0, 2.0, 1.0
    gains = [k_p, k_d, k_e]

    # Leader's configuration
    g0 = in_SE3([0.0, 0.0, 0.0, 0, 0, 0])                # Initial pose
    xi0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])       # Initial velocity
    def u_leader(t):
        return np.array([np.deg2rad(1.0), 0, np.deg2rad(2.0), 2, 1, 9.8+0.5]) if t < 10 else np.array([0, 0, 0, 0, 0, 9.8])
    
    # Followers' initial configurations
    gf = [in_SE3([np.deg2rad(-10.0), np.deg2rad(0.0), np.deg2rad(90.0), -40, 40, 40]),
          in_SE3([np.deg2rad(0.0), np.deg2rad(-15.0), np.deg2rad(45.0), -20, -25, 10]),
          in_SE3([np.deg2rad(5.0), np.deg2rad(-5.0), np.deg2rad(-90.0), -70, -10, 5])]
    
    xif = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    n_followers = 3
    following = np.array([[0, 0, 0],                    # row of zeros: only following global leader
                          [0, 0, 0],                    # row of zeros: only following global leader
                          [1, 1, 0]])                   # robot 3 is following robots 1 and 2.
    
    p_c_i = np.array([[10, 10, 0], [20, 0, 10], [0, -10, -10]])

    states = {
        "0":  [out_SE3(g0)]                             # Leader
    }
    for i in range(n_followers):                        # Followers
        states[str(i+1)] = [out_SE3(gf[i])]

    for t in np.arange(0, SIM_TIME, dt):
        u0 = u_leader(t)
        g0, xi0 = update_system(g0, xi0, u0, dt)
        states["0"].append(out_SE3(g0))

        for i in range(n_followers):
            if sum(following[i]) > 0:
                local_leaders_i = np.nonzero(following[i])[0].tolist()
                M = len(local_leaders_i)
                G_list = [gf[x] for x in local_leaders_i]
                Xi_list = [xif[x] for x in local_leaders_i]
                lambdas = np.ones(M-1)/(M-1) if M > 2 else [0.5]
                
                g_c = calc_gc(M, G_list, lambdas)
                xi_c = calc_xic(M, Xi_list, lambdas)
            else:
                g_c = g0
                xi_c = xi0

            gc_bar = get_gc_bar(xi_c, p_c_i[i])
            g_a = g_c @ gc_bar
            g_a_i = np.linalg.inv(g_a) @ gf[i]

            ui = controller(g_a_i, xi_c, xif[i], k_p, k_d, k_e, u0)
            gf[i], xif[i] = update_system(gf[i], xif[i], ui, dt)
            states[str(i+1)].append(out_SE3(gf[i]))

    for key in states.keys():
        states[key] = np.array(states[key])

    if CREATE_GIF:
        if not os.path.exists('frames'):
            os.makedirs('frames')

        tools.generate_frames(states, dt)

        frames = []
        frame_files = [f'frames/frame_{i:04d}.png' for i in range(len(os.listdir('frames')))]
        for frame_file in frame_files:
            frame = Image.open(frame_file)
            frames.append(frame)

        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=dt*1000, loop=0)

        import shutil
        shutil.rmtree('frames')
    
    if SAVE_DATA:
        tools.save_to_h5py(states, gains, filename=f"data/data_{TIMESTAMP}", dataset_name="simulation")

    # print(states["0"][-1][-1], states["1"][-1][-1], states["2"][-1][-1], states["3"][-1][-1])
    tools.generate_frames(states, dt, preview=True)

def in_SE3(pose):
    """
    Converts a pose [theta_x, theta_y, theta_z, x, y, z] to an SE(3) matrix.
    """
    theta_x, theta_y, theta_z, x, y, z = pose
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = [x, y, z]
    return g

def out_SE3(g):
    """
    Converts an SE(3) matrix to a pose [theta_x, theta_y, theta_z, x, y, z].
    """
    R = g[:3, :3]
    x, y, z = g[:3, 3]
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    theta_z = np.arctan2(R[1, 0], R[0, 0])
    return [theta_x, theta_y, theta_z, x, y, z]

def hat_map_SE3(xi):
    """
    Converts a velocity vector [omega_x, omega_y, omega_z, v_x, v_y, v_z] to an se(3) matrix.
    """
    omega_x, omega_y, omega_z, v_x, v_y, v_z = xi
    xi_hat = np.array([[0, -omega_z, omega_y, v_x],
                       [omega_z, 0, -omega_x, v_y],
                       [-omega_y, omega_x, 0, v_z],
                       [0, 0, 0, 0]])
    return xi_hat

def vee_map_SE3(xi_hat):
    """
    Converts an se(3) matrix to a velocity vector [omega_x, omega_y, omega_z, v_x, v_y, v_z].
    """
    omega_x = xi_hat[2, 1]
    omega_y = xi_hat[0, 2]
    omega_z = xi_hat[1, 0]
    v_x = xi_hat[0, 3]
    v_y = xi_hat[1, 3]
    v_z = xi_hat[2, 3]
    return [omega_x, omega_y, omega_z, v_x, v_y, v_z]

def exp_map_SE3(xi_hat):
    """
    Computes the matrix exponential of an se(3) matrix.
    """
    return expm(xi_hat)

def log_map_SE3(g):
    """
    Computes the matrix logarithm of an SE(3) matrix.
    """
    return logm(g)

# Update the controller function to handle SE(3)
def controller(g01, xi0, xi1, k_p, k_d, k_e, u0):
    """
    Computes the follower control input for trajectory tracking in SE(3).

    Parameters:
    - g01: Relative configuration matrix (SE(3)).
    - xi0: Leader's velocity vector (se(3)).
    - xi1: Follower's velocity vector (se(3)).
    - k_p, k_d, k_e: Control gains.
    - u0: Leader's control inputs.

    Returns:
    - u: Control input vector [u_omega_x, u_omega_y, u_omega_z, u_v_x, u_v_y, u_v_z].
    """
    # Relative position
    X_hat = log_map_SE3(g01)  # Logarithm map for se(3)
    theta01_x, theta01_y, theta01_z, q_x01, q_y01, q_z01 = vee_map_SE3(X_hat)  # Vectorize

    # Leader's control inputs
    u_omega_x0, u_omega_y0, u_omega_z0, u_v_x0, u_v_y0, u_v_z0 = u0

    # Relative velocity
    xi01 = xi1 - xi0
    omega01_x, omega01_y, omega01_z, v_x01, v_y01, v_z01 = xi01

    # Compute control inputs
    u_theta_x = -k_e*theta01_x - k_d*omega01_x + u_omega_x0
    u_theta_y = -k_e*theta01_y - k_d*omega01_y + u_omega_y0
    u_theta_z = -k_e*theta01_z - k_d*omega01_z + u_omega_z0
    u_x = -k_p*q_x01 - k_d*v_x01 + u_v_x0
    u_y = -k_p*q_y01 - k_d*v_y01 + u_v_y0
    u_z = -k_p*q_z01 - k_d*v_z01 + u_v_z0

    return np.array([u_theta_x, u_theta_y, u_theta_z, u_x, u_y, u_z+9.8])

# Update the update_system function to handle SE(3)
def update_system(gi, xi_i, ui, dt):
    """
    Updates the system's state based on the control input in SE(3).

    Parameters:
    - gi: Current configuration matrix (SE(3)) of agent i.
    - xi_i: Current velocity vector (se(3)) of agent i.
    - ui: Control input vector [u_omega_x, u_omega_y, u_omega_z, u_v_x, u_v_y, u_v_z] for agent i.
    - dt: Time step.

    Returns:
    - Updated configuration and velocity.
    """
    ui[5] -= 9.8
    xi_i_hat = hat_map_SE3(xi_i)                    # Algebra element
    gi_next = gi @ exp_map_SE3(xi_i_hat * dt)       # Update gi using matrix exponential
    xi_i_next = xi_i + ui * dt                      # Update xi_i using control input

    return gi_next, xi_i_next

def calc_gc(M, G_list, lambdas):
    """
    Calculates virtual leader configuration gc in SE(3).

    Parameters:
    - M: Number of local leaders.
    - G_list: List of local leader configurations.
    - lambdas: List of weights.

    Returns:
    - g_c: Virtual leader configuration.
    """
    g_c = G_list[0]
    for j in range(M-1):
        g_c = g_c @ exp_map_SE3(lambdas[j] * log_map_SE3(np.linalg.inv(g_c) @ G_list[j+1]))
    return g_c

def calc_xic(M, Xi_list, lambdas):
    """
    Calculates virtual leader velocity xic in SE(3).

    Parameters:
    - M: Number of local leaders.
    - Xi_list: List of local leader velocities.
    - lambdas: List of weights.

    Returns:
    - xi_c: Virtual leader velocity.
    """
    xi_c = Xi_list[0]
    for j in range(M-1):
        xi_c = (1 - lambdas[j]) * xi_c + lambdas[j] * Xi_list[j+1]
    return xi_c

def get_gc_bar(xi_c, p_c_i):
    """
    Computes relative configuration and velocity for desired formation in SE(3).

    Parameters:
    - xi_c: Virtual leader velocity (se(3)).
    - p_c_i: Desired position with respect to the virtual leader.

    Returns:
    - gc_bar: Relative configuration and velocity for desired formation (SE(3)).
    """
    omega_c, v_xc, v_yc, v_zc = xi_c[:3], xi_c[3], xi_c[4], xi_c[5]
    theta_c_bar = np.arctan2(omega_c[1]*p_c_i[0] - omega_c[0]*p_c_i[1], v_xc - omega_c[2]*p_c_i[2])
    gc_bar = in_SE3([theta_c_bar, 0, 0, p_c_i[0], p_c_i[1], p_c_i[2]])
    return gc_bar

if __name__ == "__main__":
    main()