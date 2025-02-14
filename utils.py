"""
Utility file for geometric control functions.

@author: ericjhkim
"""

import numpy as np

#%% Functions and mappings for converting between SE(2) and se(2)
def adj_0i(g0i, xi0):
    """
    Adjoint map: Maps an element of SE(2) to its adjoint action in se(2).
    """
    R0i = g0i[:2, :2]                       # Relative rotation
    p0i = g0i[:2, 2]                        # Relative position

    p0i_hat = np.array([[0, -p0i[1]],       # Skew symmetric matrix
                        [p0i[0], 0]])
    
    omega0, vx0, vy0 = xi0                  # Extract leader's velocity components
    v0 = np.array([vx0, vy0])               # Convert to vector

    omega0_transformed = omega0             # Angular velocity remains unchanged
    v0_transformed = -R0i.T @ p0i_hat @ R0i.T @ v0 + R0i.T @ v0
    
    xi0_transformed = np.array([omega0_transformed, v0_transformed[0], v0_transformed[1]])

    return xi0_transformed

def out_SE2(g):
    """
    Takes in a matrix g and returns the corresponding pose/state vector [theta (rads), position x, position y].
    """
    theta = np.arctan2(g[1, 0], g[0, 0])
    px = g[0, 2]
    py = g[1, 2]
    return np.array([theta, px, py])

def in_SE2(x):
    """
    Takes in a vector x [theta (rads), position x, position y] and returns the corresponding SE(2) matrix.
    """
    g = np.array([[np.cos(x[0]), -np.sin(x[0]), x[1]],
                  [np.sin(x[0]), np.cos(x[0]), x[2]],
                  [0, 0, 1]])
    return g

def hat_map(x):
    """
    Hat map: Maps a vector in R^3 to an element of the Lie algebra se(2).
    """
    return np.array([[0, -x[0], x[1]],
                    [x[0], 0, x[2]],
                    [0, 0, 0]])

def vee_map(x):
    """
    Vee map: Maps an element of the Lie algebra se(2) to a vector in R^3.
    """
    return np.array([x[1, 0], x[0, 2], x[1, 2]])

# Logarithm map for se(2)
def log_map(g):
    def A_inv(theta):
        alpha = (theta/2)/np.tan(theta/2)
        return np.array([[alpha, theta/2],
                        [-theta/2, alpha]])
    
    theta = np.arctan2(g[1, 0], g[0, 0])
    theta_hat = [[0, -theta],
                 [theta, 0]]
        
    q = A_inv(theta) @ g[:2, 2]

    return np.array([[theta_hat[0][0], theta_hat[0][1], q[0]],
                     [theta_hat[1][0], theta_hat[1][1], q[1]],
                     [0, 0, 0]])

# Exponential map for SE(2)
def exp_map(X_hat):
    """Exponential map for SE(2)"""
    theta, qx, qy = vee_map(X_hat)
    if theta == 0.0:
        # Pure translation case
        return np.array([
            [1, 0, qx],
            [0, 1, qy],
            [0, 0, 1]
        ])
    else:
        # Rotation + translation
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        A = (1/theta) * np.array([
            [np.sin(theta), -(1 - np.cos(theta))],
            [1 - np.cos(theta), np.sin(theta)]
        ])
        q = A @ np.array([qx, qy])

        return np.block([
            [R, q[:, None]],
            [0, 0, 1]
        ])