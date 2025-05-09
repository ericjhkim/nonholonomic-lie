import numpy as np
import matplotlib.pyplot as plt
import tools as tools
from scipy.linalg import expm, logm

# g0 = in_SE2([np.pi/6, 5, 3])
# print(g0)
# g1 = in_SE2([np.pi/3, 8, 7])
# g1 = g0
# print(g1)
# print(np.linalg.inv(g0) @ g1)

# # Set up the figure
# fig, ax = plt.subplots(figsize=(8, 6))

# # Define the leader's trajectory (moving in a circular motion)
# t = np.linspace(0, 2 * np.pi, 100)
# leader_x = 5 + 3 * np.cos(t)
# leader_y = 5 + 3 * np.sin(t)

# # Define the follower's trajectory (following but slightly behind)
# follower_x = 4 + 3 * np.cos(t - 0.5)  # Slight phase lag
# follower_y = 4 + 3 * np.sin(t - 0.5)

# # Plot trajectories
# ax.plot(leader_x, leader_y, 'b-', label="Leader's Path", linewidth=2)
# ax.plot(follower_x, follower_y, 'r-', label="Follower's Path", linewidth=2)

# # Draw arrows to show velocity directions
# for i in range(0, len(t), 20):
#     ax.arrow(leader_x[i], leader_y[i], 
#              (leader_x[i+1] - leader_x[i]) * 0.5, 
#              (leader_y[i+1] - leader_y[i]) * 0.5, 
#              head_width=0.2, head_length=0.2, fc='blue', ec='blue')

#     ax.arrow(follower_x[i], follower_y[i], 
#              (follower_x[i+1] - follower_x[i]) * 0.5, 
#              (follower_y[i+1] - follower_y[i]) * 0.5, 
#              head_width=0.2, head_length=0.2, fc='red', ec='red')

# # Annotate
# ax.text(leader_x[10], leader_y[10], "Leader", fontsize=12, color='blue')
# ax.text(follower_x[10], follower_y[10], "Follower", fontsize=12, color='red')

# # Set labels and grid
# ax.set_xlabel("X Position")
# ax.set_ylabel("Y Position")
# ax.set_title("Leader-Follower Motion Interaction")
# ax.legend()
# ax.grid()

# # Show plot
# plt.show()

# data = tools.load_latest_h5py(filename="data", dataset_name="data", directory="data")

# def compute_xi01(g0, g1, xi0, xi1):
#     """
#     Computes the relative velocity xi01 = xi1 - Ad_g01_inv * xi0.

#     Parameters:
#     - g0: Leader's configuration (SE(2) matrix).
#     - g1: Follower's configuration (SE(2) matrix).
#     - xi0: Leader's velocity vector (se(2)).
#     - xi1: Follower's velocity vector (se(2)).

#     Returns:
#     - xi01: Relative velocity vector (se(2)).
#     """
#     # Compute relative configuration g01
#     g01 = np.linalg.inv(g0) @ g1

#     # Extract rotation matrix and position vector from g01
#     R01 = g01[:2, :2]  # Top-left 2x2 block is the rotation matrix
#     p01 = g01[:2, 2]   # Top-right 2x1 block is the translation

#     # Compute skew-symmetric matrix of p01
#     p01_hat = np.array([[0, -p01[1]],
#                         [p01[0], 0]])

#     # Compute Ad_g01_inv * xi0
#     omega0, vx0, vy0 = xi0  # Extract leader's velocity components
#     v0 = np.array([vx0, vy0])

#     # Apply the adjoint transformation
#     omega0_transformed = omega0  # Rotation part remains the same
#     v0_transformed = -R01.T @ p01_hat @ R01.T @ np.array([vx0, vy0]) + R01.T @ v0

#     # Create transformed xi0
#     xi0_transformed = np.array([omega0_transformed, v0_transformed[0], v0_transformed[1]])

#     # Compute relative velocity
#     xi01 = xi1 - xi0_transformed
#     return xi01

# print(np.sin(np.pi/6),np.arcsin(np.sin(np.pi/6)))
# print(np.cos(np.pi/6),np.arccos(np.cos(np.pi/6)))
# print(np.arctan(np.sin(np.pi/6)/np.cos(np.pi/6)))
# print(np.arctan(np.cos(np.pi/6)/np.sin(np.pi/6)))
# print(np.arctan2(np.sin(np.pi/6),np.cos(np.pi/6)))
# print(np.arctan2(np.cos(np.pi/6),np.sin(np.pi/6)))

# import utils as utils
# g = np.array([[3, 8, -1],
#               [4, -2, -3],
#               [7, 4, 0]])

# g = np.zeros((3,3))
# g[0, 0] = -1

# print(np.trace(g))

# # theta = np.arctan2(g[0, 0], g[1, 0])
# theta = np.arctan2(g[1, 0], g[0, 0])
# theta_hat = [[0, -theta],
#                 [theta, 0]]
# print(theta_hat)
# print(np.pi)

# qy01 = 3
# qx01 = 1

# print(-np.arctan2(qy01, qx01))
# print(np.arctan2(qy01, qx01))
# print(np.arctan2(qx01, qy01))
# print(-np.arctan(qy01/qx01))

# a = np.array([1, 1, 0])
# leaders = np.nonzero(a)[0].tolist()
# print(a[0,2])

# print(np.arctan(1/2))
# print(np.arctan2(1,2))
# print(np.arctan2(2,1))

# import numpy as np
# import matplotlib.pyplot as plt

# # Generate a meshgrid of values for x and y
# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
# X, Y = np.meshgrid(x, y)

# # Compute the arctan2 of Y and X
# Z = np.arctan2(Y, X)

# # Create the plot
# plt.figure(figsize=(8, 6))
# cp = plt.contourf(X, Y, Z, cmap='hsv')  # Contour plot with a hue-based color map
# plt.colorbar(cp, label='Arctan2(Y, X)')

# # Labeling the plot
# plt.title('Visualization of numpy.arctan2(Y, X)')
# plt.xlabel('X')
# plt.ylabel('Y')

# # Show the plot
# plt.show()

# A = [[1,4],[1,1]]
# print(expm(A))

def vee_map_of_rotation(R):
    """
    Computes the vee map (logarithm) of a rotation matrix R to obtain the corresponding rotation vector.

    Parameters:
    R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
    numpy.ndarray: A 3x1 rotation vector (axis-angle representation).
    """
    # Ensure R is a valid rotation matrix
    assert R.shape == (3, 3), "R must be a 3x3 matrix"
    
    # Compute the rotation angle theta
    theta = np.arccos((np.trace(R) - 1) / 2)

    # Handle the case where theta is very small (approximately zero)
    if np.isclose(theta, 0.0):
        return np.zeros(3)  # No rotation

    # Compute the skew-symmetric matrix of the rotation axis
    hat_k = (R - R.T) / (2 * np.sin(theta))

    # Extract the rotation vector using the vee map
    rotation_vector = theta * np.array([hat_k[2, 1], hat_k[0, 2], hat_k[1, 0]])

    return rotation_vector

def hat_map(v):
    """Compute the hat map of a vector."""
    return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

theta = np.pi/6
R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
print(R)
rv = vee_map_of_rotation(R)
print(rv)
print(hat_map(rv))