import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.linalg import expm

SIM_TIME = 10
dt = 0.01

def get_x_d(t):
    # return np.zeros(3)
    return np.array([0.4*t, 0.4*np.sin(np.pi*t), 0.6*np.cos(np.pi*t)])

def get_b1_d(t):
    # return np.array([1,0,0])
    return np.array([np.cos(np.pi*t), np.sin(np.pi*t), 0])

def main():
    # Run the simulation
    quad = Quadrotor()
    x_d = np.array([1, 1, -5])                  # Desired position
    v_d = np.zeros(3)                           # Desired velocity
    a_d = np.zeros(3)                           # Desired acceleration
    R_d = np.eye(3)                             # Desired rotation matrix
    omega_d = np.zeros(3)                       # Desired angular velocity
    omega_dot_d = np.zeros(3)                   # Desired angular acceleration

    positions = []
    for t in np.arange(0, SIM_TIME, dt):

        x_d = get_x_d(t)

        f, M = quad.compute_thrust_and_moments(
            t,
            x_d=x_d, v_d=v_d, a_d=a_d,
            R_d=R_d, omega_d=omega_d, omega_dot_d=omega_dot_d
        )

        quad.update_state(f, M, dt)

    positions = np.array(quad.position_history)
    
    print(positions[0],positions[-1])
    plot_1d_trajectory(positions, x_d)
    plot_3d_trajectory(positions, x_d)

class Quadrotor:
    def __init__(self, mass=4.34, inertia=np.diag([0.0820, 0.0845, 0.1377]), d=0.315, c_tau_f=8.004e-4):
        self.m = mass
        self.J = inertia
        self.d = d
        self.c_tau_f = c_tau_f
        self.g = -9.81
        
        # State variables
        self.x = np.zeros(3)  # Position
        self.v = np.zeros(3)  # Velocity
        self.R = np.eye(3)    # Rotation matrix
        # self.R = np.array([[1,0,0],
        #                    [0, -0.9995, -0.0314],
        #                    [0, 0.0314, -0.9995]])

        self.omega = np.zeros(3)  # Angular velocity

        self.position_history = [self.x.copy()]
        
    def equations_of_motion(self, f, M):
        """Compute the derivatives of the state variables."""
        # acc = self.g + (f / self.m) * self.R[:, 2]
        acc = self.g*np.array([0,0,1]) - (f * self.R @ np.array([0,0,1]))/self.m
        omega_dot = np.linalg.inv(self.J) @ (M - np.cross(self.omega, self.J @ self.omega))
        return acc, omega_dot
    
    def update_state(self, f, M, dt):
        """Integrate the state forward in time using Euler's method."""
        acc, omega_dot = self.equations_of_motion(f, M)
        self.v += acc * dt
        self.x += self.v * dt
        self.omega += omega_dot * dt
        # dR = R.from_rotvec(self.omega * dt).as_matrix()
        dR = expm(self.hat_map(self.omega)*dt)
        self.R = self.R @ dR

        self.position_history.append(self.x.copy())

    def vee_map(self, R):
        """Compute the vee map of a rotation matrix."""
        theta = np.arccos((np.trace(R) - 1) / 2)

        # Handle the case where theta is very small (approximately zero)
        if np.isclose(theta, 0.0):
            return np.zeros(3)  # No rotation

        # Compute the skew-symmetric matrix of the rotation axis
        hat_k = (R - R.T) / (2 * np.sin(theta))

        # Extract the rotation vector using the vee map
        rotation_vector = theta * np.array([hat_k[2, 1], hat_k[0, 2], hat_k[1, 0]])

        return rotation_vector
    
    def hat_map(self, v):
        """Compute the hat map of a vector."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    
    def compute_thrust_and_moments(self, t, x_d, v_d, a_d, R_d, omega_d, omega_dot_d, kx=16, kv=5.6, kR=8.81, kOmega=2.54):
        """Compute the thrust and moments for tracking control."""
        # Position and velocity errors
        e_x = self.x - x_d
        e_v = self.v - v_d 

        # Calculate desired body frame axes
        desired_acc = -kx*e_x - kv*e_v + self.m*self.g*np.array([0,0,1]) + self.m*a_d
        b3d = -desired_acc/np.linalg.norm(desired_acc)

        b1d = get_b1_d(t)
        b2d = np.cross(b3d, b1d)/np.linalg.norm(np.cross(b3d, b1d))

        # Force
        f = -(-kx*e_x - kv*e_v - self.m*self.g*np.array([0,0,1]) + self.m*a_d) @ self.R @ np.array([0,0,1])

        R_d = np.column_stack([b1d, b2d, b3d])

        e_R = 0.5 * self.vee_map(R_d.T @ self.R - self.R.T @ R_d)
        e_omega = self.omega - self.R.T @ R_d @ omega_d

        omega_hat = self.hat_map(self.omega)

        # Moments
        M = -kR*e_R - kOmega*e_omega + np.cross(self.omega, self.J @ self.omega) - self.J @ (omega_hat @ self.R.T @ R_d @ omega_d - self.R.T @ R_d @ omega_dot_d)

        return f, M

def plot_3d_trajectory(positions, target_position):
    """Plot the trajectory in 3D."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Quadrotor Path', color='b')
    ax.scatter(*target_position, color='r', marker='o', label='Target Position')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("Quadrotor 3D Trajectory")
    ax.legend()
    ax.grid()
    plt.show()

def plot_1d_trajectory(positions, target_position):
    """Plot the trajectory in 3D."""
    fig = plt.figure(figsize=(8, 8))
    t_list = np.arange(0-0.01, 10, 0.01)
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(t_list, positions[:, i])
        plt.legend()
        plt.grid()
    plt.show()

if __name__ == "__main__":
    main()