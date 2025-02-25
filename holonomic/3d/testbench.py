import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameters
dt = 0.1  # Time step (seconds)
T = 5  # Total simulation time (seconds)
timesteps = int(T / dt)

# Define drone initial state (position, velocity, attitude error)
position = np.array([0.0, 0.0])  # (x, y) in meters
velocity = np.array([0.0, 0.0])  # (vx, vy) in m/s
attitude_error = np.radians(15)  # Initial tilt error in degrees (converted to radians)

# Desired target position
target_position = np.array([10.0, 0.0])  # 10 meters ahead

# Control gains (hypothetical values)
Kp_attitude = 2.0  # Attitude control gain
Kp_position = 1.0  # Position control gain

# Store history for visualization
position_history = []
attitude_history = []
time_history = np.arange(0, T, dt)

# Simulate drone motion
for t in range(timesteps):
    # Attitude control: Exponential convergence (assuming a simple PD-like response)
    attitude_error -= Kp_attitude * attitude_error * dt
    
    # Compute thrust direction (imperfect if attitude error exists)
    thrust_direction = np.array([np.cos(attitude_error), np.sin(attitude_error)])
    
    # Position control: Apply force in the direction of the target
    position_error = target_position - position
    thrust_force = Kp_position * position_error  # Proportional force toward target
    
    # Apply thrust with attitude error affecting direction
    acceleration = thrust_force * thrust_direction  # Incorrect thrust application due to attitude error
    
    # Update velocity and position
    velocity += acceleration * dt
    position += velocity * dt
    
    # Store history
    position_history.append(position.copy())
    attitude_history.append(np.degrees(attitude_error))  # Convert to degrees for visualization

# Convert lists to arrays for plotting
position_history = np.array(position_history)

# Plot results
plt.figure(figsize=(10, 5))

# Plot position trajectory
plt.subplot(1, 2, 1)
plt.plot(position_history[:, 0], position_history[:, 1], label="Drone Path")
plt.scatter(target_position[0], target_position[1], color='red', label="Target", zorder=3)
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Drone Trajectory with Attitude Error")
plt.legend()
plt.grid()

# Plot attitude error over time
plt.subplot(1, 2, 2)
plt.plot(time_history, attitude_history, label="Attitude Error (deg)")
plt.xlabel("Time (s)")
plt.ylabel("Attitude Error (degrees)")
plt.title("Attitude Error Convergence")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
