import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D  # Added for 3D plotting

# Load data for trajectory_0008
traj_id = '0008'
traj_path = f"/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/trajectory_{traj_id}/"
poses = np.load(f"/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/poses_{traj_id}.npy")
imu = np.load(f"{traj_path}/imu.npy")
gps = np.load(f"{traj_path}/gps.npy")

# Extract data
poses_pos = poses[:, 3:6]  # [x, y, z] from poses
gps_pos = gps[:, 0:3]  # [x, y, z] from GPS
imu_acc = imu[:, 0:3]  # [ax, ay, az] from IMU (Body frame)
poses_euler = poses[:, 0:3]  # [roll, pitch, yaw] for transforming IMU to World frame

# Transform IMU accelerometer to World frame
imu_acc_world = np.zeros_like(imu_acc)
for i in range(len(poses)):
    R = Rotation.from_euler('xyz', poses_euler[i]).as_matrix()
    imu_acc_world[i] = R @ imu_acc[i]

# Timestamps (25Hz)
timestamps = np.arange(len(poses)) / 25.0  # [0, 0.04, 0.08, ...]

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Pose and GPS X positions on the primary Y-axis
ax1.plot(timestamps, poses_pos[:, 0], label='Pose X Position (m)', color='blue')
ax1.plot(timestamps, gps_pos[:, 0], label='GPS X Position (m)', color='green', linestyle='--')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

# Create a secondary Y-axis for IMU acceleration
ax2 = ax1.twinx()
ax2.plot(timestamps, imu_acc_world[:, 0], label='IMU Acc X (World, m/s²)', color='red', alpha=0.5)
ax2.set_ylabel('Acceleration (m/s²)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')

# Title and layout
plt.title('Alignment of Pose, GPS, and IMU Data (Trajectory 0008)')
fig.tight_layout()

# Save the plot to a file instead of showing it
plt.savefig('/home/krkavinda/ProjectX-midAir/alignment_plot_trajectory_0008.png')
print("Plot saved as '/home/krkavinda/ProjectX-midAir/alignment_plot_trajectory_0008.png'")

# Create a 3D plot for ground truth and GPS trajectories (single plot)
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Extract x, y, z coordinates for 3D plotting (North, East, Down)
gt_x, gt_y, gt_z = poses_pos[:, 0], poses_pos[:, 1], poses_pos[:, 2]  # x (North), y (East), z (Down)
gps_x, gps_y, gps_z = gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2]  # x (North), y (East), z (Down)

# Debug: Print data ranges to check for issues
print(f"Ground Truth - North (X): min={np.min(gt_x):.2f}, max={np.max(gt_x):.2f}, mean={np.mean(gt_x):.2f}")
print(f"Ground Truth - East (Y): min={np.min(gt_y):.2f}, max={np.max(gt_y):.2f}, mean={np.mean(gt_y):.2f}")
print(f"Ground Truth - Down (Z): min={np.min(gt_z):.2f}, max={np.max(gt_z):.2f}, mean={np.mean(gt_z):.2f}")
print(f"GPS - North (X): min={np.min(gps_x):.2f}, max={np.max(gps_x):.2f}, mean={np.mean(gps_x):.2f}")
print(f"GPS - East (Y): min={np.min(gps_y):.2f}, max={np.max(gps_y):.2f}, mean={np.mean(gps_y):.2f}")
print(f"GPS - Down (Z): min={np.min(gps_z):.2f}, max={np.max(gps_z):.2f}, mean={np.mean(gps_z):.2f}")

# Calculate the mean offset between ground truth and GPS in X, Y, Z
offset = np.mean(poses_pos - gps_pos, axis=0)
print(f"Mean Offset (Ground Truth - GPS) - North (X): {offset[0]:.2f} m, East (Y): {offset[1]:.2f} m, Down (Z): {offset[2]:.2f} m")

# Plot both trajectories in the same 3D plot
ax_3d.plot(gt_x, gt_y, gt_z, label='Ground Truth Trajectory', color='green', linestyle='-')
ax_3d.plot(gps_x, gps_y, gps_z, label='GPS Trajectory', color='red', linestyle='--')

# Add a marker for the starting point (using ground truth start)
ax_3d.scatter([gt_x[0]], [gt_y[0]], [gt_z[0]], label='Sequence Start', marker='s', color='black')

# Set labels for the axes (NED convention)
ax_3d.set_xlabel('North (m)')
ax_3d.set_ylabel('East (m)')
ax_3d.set_zlabel('Down (m)')

# Compute ranges for equal aspect ratio in X and Y, but scale Z to make variation more visible
x_range = np.ptp(gt_x) + np.ptp(gps_x)
y_range = np.ptp(gt_y) + np.ptp(gps_y)
z_range = np.ptp(gt_z) + np.ptp(gps_z)
max_range_xy = max(x_range, y_range) / 2
z_range_scaled = z_range * 2  # Scale Z-axis to make variation more visible

# Set axis limits
x_mid = (np.mean(gt_x) + np.mean(gps_x)) / 2
y_mid = (np.mean(gt_y) + np.mean(gps_y)) / 2
z_mid = (np.mean(gt_z) + np.mean(gps_z)) / 2
ax_3d.set_xlim(x_mid - max_range_xy, x_mid + max_range_xy)
ax_3d.set_ylim(y_mid - max_range_xy, y_mid + max_range_xy)
ax_3d.set_zlim(z_mid - z_range_scaled, z_mid + z_range_scaled)

# Add title and legend
plt.title(f'Ground Truth vs GPS Trajectory: Kite_training/sunny/{traj_id}')
plt.legend()

# Save the 3D plot to a file
plt.savefig(f'/home/krkavinda/ProjectX-midAir/gt_vs_gps_3d_Kite_training_sunny_{traj_id}_combined.png')
print(f"3D combined plot saved as '/home/krkavinda/ProjectX-midAir/gt_vs_gps_3d_Kite_training_sunny_{traj_id}_combined.png'")

# Create a 2D plot for roll, pitch, yaw (ground truth orientations)
fig_euler = plt.figure(figsize=(12, 6))
ax_euler = fig_euler.add_subplot(111)

# Plot roll, pitch, yaw over time
ax_euler.plot(timestamps, poses_euler[:, 0], label='Roll (rad)', color='purple')
ax_euler.plot(timestamps, poses_euler[:, 1], label='Pitch (rad)', color='orange')
ax_euler.plot(timestamps, poses_euler[:, 2], label='Yaw (rad)', color='cyan')
ax_euler.set_xlabel('Time (s)')
ax_euler.set_ylabel('Orientation (rad)')
ax_euler.set_title('Ground Truth Orientations (Roll, Pitch, Yaw) Over Time')
ax_euler.legend()
ax_euler.grid(True)

# Save the Euler angles plot to a file
plt.savefig(f'/home/krkavinda/ProjectX-midAir/euler_angles_trajectory_{traj_id}.png')
print(f"Euler angles plot saved as '/home/krkavinda/ProjectX-midAir/euler_angles_trajectory_{traj_id}.png'")