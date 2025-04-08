import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def load_predicted_poses(file_path):
    """
    Load predicted poses from a text file.
    Each line in the file should have six comma-separated numbers:
    roll, pitch, yaw, x, y, z
    Returns:
        poses: numpy array of shape (N, 6)
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                pose = [float(x.strip()) for x in parts]
                poses.append(pose)
    return np.array(poses)

def plot_trajectory(poses, output_path):
    """
    Plot the 3D trajectory of the predicted poses and save the plot.
    This function extracts the x, y, z translation coordinates (assumed to be indices 3, 4, 5)
    and plots them in a 3D plot.
    
    Args:
      poses: numpy array of shape (N, 6)
      output_path: file path to save the plot image.
    """
    if poses.size == 0:
        print("No poses to plot.")
        return

    # Extract translation components: x, y, z
    x = poses[:, 3]
    y = poses[:, 4]
    z = poses[:, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Predicted Trajectory')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    ax.set_title('Predicted Trajectory Visualization')

    # Save the plot to the specified file
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()

if __name__ == '__main__':
    # Path for the predicted poses file
    predicted_file = "/home/krkavinda/ProjectX-midAir/result/out_Kite_training_sunny_trajectory_0008.txt"
    
    if not os.path.exists(predicted_file):
        print(f"File not found: {predicted_file}")
        exit(1)
    
    poses = load_predicted_poses(predicted_file)
    print("Loaded predicted poses shape:", poses.shape)
    print("First 5 predicted poses:\n", poses[:5])
    
    # Define the output path for the plot image
    output_plot = "/home/krkavinda/ProjectX-midAir/result/trajectory_plot_Kite_training_sunny_trajectory_0008.png"
    plot_trajectory(poses, output_plot)
