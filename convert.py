import os
import numpy as np
from helper import eulerAnglesToRotationMatrix

# Define directories
input_dir = "/home/krkavinda/ProjectX-CrossModalAttn/result/"
output_dir = "kitti-odom-eval/result/test_02/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through sequences from 00 to 10
for i in range(11):
    seq = f"{i:02d}"
    input_file = 'result/out_Kite_training_sunny_trajectory_0008.txt'
    output_file =  '/home/krkavinda/ProjectX-midAir/kitti-odom-eval/result/test/00.txt'
    
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Skipping.")
        continue

    print(f"Processing {input_file}...")
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Assuming values are comma separated:
            parts = [float(x) for x in line.split(",") if x.strip()]
            if len(parts) != 6:
                print("Skipping line (unexpected number of values):", line)
                continue
            # First three values are Euler angles, last three are translation
            euler_angles = parts[:3]
            translation = parts[3:]
            
            # Convert Euler angles to rotation matrix
            R = eulerAnglesToRotationMatrix(euler_angles)
            
            # Form the 3x4 transformation matrix [R | t]
            T = np.concatenate([R, np.array(translation).reshape(3, 1)], axis=1)
            
            # Flatten the matrix into 12 numbers (row-major order) and write space separated
            T_flat = T.flatten()
            fout.write(" ".join(map(str, T_flat)) + "\n")
    print(f"Saved converted file to {output_file}")

