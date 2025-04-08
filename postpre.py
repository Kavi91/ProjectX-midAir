import numpy as np

# Load the predicted poses
predicted_file = './result/out_Kite_training_sunny_trajectory_0008.txt'
with open(predicted_file) as f:
    out = [l.strip() for l in f.readlines()]
    for i, line in enumerate(out):
        out[i] = [float(v) for v in line.split(',')]
    out = np.array(out)

# Fix axis mapping: Predicted Z -> X (North), Y -> Y (East), X -> Z (Down)
out_remapped = out.copy()
out_remapped[:, 3] = out[:, 5]  # Predicted Z -> X (North)
out_remapped[:, 4] = out[:, 4]  # Predicted Y -> Y (East)
out_remapped[:, 5] = out[:, 3]  # Predicted X -> Z (Down)

# Flip directions to match ground truth
out_remapped[:, 3] *= -1  # Flip X (North): South to North
out_remapped[:, 4] *= -1  # Flip Y (East): West to East

# Scale factor
scale_factor = 82.625

# Scale the translations (indices 3, 4, 5: x, y, z)
out_remapped[:, 3:] *= scale_factor

# Save the remapped, flipped, and scaled poses
with open('./result/out_Kite_training_sunny_trajectory_0008_remapped_flipped_scaled_corrected.txt', 'w') as f:
    for pose in out_remapped:
        f.write(', '.join([str(p) for p in pose]))
        f.write('\n')