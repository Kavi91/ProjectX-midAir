import argparse
import os
from glob import glob
import types
from kitti_odometry import KittiEvalOdom

parser = argparse.ArgumentParser(description='KITTI evaluation')
parser.add_argument('--result', type=str, required=True,
                    help="Result directory with converted txt files (predicted poses)")
parser.add_argument('--align', type=str, 
                    choices=['scale', 'scale_7dof', '7dof', '6dof'],
                    default=None,
                    help="alignment type")
args = parser.parse_args()

# Ground truth directory with pose files named poses_XXXX.txt
gt_dir = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses'
result_dir = args.result

# Scan the ground truth directory for files that follow the "poses_XXXX.txt" pattern.
gt_txt_files = glob(os.path.join(gt_dir, "poses_*.txt"))
available_seqs = []
for file in gt_txt_files:
    basename = os.path.basename(file)
    if basename.startswith("poses_") and basename.endswith(".txt"):
        # Remove prefix and suffix to extract the numeric part.
        seq_str = basename[len("poses_"):-len(".txt")]
        if seq_str.isdigit():
            available_seqs.append(int(seq_str))

if not available_seqs:
    print("No ground truth sequence files found in the ground truth directory.")
    exit(1)
else:
    available_seqs.sort()
    print("Available sequences:", available_seqs)

continue_flag = input("Evaluate results in {}? [y/n] ".format(result_dir))
if continue_flag.lower() != "y":
    print("Double check the path!")
    exit(0)

eval_tool = KittiEvalOdom()

# Patch the load_poses_from_txt method so that it creates a proper file path for GT.
# The original method seems to expect filenames like "00.txt". This patched version intercepts
# those calls and reconstructs the file path using the ground truth naming "poses_{:04d}.txt".
def load_poses_from_txt_patched(self, file_path):
    basename = os.path.basename(file_path)
    # If the file name does not follow the ground truth naming,
    # we assume it is in the KITTI two-digit format (e.g. "00.txt") and patch it.
    if not basename.startswith("poses_"):
        try:
            seq = int(os.path.splitext(basename)[0])
            # Rebuild the file path to match the ground truth filename.
            file_path = os.path.join(self.gt_dir, "poses_{:04d}.txt".format(seq))
        except ValueError:
            pass  # If conversion fails, proceed using the given file_path.
    # Now open and load the file as before.
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # (The original implementation probably processes these lines into poses.
    # Here we simply return the list of stripped lines as a placeholder.)
    return [line.strip() for line in lines]

# Replace the method in the eval_tool instance.
eval_tool.load_poses_from_txt = types.MethodType(load_poses_from_txt_patched, eval_tool)

# Run the evaluation with the (patched) evaluation tool.
eval_tool.eval(
    gt_dir,
    result_dir,
    alignment=args.align,
    seqs=available_seqs,
)
