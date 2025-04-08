#!/usr/bin/env python3
import os
import argparse
import numpy as np

def convert_npy_to_txt(directory):
    # Iterate over all files in the directory.
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            npy_path = os.path.join(directory, filename)
            try:
                # Load the .npy file.
                data = np.load(npy_path)
            except Exception as e:
                print(f"Error loading file {npy_path}: {e}")
                continue

            # Replace the .npy extension with .txt.
            txt_filename = filename.replace('.npy', '.txt')
            txt_path = os.path.join(directory, txt_filename)
            try:
                # Save the array to .txt using a default float format.
                np.savetxt(txt_path, data, fmt='%f')
                print(f"Converted {npy_path} to {txt_path}")
            except Exception as e:
                print(f"Error saving file {txt_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert pose files in .npy format to .txt format."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses",
        help="Directory containing .npy files (default: provided poses directory)"
    )
    args = parser.parse_args()

    # Call the conversion function with the specified directory.
    convert_npy_to_txt(args.directory)
