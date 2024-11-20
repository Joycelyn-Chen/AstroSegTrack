import numpy as np
import argparse
import os

def convert_comma_to_space(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        in_header = True  # Flag to track whether we're in the header
        for line in infile:
            if in_header:
                outfile.write(line)  # Write the header lines unchanged
                if line.strip() == "end_header":
                    in_header = False  # Exit header mode after the end_header line
            else:
                # Replace commas with spaces for data lines
                outfile.write(line.replace(",", " "))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ply", help="The input ply to be modified")          # 
    parser.add_argument("--output_ply", help="The output ply filename")
    parser.add_argument("--data_root", help="The dataset path root")    
  
    args = parser.parse_args()


    convert_comma_to_space(os.path.join(args.data_root, args.input_ply), os.path.join(args.data_root, args.output_ply))

    # python filter_ply.py --input_ply "380.ply" --output_ply "380_cube.ply" --data_root /home/joycelyn/Desktop/3DIS/Point-SAM/demo/static/models   