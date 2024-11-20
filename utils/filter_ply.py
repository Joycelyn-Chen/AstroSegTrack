import numpy as np
import argparse
import os

def filter_ply(input_file, output_file, x_range, y_range, z_range):
    with open(input_file, 'r') as f:
        # Read header
        header = []
        while True:
            line = f.readline().strip()
            header.append(line)
            if line == "end_header":
                break
        
        # Read point data
        data = []
        for line in f:
            values = line.strip().split(",")
            x, y, z = map(float, values[:3])
            r, g, b = map(int, values[3:])
            data.append((x, y, z, r, g, b))

    # Convert data to numpy array for filtering
    data = np.array(data, dtype=[('x', float), ('y', float), ('z', float),
                                  ('r', int), ('g', int), ('b', int)])
    
    # Apply filter
    filtered_data = data[(x_range[0] < data['x']) & (data['x'] < x_range[1]) &
                         (y_range[0] < data['y']) & (data['y'] < y_range[1]) &
                         (z_range[0] < data['z']) & (data['z'] < z_range[1])]

    # Write filtered points back to a new .ply file
    with open(output_file, 'w') as f:
        for line in header:
            if line.startswith("element vertex"):
                f.write(f"element vertex {len(filtered_data)}\n")
            else:
                f.write(line + '\n')

        # Write filtered points
        for point in filtered_data:
            f.write(f"{point['x']},{point['y']},{point['z']},{point['r']},{point['g']},{point['b']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ply", help="The input ply to be modified")          # 
    parser.add_argument("--output_ply", help="The output ply filename")
    parser.add_argument("--data_root", help="The dataset path root")
    parser.add_argument("--x1", help="lower x", default=100, type=int)
    parser.add_argument("--x2", help="upper x", default=170, type=int)
    parser.add_argument("--y1", help="lower y", default=140, type=int)
    parser.add_argument("--y2", help="upper y", default=200, type=int)
    parser.add_argument("--z1", help="lower z", default=100, type=int)
    parser.add_argument("--z2", help="upper z", default=170, type=int)
    
  
    args = parser.parse_args()

    x_range = (args.x1, args.x2)
    y_range = (args.y1, args.y2)
    z_range = (args.z1, args.z2)

    filter_ply(os.path.join(args.data_root, args.input_ply), os.path.join(args.data_root, args.output_ply), x_range, y_range, z_range)

    # python filter_ply.py --input_ply "380.ply" --output_ply "380_cube.ply" --data_root /home/joycelyn/Desktop/3DIS/Point-SAM/demo/static/models   