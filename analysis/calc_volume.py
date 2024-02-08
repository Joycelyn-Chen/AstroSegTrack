import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def count_white_pixels(image_path):
    """Count the number of white pixels in an image."""
    with Image.open(image_path) as img:
        # Convert image to grayscale and count pixels with a value of 255
        return sum(pixel == 255 for pixel in img.convert('L').getdata())

def process_timestamps(root_dir):
    volume_dic = {}

    # 1. Loop through all timestamps
    for timestamp in os.listdir(root_dir):
        #DEBUG
        print(f"Processing {timestamp}")
        
        timestamp_path = os.path.join(root_dir, timestamp)
        if os.path.isdir(timestamp_path):
            total_white_pixels = 0

            # 2. Loop through all mask images in the timestamp directory
            for mask_file in os.listdir(timestamp_path):
                if mask_file.endswith('.png'):  # Ensure it's a PNG image
                    image_path = os.path.join(timestamp_path, mask_file)
                    total_white_pixels += count_white_pixels(image_path)

            # Store the total volume of white pixels for the current timestamp
            volume_dic[timestamp] = total_white_pixels

    return volume_dic

def plot_results(volume_dic, output_root):
    """Plot the volume of white pixels as a function of timestamp."""
    # Sort the dictionary by timestamp to ensure the plot is in order
    sorted_timestamps = sorted(volume_dic.keys())
    volumes = [volume_dic[timestamp] for timestamp in sorted_timestamps]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_timestamps, volumes, marker='o-')
    plt.xlabel('Time (Myr)')
    plt.ylabel('Accumulated Volume (pixels)')
    plt.title('Accumulated Volume Over Time')
    plt.xticks(rotation=45)
    # plt.tight_layout()
    
    # plt.show()
    plt.savefig(os.path.join(output_root, 'volume.png'))

    #DEBUG
    print(f"Volume chart saved at: {os.path.join(output_root, 'volume.png')}")

def main(args):
    volume_dic = process_timestamps(args.mask_root)
    plot_results(volume_dic, args.output_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", help="The root directory to the dataset")          # "../Dataset"
    parser.add_argument("--output_root", help="Path to output root", default = "../../Dataset/Isolated_case")
    
  
    # python analysis/calc_volume.py --mask_root "../../Dataset/Isolated/SN_20215" 
    args = parser.parse_args()
    main(args)


