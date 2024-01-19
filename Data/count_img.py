import os
import argparse

def count_images(root_folder, allowed_extensions=('png', 'jpg', 'jpeg', 'gif', 'bmp')):
    total_images = 0

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            # Check if the file has an allowed image extension
            if any(filename.lower().endswith(ext) for ext in allowed_extensions):
                total_images += 1

    return total_images

if __name__ == "__main__":
    # Replace 'path_to_folder' with the root folder containing your directory structure
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_folder", help="The path to the dataset") 
    args = parser.parse_args()
    
    total_images = count_images(args.path_to_folder)
    print(f'Total number of images in the tree structure: {total_images / 2}')
