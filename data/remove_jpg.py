import os
import argparse

def remove_unmatched_jpg(folder_path):
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    for jpg_file in jpg_files:
        jpg_path = os.path.join(folder_path, jpg_file)
        png_path = os.path.join(folder_path, jpg_file.replace('.jpg', '.png'))

        if not os.path.exists(png_path):
            print(f"Removing {jpg_path} (no corresponding .png file)")
            os.remove(jpg_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="The path to the dataset") 
    args = parser.parse_args()

    remove_unmatched_jpg(args.path)

    # python Data/remove_jpg.py --path "../Dataset/astro"
