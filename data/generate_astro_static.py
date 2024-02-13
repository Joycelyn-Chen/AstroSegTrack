import os
import shutil
import argparse

def copy_images(source_folder, destination_folder):
    # Iterate through all items in the source folder
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        # If it's a file and has an image extension, copy it to the destination folder
        if os.path.isfile(item_path) and any(item_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
            shutil.copy(item_path, destination_folder)
            print(f"Copied: {item_path} to {destination_folder}")
        
        # If it's a folder, recursively call the function
        elif os.path.isdir(item_path):
            copy_images(item_path, destination_folder)




if __name__ == "__main__":
    # Replace 'path_to_folder' with the root folder containing your directory structure
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", help="The path to the dataset") 
    parser.add_argument("--dest_dir", help = "The path to the destination folder")
    args = parser.parse_args()
    
    copy_images(args.root_dir, args.dest_dir)

    # python data/generate_astro_static.py --root_dir path --dest_dir path