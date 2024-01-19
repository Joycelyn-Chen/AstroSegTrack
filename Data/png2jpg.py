import os
import argparse

def replace_png_with_jpg(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.png'):
                # Replace '.png' with '.jpg'
                new_filename = filename.replace('.png', '.jpg')
                src_path = os.path.join(foldername, filename)
                dest_path = os.path.join(foldername, new_filename)

                # Rename the file
                os.rename(src_path, dest_path)
                print(f'Replaced: {src_path} -> {dest_path}')

if __name__ == "__main__":
    # Replace 'path_to_folder' with the root folder containing your directory structure
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_folder", help="The path to the dataset") 
    args = parser.parse_args()
    
    replace_png_with_jpg(args.path_to_folder)

    # python Data/png2jpg.py --path_to_folder "../Dataset/raw_img"
