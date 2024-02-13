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
    parser.add_argument("--duplicates", help = "Indicate if the images are duplicated in the folder (img + mask)")
    args = parser.parse_args()
    
    total_images = count_images(args.path_to_folder)
    if args.duplicates == True:
        total_images = int(total_images / 2)
    print(f'Total number of images in {args.path_to_folder}: {total_images}')

    # python Data/count_img.py --path_to_folder path --duplicates True
