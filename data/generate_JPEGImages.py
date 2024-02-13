import os
import shutil
import argparse

done_cases = ['SN_2016']

def copy_imgs(args):
    # Create JPEGImages directory if it doesn't exist
    if not os.path.exists(args.jpeg_images_root):
        os.makedirs(args.jpeg_images_root)

    # Open a file to write the cp commands
    with open("generate_JPEGImages.sh", "w") as cp_file:
        # Write the bash shebang line
        cp_file.write("#!/bin/bash\n\n")

        # Traverse the Annotations directory
        for root, dirs, files in os.walk(args.annotations_root):
            for file in files:
                if file.endswith(".png"):
                    # Construct the path for the corresponding JPEG image in the raw_img dataset
                    relative_path = os.path.relpath(root, args.annotations_root)

                    if relative_path.split("/")[0] in done_cases :
                        continue
                    
                    timestamp = relative_path.split("/")[-1]
                    raw_image_path = os.path.join(args.raw_img_root, timestamp, file.replace(".png", ".jpg"))
                    
                    
                    # Check if the raw image exists
                    if os.path.exists(raw_image_path):
                        # Construct the target path in the JPEGImages directory
                        target_path = os.path.join(args.jpeg_images_root, relative_path)
                        
                        # Create the target directory if it doesn't exist
                        if not os.path.exists(target_path):
                            os.makedirs(target_path)
                        
                        # Construct the cp command
                        cp_command = f"cp '{raw_image_path}' '{target_path}/'\n"
                        
                        # Write the cp command to the file
                        cp_file.write(cp_command)
                    else:
                        print(f"Warning: Corresponding raw image not found for {file}")
                        
                    

    print("The cp commands have been written to generate_JPEGImages.sh.")

if __name__ == "__main__":
    # Replace 'path_to_folder' with the root folder containing your directory structure
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_root", help="The path to the annotation root", default="/home/joy0921/Desktop/XMEM/astro/trainval/Annotations")
    parser.add_argument("--jpeg_images_root", help="The path to the jpeg root", default="/home/joy0921/Desktop/XMEM/astro/trainval/JPEGImages")
    parser.add_argument("--raw_img_root", help="The path to the raw img root", default="/home/joy0921/Desktop/Dataset/raw_img") 
    args = parser.parse_args()
    
    copy_imgs(args)

    # python data/construct_JPEGImages.py
