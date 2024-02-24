import os

# Paths to the root directories of the two trees
correct_tree_root = '/home/joy0921/Desktop/Evaluation/output/astro_stcn'  # Path to the tree with correct image names
wrong_tree_root = '/home/joy0921/Desktop/Evaluation/output/astro_tam'  # Path to the tree with mislabeled images

def rename_images_in_second_tree(correct_dir, wrong_dir):
    correct_images = sorted(os.listdir(correct_dir))  # Get and sort image names in the correct directory
    wrong_images = sorted(os.listdir(wrong_dir))  # Get and sort image names in the wrong directory

    for correct_image, wrong_image in zip(correct_images, wrong_images):
        if correct_image.endswith(".png") and wrong_image.endswith(".png"):  # Ensure both files are PNG images
            # Construct full paths to the images
            correct_image_path = os.path.join(correct_dir, correct_image)
            wrong_image_path = os.path.join(wrong_dir, wrong_image)
            new_wrong_image_path = os.path.join(wrong_dir, correct_image)  # New path for the wrong image

            # Rename the mislabeled image in the second tree to match the correct name from the first tree
            os.rename(wrong_image_path, new_wrong_image_path)
            print(f"Renamed {wrong_image_path} to {new_wrong_image_path}")

def iterate_and_rename(correct_root, wrong_root):
    for subdir, dirs, files in os.walk(correct_root):
        # Construct the corresponding path in the wrong tree
        relative_path = os.path.relpath(subdir, correct_root)
        corresponding_wrong_dir = os.path.join(wrong_root, relative_path)

        if os.path.exists(corresponding_wrong_dir):
            rename_images_in_second_tree(subdir, corresponding_wrong_dir)
        else:
            print(f"Warning: Corresponding directory {corresponding_wrong_dir} not found in the wrong tree.")

iterate_and_rename(correct_tree_root, wrong_tree_root)
print("Image renaming process completed.")



# import os

# root_directory = '/home/joy0921/Desktop/Evaluation/output/astro_tam'  # Set this to your root directory
# addition = {209: 763, 210: 763, 211: 763, 212: 763, 213: 763, 214: 763, 215: 763, 216: 763, 217: 763, 218: 763, 219: 763, 220: 763, 221: 763, 222: 763, 223: 763, 
#             224: 763, 225: 763, 226: 763, 227: 766, 228: 770, 229: 770, 230: 770, 231: 770}   # Number to add to the original integer value


# def rename_images_in_directory(directory, timestamp):
#     prefix = f"sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{timestamp}_z"
#     for filename in os.listdir(directory):
#         if filename.endswith(".png"):  # Check if the file is a PNG image
#             original_number = int(filename.split(".")[0])  # Convert the original filename (excluding extension) to an integer
#             new_number = original_number + addition[timestamp]  # Add 763 to the original number
#             new_filename = f"{prefix}{new_number}.png"  # Construct the new filename with the prefix and updated number, maintaining leading zeros
#             original_filepath = os.path.join(directory, filename)
#             new_filepath = os.path.join(directory, new_filename)
#             os.rename(original_filepath, new_filepath)  # Rename the file
#             print(f"Renamed {original_filepath} to {new_filepath}")

# # Iterate through all subdirectories in the root directory
# for subdir, dirs, files in os.walk(root_directory):
#     timestamp = int(subdir.split("/")[-1].split("_")[-2])
#     rename_images_in_directory(subdir, timestamp)

# print("Image renaming completed.")
