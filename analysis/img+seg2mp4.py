import os
import cv2
import numpy as np

# Function to layer mask onto image
def layer_mask(image_path, mask_path, output_path):
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    red_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Assign red color (255, 0, 0) to binary regions
    red_image[mask == 255] = [0, 0, 255]

    # Resize mask to match image size if needed


    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Create a 4-channel image from mask (adding alpha channel)
    # mask_rgba = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

    # Make the masked region red
    # mask_rgba[:, :, :3] = (0, 0, 255)
    


    # Blend the image and mask
    blended = cv2.addWeighted(image, 1, red_image, 0.5, 0)

    # Write the output image
    cv2.imwrite(output_path, blended)

# Function to create movie from images in a folder
def create_movie(timestamp_folder):
    image_folder = os.path.join(timestamp_folder, 'img')
    mask_folder = os.path.join(timestamp_folder, 'mask')

    # Get list of image files
    image_files = sorted(os.listdir(image_folder))

    # Create output folder if it doesn't exist
    output_folder = os.path.join(timestamp_folder, 'movie')
    os.makedirs(output_folder, exist_ok=True)

    # Set video parameters
    video_name = os.path.basename(timestamp_folder) + '.mp4'
    video_path = os.path.join(output_folder, video_name)
    fps = 24

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))

    # Process each image and layer mask
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        mask_file = img_file.replace('.jpg', '.png')
        mask_path = os.path.join(mask_folder, mask_file)
        output_path = os.path.join(output_folder, img_file)

        # Check if mask exists for current image
        if os.path.isfile(mask_path):
            # Layer mask onto image and write to video
            layer_mask(img_path, mask_path, output_path)
            frame = cv2.imread(output_path)
            out.write(frame)

    # Release video writer
    out.release()

# Traverse through timestamp folders and create movie for each
root_folder = '../Dataset/SB230'
timestamp_folders = sorted([os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])

for timestamp_folder in timestamp_folders:
    create_movie(timestamp_folder)

print("Movies created successfully.")
