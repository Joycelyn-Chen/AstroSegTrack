import cv2
import os
import glob

input_root = '/home/joy0921/Desktop/XMEM/astro-davis/test-dev/JPEGImages'  # Set this to your root directory
output_root = '/home/joy0921/Desktop/Dataset/movies'
fps = 24  # Frames per second

def create_video_from_images(image_folder, output_video_file):
    images = [img for img in sorted(glob.glob(f"{image_folder}/*.jpg"))]  # Add more extensions if needed
    if not images:
        print(f"No images found in the folder {image_folder}. Skipping...")
        return

    # Get dimensions of the first image
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)

    # Initialize the video writer
    out = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)  # Write the frame to the video

    out.release()  # Release the VideoWriter object

# Traverse the root directory and process each folder
for subdir, dirs, files in os.walk(input_root):
    for dir in dirs:
        folder_path = os.path.join(subdir, dir)
        video_file = os.path.join(output_root, f"{dir}.mp4")  # Output video file name
        create_video_from_images(folder_path, video_file)

print("Video creation completed.")
