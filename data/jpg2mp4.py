import cv2
import os
# import glob
import argparse

DEBUG = True

parser = argparse.ArgumentParser(
                    prog='jog2mp4',
                    description='This code takes a folder of images and convert them into a single mp4 movie to the designated location',
                    epilog='Input: path to image folder, output movie filename and path')


parser.add_argument('-i', '--input_folder') 
parser.add_argument('-o', '--output_file')  

args = parser.parse_args()

# Provide the path to the input image folder, output video file, and desired FPS
input_folder = args.input_folder
output_file = args.output_file
fps = 24  # Frames per second

def convert_images_to_video(input_folder, output_file, fps):
    # Get the list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])
    sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(input_folder, sorted_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate over each image and write it to the video
    for image_file in sorted_files:
        if (DEBUG):
            print(image_file)
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer and close the video file
    video.release()
    cv2.destroyAllWindows()

print("Video creation completed.")


# Call the function to convert the images to video
convert_images_to_video(input_folder, output_file, fps)
