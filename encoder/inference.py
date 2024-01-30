import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from autoencoder import Autoencoder3D  # Import your Autoencoder3D model class
import argparse
import os


# Function to visualize the original and reconstructed cube
def visualize_cube(original_cube, reconstructed_cube, output_dir):
    plt.figure(figsize=(8, 4))

    # Visualize original cube
    plt.subplot(1, 2, 1)
    plt.title('Original Cube')
    plt.imshow(original_cube[0, 0, :, :].cpu().numpy(), cmap='gray')
    
    # Visualize reconstructed cube
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Cube')
    plt.imshow(reconstructed_cube[0, 0, :, :].cpu().numpy(), cmap='gray')

    # plt.show()
    plt.savefig(os.path.join(output_dir, "reconstruct.png"))


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    autoencoder_3d = Autoencoder3D()
    autoencoder_3d.load_state_dict(torch.load(os.path.join(args.model_dir, 'autoencoder_3d_model.pth')))
    autoencoder_3d.to(device)
    autoencoder_3d.eval()

    # Define transforms for inference
    transform_inference = transforms.Compose([
        transforms.Resize((1000, 1000)),  # Adjusted size
        transforms.ToTensor()
    ])

    # Load a sample cube for inference
    sample_cube_path = args.input_dir
    sample_cube_timestamp = int(sample_cube_path.split("/")[-1])
    sample_cube = []
    for i in range(1000):
        slice_name = f'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{sample_cube_timestamp}_z{i}.jpg'
        slice_path = os.path.join(sample_cube_path, slice_name)
        image = Image.open(slice_path).convert('L')  # Convert to grayscale
        image = transform_inference(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        sample_cube.append(image)
    sample_cube = torch.cat(sample_cube, dim=1).to(device)  # Stack slices along the channel dimension

    # Perform inference
    with torch.no_grad():
        reconstructed_cube = autoencoder_3d(sample_cube)

    # Visualize the original and reconstructed cube
    visualize_cube(sample_cube, reconstructed_cube, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="The root directory for the image dataset")              # ../../Dataset/raw_img/330"
    parser.add_argument("--model_dir", help="The root directory for the model parameter input")       # "../models"
    parser.add_argument("--output_dir", help="The root directory for the reconstruction image output")          # "../../Dataset/models"

    args = parser.parse_args()
    main(args)

