import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import os
import argparse
import time
from architecture import Encoder3D, Decoder3D

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# Version 3:
class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        self.encoder = Encoder3D()
        self.decoder = Decoder3D()

    def forward(self, x):
        # Encode the input 3D cube to a 2D feature representation
        encoded_features = self.encoder(x)
        
        # Decode the 2D features back to the 3D cube
        reconstructed_cube = self.decoder(encoded_features)
        
        return reconstructed_cube


# Version 2:
# class Autoencoder3D(nn.Module):
#     def __init__(self):
#         super(Autoencoder3D, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         # Assuming x is the full 3D cube of size (1000, 1000, 1000)
#         processed_slices = []
#         for i in range(x.size(0)):  # Iterate through each depth slice
#             slice = TF.to_pil_image(x[i])  # Convert slice to PIL Image for downsampling
#             slice = TF.resize(slice, (224, 224))  # Downsample slice
#             slice = TF.to_tensor(slice).unsqueeze(0)  # Convert back to tensor and add batch dimension
#             slice = self.encoder(slice)  # Encode slice
#             processed_slices.append(slice)

#         # Decode each processed slice and stack to reconstruct the 3D volume
#         reconstructed_slices = [self.decoder(slice) for slice in processed_slices]
#         reconstructed_volume = torch.stack(reconstructed_slices, dim=0)

#         return reconstructed_volume

# Version 3:
class CustomDataset3D(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cube_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.cube_folders)

    def __getitem__(self, idx):
        cube_dir = self.cube_folders[idx]
        images = [os.path.join(cube_dir, f) for f in os.listdir(cube_dir) if f.endswith('.jpg')]
        images.sort()  # Ensure the slices are in order

        slices = []
        for image_path in images:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(image)
            slices.append(image)

        cube = torch.stack(slices)  # Stack along the new dimension

        # cube = cube.unsqueeze(1)  # Add a channel dimension
        
        return cube
    

# Version 2:
# class CustomDataset3D(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.cube_paths = self.get_cube_paths()

#     def get_cube_paths(self):
#         cube_paths = []
#         for folder in os.listdir(self.root_dir):
#             folder_path = os.path.join(self.root_dir, folder)
#             if os.path.isdir(folder_path):
#                 cube_paths.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
#         cube_paths.sort()  # Ensure the slices are in order
#         return cube_paths

#     def __len__(self):
#         return len(self.cube_paths) // 1000  # Assuming each cube consists of 1000 slices

#     def __getitem__(self, idx):
#         start_idx = idx * 1000
#         end_idx = start_idx + 1000
#         cube_slices = []

#         # Load each slice in the cube
#         for i in range(start_idx, end_idx):
#             slice_path = self.cube_paths[i]
#             image = Image.open(slice_path).convert('L')  # Convert to grayscale
#             if self.transform:
#                 image = self.transform(image)
#             cube_slices.append(image)

#         # Stack the slices along a new dimension to form a 3D cube
#         cube_data = torch.stack(cube_slices, dim=0).unsqueeze(0).squeeze(2)  # Shape: [1, depth, height, width]

#         #DEBUG
#         print(f"Cube data size: {cube_data.shape}\n\n")

#         return cube_data

def save_reconstructed_slices(reconstructed_cube, save_dir):
    # Check if the save directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Move the cube to CPU and remove the batch dimension
    reconstructed_cube = reconstructed_cube.squeeze(0).cpu()

    # Normalize the cube to the [0, 255] range for JPG saving
    min_val = reconstructed_cube.min()
    max_val = reconstructed_cube.max()
    reconstructed_cube = (reconstructed_cube - min_val) / (max_val - min_val) * 255

    for i, slice in enumerate(reconstructed_cube):
        # Convert the tensor slice to a PIL Image
        slice_img = Image.fromarray(slice.numpy().astype('uint8'))

        # Save the image
        slice_img.save(os.path.join(save_dir, f"slice_{i+1:04d}.jpg"))


# Version 3:
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_gray = (0.485 + 0.456 + 0.406) / 3
    std_gray = (0.229 + 0.224 + 0.225) / 3

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean_gray], std=[std_gray])
    ])

    # Load dataset
    dataset_3d = CustomDataset3D(root_dir=args.input_dir, transform=transform)
    dataloader_3d = DataLoader(dataset_3d, batch_size=1, shuffle=True, num_workers=4)

    # Initialize model, loss function, and optimizer
    autoencoder_3d = Autoencoder3D().to(device)
    criterion_3d = nn.MSELoss()
    optimizer_3d = optim.Adam(autoencoder_3d.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.num_epoch):
        epoch_start_time = time.time()
        for data_3d in dataloader_3d:
            data_3d = data_3d.to(device)
            output_3d = autoencoder_3d(data_3d)
            loss_3d = criterion_3d(output_3d, data_3d)

            optimizer_3d.zero_grad()
            loss_3d.backward()
            optimizer_3d.step()

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch + 1}/{args.num_epoch}], Loss: {loss_3d.item():.4f}, Time: {epoch_time:.2f}s')

    # Save model
    torch.save(autoencoder_3d.state_dict(), os.path.join(ensure_dir(args.output_dir), 'autoencoder_3d_model.pth'))

    # Inference with the last cube in the dataset
    with torch.no_grad():
        autoencoder_3d.eval()
        last_cube = dataset_3d[-1].to(device)  # Assuming the last cube is what you want to reconstruct
        reconstructed_cube = autoencoder_3d(last_cube.unsqueeze(0))  # Add a batch dimension

        # Call the function to save the reconstructed slices
        save_reconstructed_slices(reconstructed_cube, ensure_dir(args.result_dir))


# Version 2:
# def main(args):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # # Define transforms with resize to 224x224 and normalization
#     # Calculate the mean and std for grayscale based on the ImageNet stats
#     mean_gray = (0.485 + 0.456 + 0.406) / 3
#     std_gray = (0.229 + 0.224 + 0.225) / 3

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize each slice to 224x224
#         transforms.ToTensor(),  # Convert the PIL Image to a tensor
#         transforms.Normalize(mean=[mean_gray], std=[std_gray])  # Normalize for ResNet-18
#     ])

#     # Create the dataset and DataLoader for 3D cubes
#     dataset_3d = CustomDataset3D(root_dir=args.input_dir, transform=transform)
#     dataloader_3d = DataLoader(dataset_3d, batch_size=1, shuffle=True, num_workers=4)

#     # Create the updated 3D autoencoder model
#     autoencoder_3d = Autoencoder3D().to(device)
    
#     # Define loss function and optimizer
#     criterion_3d = nn.MSELoss()
#     optimizer_3d = optim.Adam(autoencoder_3d.parameters(), lr=0.001)

#     # Training loop for 3D cubes
#     num_epochs_3d = 10
#     for epoch in range(num_epochs_3d):
#         for data_3d in dataloader_3d:
#             data_3d = data_3d.to(device)

#             # Forward pass
#             output_3d = autoencoder_3d(data_3d)
#             loss_3d = criterion_3d(output_3d, data_3d)
            
#             # Backward pass and optimization
#             optimizer_3d.zero_grad()
#             loss_3d.backward()
#             optimizer_3d.step()

#         print(f'Epoch [{epoch+1}/{num_epochs_3d}], Loss: {loss_3d.item():.4f}')

#     # Save the trained 3D autoencoder model
#     torch.save(autoencoder_3d.state_dict(), os.path.join(ensure_dir(args.output_dir), 'autoencoder_3d_model.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="The root directory for the hdf5 dataset", default = "../Dataset/raw_img")              # ../../Dataset/raw_img"
    parser.add_argument("--output_dir", help="The root directory for the network parameter output", default = "models")          # "models"
    parser.add_argument("--result_dir", help="The root directory for the reconstruction output", default = "results")          # "results"
    parser.add_argument("--lr", help="Learning rate for training the auto encoder", default = 0.001, type = float)              # 0.001
    parser.add_argument("--num_epoch", help="Training epoch for training the auto encoder", default = 20, type = int)              # 20

    # python encoder/autoencoder.py --input_dir "../Dataset/tmp" --output_dir "models" --result_dir "results" --lr 0.001 --num_epoch 5

    args = parser.parse_args()
    main(args)
