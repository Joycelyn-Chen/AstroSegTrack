import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from torchsummary import summary
from PIL import Image
import os
import argparse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Define the updated 3D Autoencoder model
class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()

        # Encoder (ResNet-18 for all slices)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            resnet18(pretrained=True),  # Replace with resnet18 for each slice
            nn.AdaptiveAvgPool2d((1, 1))  # Output 1x1 feature map for the entire cube
        )

        # Decoder (2D transpose convolutional layer)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output should be in the range [0, 1]
        )

    def forward(self, x):
        # Reshape the input data to (batch_size, channels, depth, height, width)
        x = x.unsqueeze(2)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Custom dataset class for 3D cubes
class CustomDataset3D(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cube_paths = self.get_cube_paths()

    def get_cube_paths(self):
        cube_paths = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                cube_paths.append(folder_path)
        return cube_paths

    def __len__(self):
        return len(self.cube_paths)

    def __getitem__(self, idx):
        cube_path = self.cube_paths[idx]
        cube_timestamp = int(cube_path.split("/")[-1])
        cube_slices = []

        # Load each slice in the cube
        for i in range(1000):
            slice_name = f'sn34_smd132_bx5_pe300_hdf5_plt_cnt_0{cube_timestamp}_z{i}.jpg'
            slice_path = os.path.join(cube_path, slice_name)
            image = Image.open(slice_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(image)
            cube_slices.append(image.unsqueeze(0))  # Add channel dimension

        # Stack the slices along the channel dimension to form a 3D cube
        cube_data = torch.cat(cube_slices, dim=0)
        return cube_data


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),  # Adjusted size
        transforms.ToTensor()
    ])

    # Create the updated 3D autoencoder model
    autoencoder_3d = Autoencoder3D().to(device)

    # Create the dataset and DataLoader for 3D cubes
    dataset_3d = CustomDataset3D(root_dir=args.input_dir, transform=transform)
    dataloader_3d = DataLoader(dataset_3d, batch_size=1, shuffle=True, num_workers=4)

    # Define loss function and optimizer
    criterion_3d = nn.MSELoss()
    optimizer_3d = optim.Adam(autoencoder_3d.parameters(), lr=0.001)

    # Training loop for 3D cubes
    num_epochs_3d = 10
    for epoch in range(num_epochs_3d):
        for data_3d in dataloader_3d:
            data_3d = data_3d.to(device)

            # Forward pass
            output_3d = autoencoder_3d(data_3d)
            loss_3d = criterion_3d(output_3d, data_3d)
            
            # Backward pass and optimization
            optimizer_3d.zero_grad()
            loss_3d.backward()
            optimizer_3d.step()

        print(f'Epoch [{epoch+1}/{num_epochs_3d}], Loss: {loss_3d.item():.4f}')

    # Save the trained 3D autoencoder model
    torch.save(autoencoder_3d.state_dict(), os.path.join(ensure_dir(args.output_dir), 'autoencoder_3d_model.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="The root directory for the hdf5 dataset", default = "../Dataset/raw_img")              # ../../Dataset/raw_img"
    parser.add_argument("--output_dir", help="The root directory for the network parameter output", default = "models")          # "models"

    # python encoder/autoencoder.py --input_dir "../Dataset/raw_img" --output_dir "models"

    args = parser.parse_args()
    main(args)
