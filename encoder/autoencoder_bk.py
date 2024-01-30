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

        # Initialize ResNet-18 pre-trained model
        self.resnet18 = resnet18(pretrained=True)
        # Modify the first convolutional layer to accept 1-channel input
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Remove the fully connected layer
        self.resnet18.fc = nn.Identity()

        # Encoder with initial 3D convolution
        self.encoder_conv3d = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.encoder_relu = nn.ReLU(inplace=True)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 64, kernel_size=(1, 16, 16), stride=(1, 2, 2), padding=(0, 7, 7), output_padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 1, kernel_size=(1, 16, 16), stride=(1, 2, 2), padding=(0, 7, 7), output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # DEBUG
        print(f"X shape: {x.shape}")

        batch_size, C, D, H, W = x.shape       #x.size()
        x = self.encoder_relu(self.encoder_conv3d(x))  # Apply initial 3D conv

        # Prepare data for ResNet-18
        # x = x.view(batch_size * D, C, H // 2, W // 2)  # Adjust for ResNet-18 input, considering the effect of the 3D conv
        # x = x.view(batch_size * D, 1, H, W) 
        x = x.mean(dim=1, keepdim=True)  # Reduce across the channel dimension, resulting in [1, 1, 1000, 112, 112]
        x = x.view(-1, C, H // 2, W // 2)  # Reshape for processing, resulting in [1000, 1, 112, 112]

        # DEBUG
        print(f"X shape: {x.shape}")

        # Process each slice using ResNet-18 in a more batch-efficient manner
        # x = torch.cat([self.resnet18(x[i * D:(i + 1) * D].unsqueeze(1)) for i in range(batch_size)], dim=0)
        x = torch.stack([self.resnet18(x[i].unsqueeze(0)) for i in range(x.size(0))])

        # DEBUG
        print(f"X shape: {x.shape}")
        
        
        # Reshape the output to have the correct dimensions for the decoder
        x = x.view(batch_size, D, 512, H // 32, W // 32)  # Adjust shape back to 3D, considering ResNet-18's downsampling

        x = self.decoder(x)  # Decode
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
                cube_paths.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
        cube_paths.sort()  # Ensure the slices are in order
        return cube_paths

    def __len__(self):
        return len(self.cube_paths) // 1000  # Assuming each cube consists of 1000 slices

    def __getitem__(self, idx):
        start_idx = idx * 1000
        end_idx = start_idx + 1000
        cube_slices = []

        # Load each slice in the cube
        for i in range(start_idx, end_idx):
            slice_path = self.cube_paths[i]
            image = Image.open(slice_path).convert('L')  # Convert to grayscale
            if self.transform:
                image = self.transform(image)
            cube_slices.append(image)

        # Stack the slices along a new dimension to form a 3D cube
        cube_data = torch.stack(cube_slices, dim=0).unsqueeze(0).squeeze(2)  # Shape: [1, depth, height, width]

        #DEBUG
        print(f"Cube data size: {cube_data.shape}\n\n")

        return cube_data



def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Define transforms with resize to 224x224 and normalization
    # Calculate the mean and std for grayscale based on the ImageNet stats
    mean_gray = (0.485 + 0.456 + 0.406) / 3
    std_gray = (0.229 + 0.224 + 0.225) / 3

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize each slice to 224x224
        transforms.ToTensor(),  # Convert the PIL Image to a tensor
        transforms.Normalize(mean=[mean_gray], std=[std_gray])  # Normalize for ResNet-18
    ])

    # Create the dataset and DataLoader for 3D cubes
    dataset_3d = CustomDataset3D(root_dir=args.input_dir, transform=transform)
    dataloader_3d = DataLoader(dataset_3d, batch_size=1, shuffle=True, num_workers=4)

    # Create the updated 3D autoencoder model
    autoencoder_3d = Autoencoder3D().to(device)
    
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
