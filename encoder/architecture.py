import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# class SimpleResNetBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(SimpleResNetBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_channels)

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity  # Skip connection
#         out = self.relu(out)

#         return out

class SimplifiedResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(SimplifiedResNet18, self).__init__()
        self.in_channels = 64
        # Adjusted for 1-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Version 3:
class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        # Reduce spatial dimensions and depth
        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=(7, 7, 7), stride=(5, 5, 5), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=(5, 5, 5), stride=(4, 4, 4), padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(32, 1, kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=(1, 1, 1))
        
        # Flatten the output and apply 2D convolutions
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 1, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Assuming x is of shape [batch_size, channels=1, depth=1000, height=1000, width=1000]
        x = self.relu(self.conv3d_1(x))
        x = self.relu(self.conv3d_2(x))
        x = self.relu(self.conv3d_3(x))
        # After 3D convolutions, shape will be approximately [batch_size, 1, depth, height, width],
        # where depth is significantly reduced, and height and width are around 224
        
        # Select a 2D slice from the reduced volume. Here we choose the middle slice for simplicity.
        # TODO: might need to sum or average or sth to choose one slice that represents the entire cube
        x = x[:, :, x.size(2) // 2, :, :]
        
        # Apply 2D convolutions to get the final 2D feature map of shape [1, 224, 224]
        x = self.relu(self.conv2d_1(x))
        x = self.conv2d_2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Version 2:
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         # Initialize the SimplifiedResNet18 with BasicBlock and layers configuration for ResNet-18
#         self.simplified_resnet18 = SimplifiedResNet18(BasicBlock, [2, 2, 2, 2])

#     def forward(self, x):
#         # Assuming x is a batch of 2D slices from the 3D cube, each slice of size (1000, 1000)
#         # Downsample each slice before feeding it into the SimplifiedResNet18
#         downsampled_slices = [F.interpolate(slice.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False) for slice in x]
#         downsampled_slices = torch.cat(downsampled_slices, dim=0)  # Re-stack the batch

#         # Pass each downsampled slice through the SimplifiedResNet18
#         encoded_slices = self.simplified_resnet18(downsampled_slices)

#         return encoded_slices

# Version 1:
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.downsample = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.resnet_block = SimpleResNetBlock(64)

#     def forward(self, x):
#         x = self.downsample(x)  # Assuming x is a single depth slice of size (1000, 1000)
#         x = self.resnet_block(x)  # Process through ResNet block
#         return x

# Version 3:
class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        # Start with 2D upsampling
        self.conv2d_1 = nn.ConvTranspose2d(1, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2d_2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Expand to 3D using 3D transposed convolutions
        self.conv3d_1 = nn.ConvTranspose3d(1, 32, kernel_size=(4, 4, 4), stride=(4, 4, 4), padding=(1, 1, 1))
        self.conv3d_2 = nn.ConvTranspose3d(32, 16, kernel_size=(5, 5, 5), stride=(4, 4, 4), padding=(1, 1, 1))
        self.conv3d_3 = nn.ConvTranspose3d(16, 1, kernel_size=(7, 7, 7), stride=(5, 5, 5), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Assuming x is the 2D feature map of shape [batch_size, channels=1, height=224, width=224]
        x = self.relu(self.conv2d_1(x))
        x = self.conv2d_2(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # Add a new depth dimension and unsqueeze to prepare for 3D convolutions
        x = x.unsqueeze(2)
        
        # Apply 3D transposed convolutions to reconstruct the original 3D volume
        x = self.relu(self.conv3d_1(x))
        x = self.relu(self.conv3d_2(x))
        x = self.conv3d_3(x)
        return x

# Version 2:
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         # Incremental upsampling for spatial dimensions (height and width)
#         self.spatial_upsample = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample to (448, 448)
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample to (896, 896)
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample to (1792, 1792)
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # Upsample to (3584, 3584)
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # Upsample to (7168, 7168)
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # Upsample to (1000, 1000) with cropping
#             nn.Sigmoid()
#         )

#         # Separate 1D upsampling for depth
#         self.depth_upsample = nn.Sequential(
#             nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1),  # Upsample depth to 2
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1),  # Upsample depth to 4            
#             nn.ReLU(inplace=True),
#             # TODO: Additional upsampling layers here: repeat above 500 times to retrieve 1000 slices
#         )
#         # Final layer for fine adjustment to reach 1000 depth, considering padding and output_padding carefully
#         self.final_depth_adjust = nn.ConvTranspose1d(1, 1, kernel_size=3, stride=250, padding=1, output_padding=1)


#     def forward(self, x):
#         # Assuming x is the encoded feature of shape (batch_size=1, channels=1, height=224, width=224)
#         # Reshape and permute to prepare for depth upsampling
#         x = x.squeeze().unsqueeze(1)  # Change to (1, 1, 224) for depth upsampling, treating height as 'depth'
        
#         x = self.spatial_upsample(x)  # Upsample spatial dimensions

#         # Now x is (1, 1, 1000, 1000), reshape for depth upsampling
#         x = x.squeeze(1)  # Squeeze to get (1, 1000, 1000)
#         x = x.permute(0, 2, 1)  # Permute to get (1, 1000, 1000) as (batch, channels, depth)

#         # Upsample depth dimension
#         x = self.depth_upsample(x)
#         # Final adjustment to reach the target depth of 1000
#         x = self.final_depth_adjust(x)  # Resulting size should be (1, 1, 1000)
        
#         # Permute and unsqueeze to prepare for replication across the spatial dimensions
#         x = x.permute(0, 2, 1).unsqueeze(3)  # Change to (1, 1000, 1, 1)
        
#         # Replicate the depth features across the spatial dimensions to get the final (1000, 1000, 1000) volume
#         x = x.repeat(1, 1, 224, 224)  # Now x is (1, 1000, 224, 224)
        

#         # # Final reshape to (1000, 1000, 1000)
#         # x = x.permute(0, 2, 1)  # Permute back to (1, 1000, 1000)
#         # x = x.view(1000, 1000, 1000)  # Reshape to get final 3D volume

#         return x