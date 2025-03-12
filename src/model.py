import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        # Encoding path (Downsampling)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoding path (Upsampling + Skip Connections)
        self.decoder4 = self.deconv_block(1024, 512)
        self.decoder3 = self.deconv_block(512, 256)
        self.decoder2 = self.deconv_block(256, 128)
        self.decoder1 = self.deconv_block(128, 64)

        # Adjusted conv blocks after concatenation
        self.conv_dec4 = self.conv_block(1024, 512)  # Concatenated: 512 (decoder) + 512 (encoder)
        self.conv_dec3 = self.conv_block(512, 256)   # Concatenated: 256 (decoder) + 256 (encoder)
        self.conv_dec2 = self.conv_block(256, 128)   # Concatenated: 128 (decoder) + 128 (encoder)
        self.conv_dec1 = self.conv_block(128, 64)    # Concatenated: 64 (decoder) + 64 (encoder)
        
        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoding path with skip connections
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Concatenation
        dec4 = self.conv_dec4(dec4)

        dec3 = self.decoder3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.conv_dec3(dec3)

        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.conv_dec2(dec2)

        dec1 = self.decoder1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.conv_dec1(dec1)

        return self.output(dec1)
        
def train_model(dataloader, model, device, criterion, optimizer, num_epochs=10):
    
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, masks in dataloader:
            # Move the data to the same device as the model (GPU/CPU)
            images, masks = images.to(device), masks.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Compute the loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            
            optimizer.step()
            
            # Track the loss
            running_loss += loss.item()
        
        # Print statistics after each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
    
    print("Training completed.")
