import os
import glob
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# Custom Dataset Definition
# -------------------------
class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        """
        Args:
            image_dir (str): Directory containing color satellite images.
            mask_dir (str): Directory containing binary segmentation masks.
            transform (callable, optional): Transformations to apply to images.
            mask_transform (callable, optional): Transformations to apply to masks.
        """
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*')))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()  # threshold to produce binary mask
        
        return image, mask

# -------------------------
# U-Net Model Definition
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx](x)
        
        return torch.sigmoid(self.final_conv(x))

# -------------------------
# Training Loop
# -------------------------
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return model

# -------------------------
# Main Function (for SageMaker)
# -------------------------
def main(args):
    # SageMaker training data directory; expected structure:
    #   train/images_color/ and train/masks/
    train_dir = args.train_dir  # e.g., '/opt/ml/input/data/train'
    image_dir = os.path.join(train_dir, 'images_color')
    mask_dir = os.path.join(train_dir, 'masks')
    
    # Define transformations for images and masks
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    dataset = SatelliteDataset(image_dir=image_dir, mask_dir=mask_dir,
                               transform=transform, mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Initialize the model, loss function, and optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    model = train_model(model, dataloader, criterion, optimizer, args.epochs, device)
    
    # Save the trained model to the SageMaker model directory
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print("Model saved to:", model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()
    main(args)
