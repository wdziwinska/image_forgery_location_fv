import os
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

# Dataset
class ManipulationDataset(Dataset):
    def __init__(self, img_dir, fft_dir, mask_dir, size=(256, 256)):
        self.img_dir = img_dir
        self.fft_dir = fft_dir
        self.mask_dir = mask_dir
        self.size = size
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.tif')]

        self.img_transform = T.Compose([
            T.Resize(size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(size, interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{id_}.tif")
        fft_path = os.path.join(self.fft_dir, f"{id_}_fft.png")
        mask_path = os.path.join(self.mask_dir, f"{id_}.tif")

        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Load FFT spectrum as an image
        fft_map = cv2.imread(fft_path, cv2.IMREAD_GRAYSCALE)
        fft_tensor = torch.tensor(fft_map).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]

        # Apply transformations
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        # Resize FFT tensor to match image size
        if fft_tensor.shape[-2:] != img.shape[-2:]:
            fft_tensor = torch.nn.functional.interpolate(fft_tensor.unsqueeze(0), size=img.shape[-2:], mode='bilinear',
                                                         align_corners=False)[0]
        # Combine RGB + FFT
        img_combined = torch.cat([img, fft_tensor], dim=0)  # [4, H, W]
        return img_combined, mask, id_

if __name__ == "__main__":
    # Ścieżki
    base = "split"
    train_img = os.path.join(base, "train/Datasets/defacto-inpainting/inpainting_img/img")
    train_fft = os.path.join("fft_spectrum/train/Datasets/defacto-inpainting/inpainting_img/img")
    train_mask = os.path.join(base, "train/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask")

    val_img = os.path.join(base, "val/Datasets/defacto-inpainting/inpainting_img/img")
    val_fft = os.path.join("fft_spectrum/val/Datasets/defacto-inpainting/inpainting_img/img")
    val_mask = os.path.join(base, "val/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask")

    # Dataset i DataLoader
    train_ds = ManipulationDataset(train_img, train_fft, train_mask)
    val_ds = ManipulationDataset(val_img, val_fft, val_mask)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(pretrained=True)

    # Zamiana 1. warstwy wejściowej na 4 kanały
    new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.weight[:, :3] = model.backbone.conv1.weight  # kopiujemy RGB
        new_conv1.weight[:, 3] = model.backbone.conv1.weight[:, 0] * 0.0  # inicjalizacja kanału fft
    model.backbone.conv1 = new_conv1

    # Dopasowanie wyjścia do klasy binarnej
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model = model.to(device)

    # Optymalizacja
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Trening
    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, masks, ids in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), "./trained_models/manipulation_detector_fft_v6.pt")
        print("Model zapisany jako manipulation_detector_fft_v6.pt")

    # Ewaluacja
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item()
        print(f"Validation Loss: {val_loss:.4f}")
