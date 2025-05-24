import os
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

# Dataset
class ManipulationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(256, 256)):
        self.img_dir = img_dir
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
        mask_path = os.path.join(self.mask_dir, f"{id_}.tif")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        return img, mask

if __name__ == "__main__":
    train_img = "split/train/img"
    train_mask = "split/train/inpaint_mask"
    val_img = "split/val/img"
    val_mask = "split/val/inpaint_mask"

    # Ścieżki
    train_img = "split/train/Datasets/defacto-inpainting/inpainting_img/img"
    train_mask = "split/train/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask"
    val_img = "split/val/Datasets/defacto-inpainting/inpainting_img/img"
    val_mask = "split/val/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask"

    # Dataset i DataLoader
    train_ds = ManipulationDataset(train_img, train_mask)
    val_ds = ManipulationDataset(val_img, val_mask)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # binary output
    model = model.to(device)

    # Optymalizacja
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Trening
    for epoch in range(5):  # Zwiększ, jeśli chcesz
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f}")

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

    # Zapis modelu
    torch.save(model.state_dict(), "manipulation_detector_v1.pt")
    print("Model zapisany jako manipulation_detector_v1.pt")
