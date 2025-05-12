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
    def __init__(self, img_dir, dct_dir, mask_dir, size=(256, 256)):
        self.img_dir = img_dir
        self.dct_dir = dct_dir
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

        # Wczytanie obrazu i maski
        img_path = os.path.join(self.img_dir, f"{id_}.tif")
        mask_path = os.path.join(self.mask_dir, f"{id_}.tif")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Wczytanie widma DCT
        dct_path = os.path.join(self.dct_dir, f"{id_}.npy")
        dct_map = np.load(dct_path)
        dct_tensor = torch.tensor(dct_map).unsqueeze(0).float()  # [1, H, W]

        # Transformacje
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        # Dopasowanie rozmiaru dct do obrazu (resize jako backup)
        if dct_tensor.shape[-2:] != img.shape[-2:]:
            dct_tensor = torch.nn.functional.interpolate(dct_tensor.unsqueeze(0), size=img.shape[-2:], mode='bilinear', align_corners=False)[0]

        # Połączenie RGB + DCT
        img_combined = torch.cat([img, dct_tensor], dim=0)  # [4, H, W]

        return img_combined, mask

# Ścieżki
base = "split"
train_img = os.path.join(base, "train/Datasets/defacto-inpainting/inpainting_img/img")
train_dct = os.path.join(base, "train/Datasets/defacto-inpainting/inpainting_img/dct_spectrum")
train_mask = os.path.join(base, "train/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask")

val_img = os.path.join(base, "val/Datasets/defacto-inpainting/inpainting_img/img")
val_dct = os.path.join(base, "val/Datasets/defacto-inpainting/inpainting_img/dct_spectrum")
val_mask = os.path.join(base, "val/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask")

# Dataset i DataLoader
train_ds = ManipulationDataset(train_img, train_dct, train_mask)
val_ds = ManipulationDataset(val_img, val_dct, val_mask)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(pretrained=True)

# Zamiana 1. warstwy wejściowej na 4 kanały
new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    new_conv1.weight[:, :3] = model.backbone.conv1.weight  # kopiujemy RGB
    new_conv1.weight[:, 3] = model.backbone.conv1.weight[:, 0] * 0.0  # inicjalizacja kanału DCT
model.backbone.conv1 = new_conv1

# Dopasowanie wyjścia do klasy binarnej
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
model = model.to(device)

# Optymalizacja
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Trening
for epoch in range(5):
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
    torch.save(model.state_dict(), "./trained_models/manipulation_detector_v4.pt")
    print("Model zapisany jako manipulation_detector_v4.pt")

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
