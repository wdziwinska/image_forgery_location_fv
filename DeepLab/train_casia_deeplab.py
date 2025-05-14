import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# === Dataset (bez FFT) ===
class CASIA2Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        filename = os.path.splitext(img_filename)[0]
        mask_path = os.path.join(self.mask_dir, filename + '_gt.png')

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maska nie znaleziona: {mask_path}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()  # binarna maska
        return image, mask

# === Transformacje ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# === DataLoader ===
dataset = CASIA2Dataset(
    img_dir='dataset/new_with_masks/train/manipulated',
    mask_dir='dataset/new_with_masks/train/groundtruth',
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# === Model ===
class ModifiedDeepLab(nn.Module):
    def __init__(self, in_channels=3):
        super(ModifiedDeepLab, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

# === Inicjalizacja ===
device = torch.device('cpu')
model = ModifiedDeepLab(in_channels=3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Trening ===
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

# === Zapis modelu ===
torch.save(model.state_dict(), 'deeplabv3_casia2_v3_rgb_only.pth')
