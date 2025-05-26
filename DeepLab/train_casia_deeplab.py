import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np


# === Dataset z DCT w formacie .npy ===
class CASIA2Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, dct_dir=None, transform=None, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dct_dir = dct_dir
        self.transform = transform
        self.augment = augment

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))
        ])

        if self.augment:
            self.augment_pipeline = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        base_name = os.path.splitext(img_filename)[0]

        img_path = os.path.join(self.img_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, base_name + '_gt.png')
        dct_path = os.path.join(self.dct_dir, base_name + '_dct.npy')

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Brak maski: {mask_path}")
        if not os.path.exists(dct_path):
            raise FileNotFoundError(f"Brak DCT: {dct_path}")

        # === Wczytaj obraz i maskę
        image = Image.open(img_path).convert('RGB').resize((256, 256))
        mask = Image.open(mask_path).convert('L').resize((256, 256))

        # === Augmentacja RGB (tylko obraz, nie maska ani DCT)
        if self.augment:
            image = self.augment_pipeline(image)

        image_tensor = transforms.ToTensor()(image)  # RGB: (3, H, W)

        # === Wczytaj DCT z pliku .npy
        dct_array = np.load(dct_path).astype(np.float32)  # (3, H, W)
        dct_array -= dct_array.min()
        dct_array /= dct_array.max() + 1e-8
        dct_tensor = torch.from_numpy(dct_array)  # (3, H, W)

        combined = torch.cat([image_tensor, dct_tensor], dim=0)  # (6, H, W)

        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor > 0).float()

        return combined, mask_tensor


# === Model ===
class ModifiedDeepLab(nn.Module):
    def __init__(self, in_channels=6):
        super(ModifiedDeepLab, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']


# === Inicjalizacja i trening ===
device = torch.device('cpu')
dataset = CASIA2Dataset(
    img_dir='dataset/new_with_masks/train/manipulated',
    mask_dir='dataset/new_with_masks/train/groundtruth',
    dct_dir='dataset/new_with_masks/train/dct'
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = ModifiedDeepLab(in_channels=6).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
optimizer = optim.Adam(model.parameters(), lr=1e-4)

if __name__ == "__main__":
    for epoch in range(40):
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
        torch.save(model.state_dict(), f'deeplabv3_casia2_dct_epoch{epoch+1}_v6.pth')

    torch.save(model.state_dict(), f'deeplabv3_casia2_dct_v6.pth')
