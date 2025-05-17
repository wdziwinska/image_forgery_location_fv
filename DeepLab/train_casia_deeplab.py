import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from scipy.fftpack import dct
from torchvision.transforms.functional import resize as tf_resize


# === Dataset z DCT ===
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

    def compute_dct(self, image_np):
        # image_np: (H, W, 3), dtype=uint8
        dct_channels = []
        for i in range(3):  # R, G, B
            channel = image_np[:, :, i].astype(float)
            dct_channel = dct(dct(channel.T, norm='ortho').T, norm='ortho')
            dct_channels.append(dct_channel)
        dct_img = np.stack(dct_channels, axis=2)
        return dct_img

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        base_name = os.path.splitext(img_filename)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '_gt.png')

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maska nie znaleziona: {mask_path}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = image.resize((256, 256))
        mask = mask.resize((256, 256))
        image_np = np.array(image)

        dct_np = self.compute_dct(image_np)

        combined = np.concatenate([image_np, dct_np], axis=2)  # (H, W, 6)
        combined = combined.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(combined).permute(2, 0, 1)  # (C, H, W)
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        mask_tensor = (mask_tensor > 0).float()

        return image_tensor, mask_tensor


# === Transform (dla masek RGB-DCT nie trzeba tu) ===
transform = None

# === DataLoader ===
dataset = CASIA2Dataset(
    img_dir='dataset/new_with_masks/train/manipulated',
    mask_dir='dataset/new_with_masks/train/groundtruth',
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# === Model ===
class ModifiedDeepLab(nn.Module):
    def __init__(self, in_channels=6):
        super(ModifiedDeepLab, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

# === Trening ===
device = torch.device('cpu')
model = ModifiedDeepLab(in_channels=6).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
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
    torch.save(model.state_dict(), f'deeplabv3_casia2_dct_epoch{epoch+1}_v5.pth')

torch.save(model.state_dict(), f'deeplabv3_casia2_dct_v5.pth')
