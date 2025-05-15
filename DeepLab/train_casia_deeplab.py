import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from torchvision.transforms.functional import resize as tf_resize


# Dataset
class CASIA2Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, fft_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.fft_dir = fft_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    from torchvision.transforms.functional import resize as tf_resize

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        filename = os.path.splitext(self.images[idx])[0]
        mask_path = os.path.join(self.mask_dir, filename + '_gt.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.fft_dir:
            fft_path = os.path.join(self.fft_dir, filename + '_fft.png')
            fft = Image.open(fft_path).convert('RGB')

            # Resize both before stacking
            image = image.resize((256, 256))
            fft = fft.resize((256, 256))

            image_np = np.concatenate([
                np.array(image), np.array(fft)
            ], axis=2)  # (H, W, 6)
            image_np = image_np.astype(np.float32) / 255.0
            image = torch.from_numpy(image_np).permute(2, 0, 1)  # (C, H, W)
        else:
            if self.transform:
                image = self.transform(image)

        if self.transform:
            mask = self.transform(mask)

        mask = (mask > 0).float()

        return image, mask


# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = CASIA2Dataset('dataset/new_with_masks/train/manipulated', 'dataset/new_with_masks/train/groundtruth', fft_dir='dataset/new_with_masks/train/fft', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Model
class ModifiedDeepLab(nn.Module):
    def __init__(self, in_channels=6):
        super(ModifiedDeepLab, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # binary output

    def forward(self, x):
        return self.model(x)['out']


# Initialize model, loss, optimizer
device = torch.device('cpu')
model = ModifiedDeepLab(in_channels=6).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)  # shape: [B, 1, H, W]
        loss = criterion(outputs, masks)  # masks też ma shape [B, 1, H, W]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
    # Save model
    torch.save(model.state_dict(), 'deeplabv3_casia2_v4.pth')
