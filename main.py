import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from torchvision.ops import sigmoid_focal_loss

# ---------------- Dataset ----------------
class CasiaSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.manipulated_dir = os.path.join(root_dir, "manipulated")
        self.fft_dir = os.path.join(root_dir, "fft")
        self.gt_dir = os.path.join(root_dir, "groundtruth")

        self.filenames = [f for f in os.listdir(self.manipulated_dir)
                          if os.path.isfile(os.path.join(self.manipulated_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        f = self.filenames[idx]
        base = os.path.splitext(f)[0]
        img = Image.open(os.path.join(self.manipulated_dir, f)).convert("RGB")
        fft = Image.open(os.path.join(self.fft_dir, base + "_fft.png")).convert("L")
        mask = Image.open(os.path.join(self.gt_dir, base + "_gt.png")).convert("L")

        if self.transform:
            img = self.transform(img)
            fft = self.transform(fft)
            mask = T.Resize(img.shape[1:])(T.ToTensor()(mask))

        x = torch.cat([img, fft], dim=0)
        y = (mask > 0.5).float()
        return x, y

# ---------------- Model (UNet-like) ----------------
class ConvNextUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("convnext_tiny", pretrained=True, in_chans=4, features_only=True)
        feats = self.encoder.feature_info
        self.chs = [f['num_chs'] for f in feats]
        # Upsample layers
        self.up3 = nn.ConvTranspose2d(self.chs[3], self.chs[2], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.chs[2]*2, self.chs[2], 3, padding=1), nn.ReLU(),
            nn.Conv2d(self.chs[2], self.chs[2], 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(self.chs[2], self.chs[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.chs[1]*2, self.chs[1], 3, padding=1), nn.ReLU(),
            nn.Conv2d(self.chs[1], self.chs[1], 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(self.chs[1], self.chs[0], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.chs[0]*2, self.chs[0], 3, padding=1), nn.ReLU(),
            nn.Conv2d(self.chs[0], self.chs[0], 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(self.chs[0], 1, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        d3 = self.up3(s4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False))
        return out

# ---------------- Loss functions ----------------
def dice_loss(logits, target, smooth=1e-6):
    pred = torch.sigmoid(logits)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

# ---------------- Training ----------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data augmentations: spatial + color
    transform = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor()
    ])

    train_ds = CasiaSegmentationDataset("dataset/new_with_masks/train", transform)
    val_ds   = CasiaSegmentationDataset("dataset/new_with_masks/val",   transform)

    # Oversampling manipulated samples
    # Compute sample weights: higher for 'manipulated' (mask contains positives)
    sample_weights = []
    # Estimate pos presence per sample
    for x, y in train_ds:
        # y: [1,H,W]
        if y.sum() > 0:
            sample_weights.append(2.0)  # weight for manipulated
        else:
            sample_weights.append(1.0)  # weight for background
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    model = ConvNextUNet().to(device)
    # pos_weight for BCE
    total, pos = 0, 0
    for _, m in train_loader:
        total += m.numel()
        pos += (m>0).sum().item()
    neg = total - pos
    pos_weight = torch.tensor([neg/pos]).to(device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10

    mlflow.set_experiment("convnext_unet_weights_v4")
    with mlflow.start_run():
        mlflow.log_params({"model": "convnext_unet", "epochs": num_epochs,
                           "batch_size": 8, "pos_weight": pos_weight.item()})

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for x, y in tqdm(train_loader, desc=f"Train e{epoch+1}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                # Combined loss
                loss_bce = bce_fn(logits, y)
                loss_focal = sigmoid_focal_loss(logits, y, alpha=0.25, gamma=2.0, reduction='mean')
                loss_dice = dice_loss(logits, y)
                loss = loss_bce + loss_focal + loss_dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            mlflow.log_metric("train_loss", train_loss/len(train_loader), step=epoch)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Val   e{epoch+1}"):
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    # Combined loss with mean reduction
                    loss_bce = bce_fn(logits, y)
                    loss_focal = sigmoid_focal_loss(logits, y, alpha=0.25, gamma=2.0, reduction='mean')
                    loss_dice = dice_loss(logits, y)
                    batch_loss = (loss_bce + loss_focal + loss_dice).item()
                    val_loss += batch_loss
            mlflow.log_metric("val_loss", val_loss/len(val_loader), step=epoch)

            print(f"Epoch {epoch+1}/{num_epochs} -> train_loss: {train_loss/len(train_loader):.4f}, \
                  val_loss: {val_loss/len(val_loader):.4f}")

        # Save and log model
        mlflow.pytorch.log_model(model, artifact_path="model")
        torch.save(model.state_dict(), "convnext_unet_weights_v4.pth")
        mlflow.log_artifact("convnext_unet_weights.pth")
        print("Local model weights saved.")

if __name__ == "__main__":
    train_model()
