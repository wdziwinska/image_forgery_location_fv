import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from torchvision.ops import sigmoid_focal_loss

# ---------------- Patch-based Segmentation Dataset ----------------
class CasiaPatchSegDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, patches_per_image=5, transform=None):
        self.manipulated_dir = os.path.join(root_dir, "manipulated")
        self.fft_dir = os.path.join(root_dir, "fft")
        self.gt_dir = os.path.join(root_dir, "groundtruth")
        self.filenames = [f for f in os.listdir(self.manipulated_dir)
                          if os.path.isfile(os.path.join(self.manipulated_dir, f))]
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform

    def __len__(self):
        return len(self.filenames) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        fname = self.filenames[img_idx]
        base = os.path.splitext(fname)[0]

        img = Image.open(os.path.join(self.manipulated_dir, fname)).convert("RGB")
        fft = Image.open(os.path.join(self.fft_dir, base + "_fft.png")).convert("L")
        mask = Image.open(os.path.join(self.gt_dir, base + "_gt.png")).convert("L")

        w, h = img.size
        ps = self.patch_size
        x0 = random.randint(0, w - ps)
        y0 = random.randint(0, h - ps)
        img_patch = img.crop((x0, y0, x0+ps, y0+ps))
        fft_patch = fft.crop((x0, y0, x0+ps, y0+ps))
        mask_patch = mask.crop((x0, y0, x0+ps, y0+ps))

        if self.transform:
            img_patch = self.transform(img_patch)
            fft_patch = self.transform(fft_patch)
            mask_patch = T.Resize((ps, ps))(T.ToTensor()(mask_patch))

        fft_patch = fft_patch[0:1]
        x = torch.cat([img_patch, fft_patch], dim=0)
        y = (mask_patch > 0.5).float()
        return x, y

# ---------------- UNet-like segmentation model ----------------
class ConvNextUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use convnext_base backbone for richer feature extraction
        self.encoder = timm.create_model("convnext_base", pretrained=True, in_chans=4, features_only=True)
        feats = self.encoder.feature_info
        chs = [f['num_chs'] for f in feats]  # [s1, s2, s3, s4]
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(chs[2]*2, chs[2], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[2], chs[2], 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(chs[1]*2, chs[1], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[1], chs[1], 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(chs[0]*2, chs[0], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[0], chs[0], 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        d3 = self.up3(s4); d3 = torch.cat([d3, s3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, s2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, s1], dim=1); d1 = self.dec1(d1)
        out = self.final(F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False))
        return out

# ---------------- Training ----------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.Resize((128,128)),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor()
    ])
    train_ds = CasiaPatchSegDataset("dataset/new_with_masks/train", patch_size=128, patches_per_image=10, transform=transform)
    val_ds   = CasiaPatchSegDataset("dataset/new_with_masks/val",   patch_size=128, patches_per_image=5,  transform=transform)

    weights = [2.0 if train_ds[i][1].sum()>0 else 1.0 for i in range(len(train_ds))]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = ConvNextUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)
    num_epochs = 20

    mlflow.set_experiment("convnext_patch_segmentation_base")
    with mlflow.start_run():
        mlflow.log_params({
            "backbone": "convnext_base",
            "patch_size": 128,
            "patches_per_image": 10,
            "batch_size": 8,
            "epochs": num_epochs,
            "lr_init": 1e-5
        })
        for epoch in range(num_epochs):
            model.train(); train_loss=0
            for x, y in tqdm(train_loader, desc=f"Train e{epoch+1}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y) + sigmoid_focal_loss(logits, y, alpha=0.25, gamma=2.0, reduction='mean')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_loss += loss.item()
            avg_tr = train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_tr, step=epoch)

            model.eval(); val_loss=0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Val   e{epoch+1}"):
                    x, y = x.to(device), y.to(device)
                    val_loss += criterion(model(x), y).item()
            avg_v = val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_v, step=epoch)
            scheduler.step(avg_v)
            print(f"Epoch {epoch+1}/{num_epochs} tr={avg_tr:.4f} val={avg_v:.4f} lr={optimizer.param_groups[0]['lr']:.1e}")

        mlflow.pytorch.log_model(model, artifact_path="model")
        torch.save(model.state_dict(), "convnext_patch_segmentation_base.pth")
        mlflow.log_artifact("convnext_patch_segmentation_base.pth")

if __name__ == "__main__":
    train_model()
