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

# ---------------- Combined Patch-based Dataset ----------------
class CombinedPatchSegDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, patches_per_image=5, transform=None):
        self.manip_dir = os.path.join(root_dir, "manipulated")
        self.fft_dir = os.path.join(root_dir, "fft")
        self.gt_dir = os.path.join(root_dir, "groundtruth")
        self.orig_dir = os.path.join(root_dir, "original")
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.manip_files = [f for f in os.listdir(self.manip_dir)
                            if os.path.splitext(f)[1].lower() in exts]
        self.orig_files = [f for f in os.listdir(self.orig_dir)
                           if os.path.splitext(f)[1].lower() in exts]
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform
        self.index_list = [(f, True) for f in self.manip_files] + [(f, False) for f in self.orig_files]

    def __len__(self):
        return len(self.index_list) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        fname, is_manip = self.index_list[img_idx]
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(self.manip_dir if is_manip else self.orig_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if is_manip:
            fft = Image.open(os.path.join(self.fft_dir, base + "_fft.png")).convert("L")
            mask = Image.open(os.path.join(self.gt_dir, base + "_gt.png")).convert("L")
        else:
            fft = img.convert("L")
            mask = Image.new("L", img.size, 0)
        w, h = img.size
        ps = self.patch_size
        x0 = random.randint(0, w - ps)
        y0 = random.randint(0, h - ps)
        img_p = img.crop((x0, y0, x0+ps, y0+ps))
        fft_p = fft.crop((x0, y0, x0+ps, y0+ps))
        mask_p = mask.crop((x0, y0, x0+ps, y0+ps))
        if self.transform:
            img_p = self.transform(img_p)
            fft_p = self.transform(fft_p)
            mask_p = T.Resize((ps,ps))(T.ToTensor()(mask_p))
        fft_p = fft_p[0:1]
        x = torch.cat([img_p, fft_p], dim=0)
        y = (mask_p > 0.5).float()
        return x, y

# ---------------- UNet-like Model ----------------
class ConvNextUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("convnext_base", pretrained=True, in_chans=4, features_only=True)
        feats = self.encoder.feature_info
        chs = [f['num_chs'] for f in feats]
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(chs[2]*2, chs[2], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[2], chs[2], 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(chs[1]*2, chs[1], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[1], chs[1], 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(chs[0]*2, chs[0], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[0], chs[0], 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        d3 = self.up3(s4); d3 = torch.cat([d3, s3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, s2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, s1], dim=1); d1 = self.dec1(d1)
        return self.final(F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False))

# ---------------- Training ----------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.Resize((128,128)), T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomHorizontalFlip(), T.RandomRotation(10), T.ToTensor()])
    ds_train = CombinedPatchSegDataset("dataset/new_with_masks/train", patch_size=128,
                                       patches_per_image=10, transform=transform)
    ds_val   = CombinedPatchSegDataset("dataset/new_with_masks/val",   patch_size=128,
                                       patches_per_image=5,  transform=transform)
    weights = [2.0 if ds_train[i][1].sum()>0 else 1.0 for i in range(len(ds_train))]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader_train = DataLoader(ds_train, batch_size=8, sampler=sampler)
    loader_val   = DataLoader(ds_val,   batch_size=8, shuffle=False)

    model = ConvNextUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    epochs = 5

    best_val_loss = float('inf')
    mlflow.set_experiment("convnext_patch_combined_v10")
    with mlflow.start_run():
        mlflow.log_params({"backbone": "convnext_base", "patch_size":128,
                           "patches":10, "batch":8, "epochs":epochs, "lr":1e-5})
        for e in range(epochs):
            model.train(); total_train_loss = 0
            for x, y in tqdm(loader_train, desc=f"Train e{e+1}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y) + \
                       sigmoid_focal_loss(logits, y, alpha=0.25, gamma=2.0, reduction='mean')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(loader_train)
            mlflow.log_metric("train_loss", avg_train_loss, step=e)

            model.eval(); total_val_loss = 0
            with torch.no_grad():
                for x, y in tqdm(loader_val, desc=f"Val   e{e+1}"):
                    x, y = x.to(device), y.to(device)
                    val_loss = criterion(model(x), y) + \
                               sigmoid_focal_loss(model(x), y, alpha=0.25, gamma=2.0, reduction='mean')
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(loader_val)
            mlflow.log_metric("val_loss", avg_val_loss, step=e)
            print(f"Epoch {e+1}/{epochs} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = f"best_epoch_{e+1}_loss_{best_val_loss:.4f}.pth"
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(best_path)
                print(f"Saved new best model: {best_path}")

            scheduler.step(avg_val_loss)

        # Final artifact
        mlflow.pytorch.log_model(model, artifact_path="model")
        torch.save(model.state_dict(), "convnext_v10.pth")
        mlflow.log_artifact("convnext_v10.pth")

if __name__ == "__main__":
    train_model()
