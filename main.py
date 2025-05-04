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
        self.manipulated_dir = os.path.join(root_dir, "manipulated")
        self.fft_dir = os.path.join(root_dir, "fft")
        self.gt_dir = os.path.join(root_dir, "groundtruth")
        self.filenames = [f for f in os.listdir(self.manipulated_dir)
                          if os.path.isfile(os.path.join(self.manipulated_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        base = os.path.splitext(fname)[0]
        img = Image.open(os.path.join(self.manipulated_dir, fname)).convert("RGB")
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
        chs = [f['num_chs'] for f in feats]  # [s1,s2,s3,s4]
        # decoder blocks
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
        self.final = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        d3 = self.up3(s4); d3 = torch.cat([d3, s3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, s2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, s1], dim=1); d1 = self.dec1(d1)
        out = self.final(F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False))
        return out  # logits

# ---------------- Loss functions ----------------
def dice_loss(logits, target, smooth=1e-6):
    pred = torch.sigmoid(logits).view(-1)
    tgt = target.view(-1)
    inter = (pred * tgt).sum()
    return 1 - ((2 * inter + smooth) / (pred.sum() + tgt.sum() + smooth))

# ---------------- Training ----------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # augmentations
    transform = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5),
        T.RandomRotation(15),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.ToTensor()
    ])
    train_ds = CasiaSegmentationDataset("dataset/new_with_masks/train", transform)
    val_ds   = CasiaSegmentationDataset("dataset/new_with_masks/val",   transform)
    # sampler for oversampling
    weights = [2.0 if (CasiaSegmentationDataset.__getitem__(train_ds, i)[1].sum()>0) else 1.0
               for i in range(len(train_ds))]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = ConvNextUNet().to(device)
    # compute pos_weight
    total=pos=0
    for _,m in train_loader:
        total+=m.numel(); pos+=(m>0).sum().item()
    neg = total-pos
    pos_w = torch.tensor([neg/pos], device=device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # scheduler: reduce LR when val_loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=2, min_lr=1e-6)
    num_epochs = 5

    mlflow.set_experiment("convnext_unet_weights_v5")
    with mlflow.start_run():
        mlflow.log_params({"model":"convnext_unet","epochs":num_epochs,
                           "batch_size":8,"pos_weight":pos_w.item()})
        mlflow.log_params({"optimizer":"Adam","lr_init":1e-4,
                           "scheduler":"ReduceLROnPlateau"})

        for epoch in range(num_epochs):
            model.train(); train_loss=0
            for x,y in tqdm(train_loader, desc=f"Train e{epoch+1}"):
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss_bce   = bce_fn(logits,y)
                loss_focal = sigmoid_focal_loss(logits,y,alpha=0.25,gamma=2.0,reduction='mean')
                loss_dice  = dice_loss(logits,y)
                loss = loss_bce+loss_focal+loss_dice
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_loss+=loss.item()
            avg_tr = train_loss/len(train_loader)
            mlflow.log_metric("train_loss", avg_tr, step=epoch)

            model.eval(); val_loss=0
            with torch.no_grad():
                for x,y in tqdm(val_loader, desc=f"Val   e{epoch+1}"):
                    x,y = x.to(device), y.to(device)
                    logits = model(x)
                    l = (bce_fn(logits,y)
                         + sigmoid_focal_loss(logits,y,alpha=0.25,gamma=2.0,reduction='mean')
                         + dice_loss(logits,y))
                    val_loss+=l.item()
            avg_val = val_loss/len(val_loader)
            mlflow.log_metric("val_loss", avg_val, step=epoch)
            # step scheduler on validation loss
            scheduler.step(avg_val)

            print(f"Epoch {epoch+1}/{num_epochs} -> train_loss: {avg_tr:.4f}, val_loss: {avg_val:.4f}, lr: {optimizer.param_groups[0]['lr']:.1e}")

        # log & save model
        mlflow.pytorch.log_model(model, artifact_path="model")
        torch.save(model.state_dict(), "convnext_unet_weights_v5.pth")
        mlflow.log_artifact("convnext_unet_weights_v5.pth")
        print("Local model weights saved.")

if __name__ == "__main__":
    train_model()
