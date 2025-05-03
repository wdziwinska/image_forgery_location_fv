import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import timm
from tqdm import tqdm
import mlflow
import mlflow.pytorch

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

        # concat RGB + FFT
        x = torch.cat([img, fft], dim=0)
        y = (mask > 0.5).float()
        return x, y

# ---------------- Model ----------------
class ConvNextSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("convnext_tiny", pretrained=True, in_chans=4, features_only=True)
        c = self.encoder.feature_info[-1]['num_chs']
        self.decoder = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.encoder(x)[-1]
        out = self.decoder(f)
        return nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

# ---------------- Training ----------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])

    train_ds = CasiaSegmentationDataset("dataset/new_with_masks/train", transform)
    val_ds   = CasiaSegmentationDataset("dataset/new_with_masks/val",   transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    model = ConvNextSegmenter().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    save_local = "convnext_segmentation_v2.pth"

    mlflow.set_experiment("convnext_casia_segmentation")
    with mlflow.start_run():
        mlflow.log_params({"model": "convnext_tiny", "epochs": num_epochs, "batch_size": 8})

        for e in range(num_epochs):
            model.train()
            train_loss = 0
            for x,y in tqdm(train_loader, desc=f"Train e{e+1}"):
                x,y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_loss += loss.item()
            avg_tr = train_loss/len(train_loader)
            mlflow.log_metric("train_loss", avg_tr, step=e)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x,y in tqdm(val_loader, desc=f"Val   e{e+1}"):
                    x,y = x.to(device), y.to(device)
                    loss = criterion(model(x), y)
                    val_loss += loss.item()
            avg_val = val_loss/len(val_loader)
            mlflow.log_metric("val_loss", avg_val, step=e)

            print(f"Epoch {e+1}/{num_epochs} -> train_loss: {avg_tr:.4f}, val_loss: {avg_val:.4f}")

        # log model to MLflow
        mlflow.pytorch.log_model(model, artifact_path="model")

        # save local weights
        torch.save(model.state_dict(), save_local)
        mlflow.log_artifact(save_local)
        print(f"Local model weights saved to {save_local}")

if __name__ == "__main__":
    train_model()
