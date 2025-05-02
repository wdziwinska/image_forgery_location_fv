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
        manipulated_file = self.filenames[idx]
        base_name = os.path.splitext(manipulated_file)[0]

        img_path = os.path.join(self.manipulated_dir, manipulated_file)
        fft_path = os.path.join(self.fft_dir, base_name + "_fft.png")
        mask_path = os.path.join(self.gt_dir, base_name + "_gt.png")

        image = Image.open(img_path).convert("RGB")
        fft = Image.open(fft_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            fft = self.transform(fft)  # fft: [1, 224, 224]
            mask = T.Resize(image.shape[1:])(T.ToTensor()(mask))

        # fft: [1, H, W]
        input_tensor = torch.cat([image, fft], dim=0)  # [4, H, W]

        mask = (mask > 0.5).float()

        return input_tensor, mask

# ---------------- Model ----------------
class ConvNextSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("convnext_tiny", pretrained=True, in_chans=4, features_only=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder.feature_info[-1]['num_chs'], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)[-1]
        out = self.decoder(features)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# ---------------- Training with MLflow ----------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    train_dataset = CasiaSegmentationDataset("dataset/new_with_masks/train", transform=transform)
    val_dataset = CasiaSegmentationDataset("dataset/new_with_masks/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = ConvNextSegmenter().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5

    mlflow.set_experiment("convnext_casia_segmentation")

    with mlflow.start_run():
        mlflow.log_param("model", "convnext_tiny")
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("input_channels", 4)
        mlflow.log_param("loss_function", "BCELoss")
        mlflow.log_param("optimizer", "Adam")

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                inputs, masks = inputs.to(device), masks.to(device)
                preds = model(inputs)
                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            print(f"[{epoch+1}] Train loss: {avg_train_loss:.4f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    inputs, masks = inputs.to(device), masks.to(device)
                    preds = model(inputs)
                    loss = criterion(preds, masks)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            print(f"[{epoch+1}] Val loss: {avg_val_loss:.4f}")

        # Zapis modelu do MLflow
        mlflow.pytorch.log_model(model, artifact_path="model")

if __name__ == "__main__":
    train_model()
