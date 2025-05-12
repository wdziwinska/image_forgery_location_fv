import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image

# === Dataset z DCT ===
class ManipulationDataset(Dataset):
    def __init__(self, img_dir, dct_dir, mask_dir, size=(256, 256)):
        self.img_dir = img_dir
        self.dct_dir = dct_dir
        self.mask_dir = mask_dir
        self.size = size
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.tif')]

        self.img_transform = T.Compose([
            T.Resize(size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(size, interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{id_}.tif")
        dct_path = os.path.join(self.dct_dir, f"{id_}.npy")
        mask_path = os.path.join(self.mask_dir, f"{id_}.tif")

        # Wczytanie
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        dct = np.load(dct_path)
        dct_tensor = torch.tensor(dct).unsqueeze(0).float()  # [1, H, W]

        # Transformacje
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        # Dopasowanie rozmiaru DCT
        if dct_tensor.shape[-2:] != img.shape[-2:]:
            dct_tensor = torch.nn.functional.interpolate(dct_tensor.unsqueeze(0), size=img.shape[-2:], mode='bilinear', align_corners=False)[0]

        img_combined = torch.cat([img, dct_tensor], dim=0)  # [4, H, W]
        return img_combined, mask, id_

# === Ścieżki ===
val_img = "split/val/Datasets/defacto-inpainting/inpainting_img/img"
val_dct = "split/val/Datasets/defacto-inpainting/inpainting_img/dct_spectrum"
val_mask = "split/val/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask"
model_path = "trained_models/manipulation_detector_v4.pt"

# === Dataset + loader
val_ds = ManipulationDataset(val_img, val_dct, val_mask)
val_loader = DataLoader(val_ds, batch_size=1)

# === Model z 4-kanałowym wejściem
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(pretrained=True)

# Zmiana wejścia na 4 kanały
new_conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    new_conv1.weight[:, :3] = model.backbone.conv1.weight  # kopiuj RGB
    new_conv1.weight[:, 3] = model.backbone.conv1.weight[:, 0] * 0.0  # zainicjalizuj DCT
model.backbone.conv1 = new_conv1

# Binary output
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)

# Wczytaj wagę modelu (trzeba mieć identyczną architekturę!)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === Ewaluacja
all_preds = []
all_labels = []
os.makedirs("val_predictions", exist_ok=True)

with torch.no_grad():
    for i, (img, mask, id_) in enumerate(val_loader):
        img, mask = img.to(device), mask.to(device)
        out = torch.sigmoid(model(img)['out'])
        pred = (out > 0.5).float()

        all_preds.extend(pred.cpu().numpy().ravel())
        all_labels.extend(mask.cpu().numpy().ravel())

        if i < 10:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(to_pil_image(img[0, :3].cpu()))  # tylko RGB
            axs[0].set_title("Obraz RGB")
            axs[1].imshow(mask[0].cpu().squeeze(), cmap='gray')
            axs[1].set_title("Maska GT")
            axs[2].imshow(pred[0].cpu().squeeze(), cmap='gray')
            axs[2].set_title("Predykcja")
            for ax in axs: ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"val_predictions/{id_[0]}_manipulation_detector_v4.png")
            plt.close()

# === Raport
all_preds_bin = np.array(all_preds).astype(int)
all_labels_bin = np.array(all_labels).astype(int)
print(classification_report(all_labels_bin, all_preds_bin, target_names=["Tło", "Manipulacja"]))
