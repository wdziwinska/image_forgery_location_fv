import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from train_casia_deeplab import CASIA2Dataset, ModifiedDeepLab  # <- zakładamy że są w osobnym pliku

# === Ścieżki
models_dir = "./"  # folder z plikami .pth
val_img_dir = "dataset/new_with_masks/val/manipulated"
val_mask_dir = "dataset/new_with_masks/val/groundtruth"
val_dct_dir = "dataset/new_with_masks/val/dct"

output_path = "raport_zbiorczy.txt"

# === Dane walidacyjne
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_dataset = CASIA2Dataset(
    img_dir=val_img_dir,
    mask_dir=val_mask_dir,
    dct_dir=val_dct_dir,
    transform=transform,
    augment=False
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# === Ewaluacja modelu
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Ewaluacja"):
            images, masks = images.to('cpu'), masks.to('cpu')
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.append(preds.view(-1).numpy())
            all_targets.append(masks.view(-1).numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return classification_report(y_true, y_pred, target_names=["clean", "tampered"], digits=4)

# === Główna pętla po modelach
with open(output_path, 'w') as out_file:
    for fname in sorted(os.listdir(models_dir)):
        if fname.endswith('.pth'):
            model_path = os.path.join(models_dir, fname)
            print(f"Sprawdzam model: {fname}")

            # Wczytaj model
            model = ModifiedDeepLab(in_channels=6)
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

            # Ewaluacja
            report = evaluate_model(model, val_loader)

            # Zapisz do pliku
            out_file.write(f"{'=' * 40}\nMODEL: {fname}\n{'=' * 40}\n")
            out_file.write(report + "\n\n")
            print(f"Zapisano raport dla modelu: {fname}")
