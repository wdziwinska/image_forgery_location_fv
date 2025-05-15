import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# === Dataset class ===
class CASIA2Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, fft_dir=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.fft_dir = fft_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.tif', '.png'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.img_dir, img_filename)

        base_name = os.path.splitext(img_filename)[0]
        mask_filename = base_name + '_gt.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Maska nie znaleziona: {mask_path}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.fft_dir:
            fft_path = os.path.join(self.fft_dir, base_name + '_fft.png')
            if not os.path.exists(fft_path):
                raise FileNotFoundError(f"Brak FFT: {fft_path}")
            fft = Image.open(fft_path).convert('RGB')
            image = image.resize((256, 256))
            fft = fft.resize((256, 256))
            image_np = np.concatenate([np.array(image), np.array(fft)], axis=2)
            image_np = image_np.astype(np.float32) / 255.0
            image = torch.from_numpy(image_np).permute(2, 0, 1)
        else:
            if self.transform:
                image = self.transform(image)

        if self.transform:
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask

# === Model ===
class ModifiedDeepLab(nn.Module):
    def __init__(self, in_channels=6):
        super(ModifiedDeepLab, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=False)
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']

# === Evaluation ===
def evaluate_model(model, dataloader, model_name="deeplabv3_casia2_v2", output_dir="val_output", num_visuals=5):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_targets = []
    count = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to('cpu')
            masks = masks.to('cpu')

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.view(-1).numpy())
            all_targets.append(masks.view(-1).numpy())

            if count < num_visuals:
                for b in range(images.shape[0]):
                    if count >= num_visuals:
                        break

                    img = images[b][:3].permute(1, 2, 0).numpy()
                    gt_mask = masks[b][0].numpy()
                    pred_mask = preds[b][0].numpy()

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(img)
                    axs[0].set_title("Obraz")
                    axs[0].axis('off')

                    axs[1].imshow(gt_mask, cmap='gray')
                    axs[1].set_title("Maska GT")
                    axs[1].axis('off')

                    axs[2].imshow(pred_mask, cmap='gray')
                    axs[2].set_title("Predykcja")
                    axs[2].axis('off')

                    output_path = os.path.join(output_dir, f"{model_name}_{count}.png")
                    plt.savefig(output_path)
                    plt.close()
                    count += 1

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    report = classification_report(y_true, y_pred, target_names=["clean", "tampered"])
    print("=== Raport klasyfikacji (per piksel) ===")
    print(report)

    with open(os.path.join(output_dir, f"{model_name}_report.txt"), "w") as f:
        f.write(report)

# === Main ===
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    model = ModifiedDeepLab(in_channels=6)
    state_dict = torch.load("deeplabv3_casia2_v4.pth", map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    val_dataset = CASIA2Dataset(
        img_dir='dataset/new_with_masks/val/manipulated',
        mask_dir='dataset/new_with_masks/val/groundtruth',
        fft_dir='dataset/new_with_masks/val/fft',
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    evaluate_model(model, val_loader, model_name="deeplabv3_casia2_v4")
