import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score
from main import ConvNextSegmenter, CasiaSegmentationDataset  # załóżmy, że masz ten kod w model.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Metryki ----------------
def compute_metrics(preds, targets):
    preds = preds > 0.5
    targets = targets > 0.5

    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()

    iou = jaccard_score(targets, preds, zero_division=0)
    dice = f1_score(targets, preds, zero_division=0)
    acc = (preds == targets).mean()

    return iou, dice, acc

# ---------------- Wizualizacja ----------------
def visualize(image_tensor, gt_mask, pred_mask, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    image = image_tensor[:3].permute(1, 2, 0).cpu().numpy()
    gt = gt_mask.squeeze().cpu().numpy()
    pred = pred_mask.squeeze().cpu().detach().numpy()

    axs[0].imshow(image)
    axs[0].set_title("Manipulated image")

    axs[1].imshow(gt, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Prediction")

    for ax in axs:
        ax.axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()

# ---------------- Ewaluacja ----------------
def evaluate_model(model_path, data_path, save_viz=True):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    dataset = CasiaSegmentationDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    loaded = torch.load(model_path, map_location=device, weights_only=False)  # ← KLUCZOWE!

    if isinstance(loaded, dict):
        # Zapisano tylko wagi (state_dict)
        model = ConvNextSegmenter().to(device)
        model.load_state_dict(loaded)
    else:
        # Zapisano cały model — użyj go bez load_state_dict
        model = loaded.to(device)

    model.eval()

    ious, dices, accs = [], [], []

    os.makedirs("eval_outputs", exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, masks) in enumerate(dataloader):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)

            iou, dice, acc = compute_metrics(outputs, masks)
            ious.append(iou)
            dices.append(dice)
            accs.append(acc)

            if save_viz and idx < 20:  # tylko pierwsze 20
                visualize(inputs[0], masks[0], outputs[0], save_path=f"eval_outputs/viz_{idx}.png")

    print(f"IoU avg:  {sum(ious)/len(ious):.4f}")
    print(f"Dice avg: {sum(dices)/len(dices):.4f}")
    print(f"Acc avg:  {sum(accs)/len(accs):.4f}")

if __name__ == "__main__":
    evaluate_model(
        model_path="mlruns/323918849388816282/e992d302328640f99b69577fb3662561/artifacts/model/data/model.pth",  # ← podmień na właściwą ścieżkę
        data_path="dataset/new_with_masks/val"
    )
