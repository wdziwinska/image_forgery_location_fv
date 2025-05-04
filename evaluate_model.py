import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, classification_report
from PIL import Image
from main import ConvNextUNet, CasiaSegmentationDataset  # ensure these classes are defined

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = "convnext_unet_final_weights_v6.pth"
DATA_PATH = "dataset/new_with_masks/val"
VIZ_DIR  = "eval_outputs"
os.makedirs(VIZ_DIR, exist_ok=True)

# ---------------- Evaluation with Threshold Selection ----------------
def evaluate_and_select_threshold(model_path, data_path):
    # Load model
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        model = ConvNextUNet().to(device)
        model.load_state_dict(loaded)
    else:
        model = loaded.to(device)
    model.eval()

    # Data loader
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    dataset = CasiaSegmentationDataset(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Collect scores and labels
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            logits = model(img)
            scores = torch.sigmoid(logits).cpu().view(-1).numpy()
            labels = mask.view(-1).cpu().numpy().astype(int)
            all_scores.append(scores)
            all_labels.append(labels)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # Precision-Recall and F1
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"Best threshold = {best_thresh:.3f}, F1 = {best_f1:.3f}")

    # Plot and save PR curve
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label="PR Curve")
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f"Best F1={best_f1:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "pr_curve.png"))
    plt.close()

    return best_thresh


def evaluate_with_threshold(model_path, data_path, threshold):
    # Load model
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        model = ConvNextUNet().to(device)
        model.load_state_dict(loaded)
    else:
        model = loaded.to(device)
    model.eval()

    # Data loader
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    dataset = CasiaSegmentationDataset(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    y_true = []
    y_pred = []

    # Evaluate and visualize
    for idx, (img, mask) in enumerate(loader):
        img = img.to(device)
        logits = model(img)
        probs = torch.sigmoid(logits).cpu().view(-1).detach().numpy()
        preds = (probs > threshold).astype(int)
        labels = mask.view(-1).cpu().numpy().astype(int)
        y_true.append(labels)
        y_pred.append(preds)
        # visualize first 10
        if idx < 10:
            visualize_sample(img[0], mask[0], probs.reshape(mask.shape), (probs > threshold).reshape(mask.shape), idx, threshold)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["background","manipulated"], zero_division=0))


def visualize_sample(image_tensor, gt_mask, prob_map, bin_mask, idx, threshold):
    img = image_tensor[:3].permute(1,2,0).cpu().numpy()
    gt = gt_mask.squeeze().cpu().numpy()
    prob_map = prob_map.squeeze()
    bin_mask = bin_mask.squeeze()

    fig, axs = plt.subplots(1,4,figsize=(16,4))
    axs[0].imshow(img); axs[0].set_title("Image"); axs[0].axis('off')
    axs[1].imshow(gt, cmap='gray'); axs[1].set_title("Ground Truth"); axs[1].axis('off')
    axs[2].imshow(prob_map, cmap='viridis'); axs[2].set_title("Probability"); axs[2].axis('off')
    axs[3].imshow(bin_mask, cmap='gray'); axs[3].set_title(f"Binary @ {threshold:.2f}"); axs[3].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"{MODEL_PATH}_2_{idx}.png"))
    plt.close()


if __name__ == "__main__":
    best_thresh = evaluate_and_select_threshold(MODEL_PATH, DATA_PATH)
    evaluate_with_threshold(MODEL_PATH, DATA_PATH, best_thresh)
