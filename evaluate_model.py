import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, classification_report
from main import ConvNextUNet, CasiaPatchSegDataset  # patch-based dataset and UNet model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = "convnext_patch_segmentation_base.pth"
DATA_PATH  = "dataset/new_with_masks/val"
VIZ_DIR    = "eval_outputs"
os.makedirs(VIZ_DIR, exist_ok=True)

# ---------------- Evaluation with Threshold Selection ----------------
def evaluate_and_select_threshold(model_path, data_path, patch_size=128, patches_per_image=5):
    state = torch.load(model_path, map_location=device, weights_only=False)
    model = ConvNextUNet().to(device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state.to(device)
    model.eval()

    transform = T.Compose([T.Resize((patch_size, patch_size)), T.ToTensor()])
    dataset = CasiaPatchSegDataset(data_path,
                                   patch_size=patch_size,
                                   patches_per_image=patches_per_image,
                                   transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            logits = model(img)
            probs  = torch.sigmoid(logits)
            all_scores.append(probs.view(-1).cpu().numpy())
            all_labels.append(mask.view(-1).cpu().numpy().astype(int))
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx   = np.nanargmax(f1_scores)
    best_thresh= thresholds[best_idx]
    best_f1    = f1_scores[best_idx]
    print(f"Best threshold = {best_thresh:.3f}, F1 = {best_f1:.3f}")

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

# ---------------- Evaluate with chosen threshold ----------------
def evaluate_with_threshold(model_path, data_path, threshold, patch_size=128, patches_per_image=5):
    state = torch.load(model_path, map_location=device, weights_only=False)
    model = ConvNextUNet().to(device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state.to(device)
    model.eval()

    transform = T.Compose([T.Resize((patch_size, patch_size)), T.ToTensor()])
    dataset = CasiaPatchSegDataset(data_path,
                                   patch_size=patch_size,
                                   patches_per_image=patches_per_image,
                                   transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    y_true, y_pred = [], []
    for idx, (img, mask) in enumerate(loader):
        img = img.to(device)
        logits = model(img)
        probs  = torch.sigmoid(logits)
        preds  = (probs > threshold).int()

        y_true.append(mask.view(-1).cpu().numpy().astype(int))
        y_pred.append(preds.view(-1).cpu().numpy().astype(int))

        if idx < 10:
            visualize_sample(img[0], mask[0], probs[0], preds[0], idx, threshold)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred,
                                target_names=["background","manipulated"],
                                zero_division=0))

# ---------------- Visualization ----------------
def visualize_sample(img_t, gt_t, prob_t, bin_t, idx, threshold):
    img = img_t[:3].permute(1,2,0).cpu().numpy()
    gt  = gt_t.squeeze().cpu().numpy()
    prob= prob_t.detach().cpu().numpy().squeeze()
    binm= bin_t.detach().cpu().numpy().squeeze()

    fig, axs = plt.subplots(1,4,figsize=(16,4))
    axs[0].imshow(img); axs[0].set_title("Image"); axs[0].axis('off')
    axs[1].imshow(gt, cmap='gray'); axs[1].set_title("GT"); axs[1].axis('off')
    axs[2].imshow(prob, cmap='viridis'); axs[2].set_title("Prob"); axs[2].axis('off')
    axs[3].imshow(binm, cmap='gray'); axs[3].set_title(f"Bin @ {threshold:.2f}"); axs[3].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"{MODEL_PATH}_{idx}.png"))
    plt.close()

if __name__ == "__main__":
    best_thresh = evaluate_and_select_threshold(MODEL_PATH, DATA_PATH)
    evaluate_with_threshold(MODEL_PATH, DATA_PATH, best_thresh)
