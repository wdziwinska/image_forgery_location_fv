import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from tqdm import tqdm

def compute_dct_image(image_np):
    """Zamienia obraz RGB na macierz DCT (R, G, B osobno)"""
    dct_channels = []
    for i in range(3):  # R, G, B
        channel = image_np[:, :, i].astype(float)
        dct_channel = dct(dct(channel.T, norm='ortho').T, norm='ortho')
        dct_channels.append(dct_channel)
    return np.stack(dct_channels, axis=0)  # shape: (3, H, W)

def generate_dct_for_split(root_dir, split_name, target_size=(256, 256)):
    manipulated_dir = os.path.join(root_dir, split_name, "manipulated")
    dct_dir = os.path.join(root_dir, split_name, "dct")
    os.makedirs(dct_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(manipulated_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.tif'))
    ]

    print(f"Przetwarzanie splitu: {split_name} ({len(image_files)} plików)")
    for fname in tqdm(image_files):
        img_path = os.path.join(manipulated_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)

        img_np = np.array(img)  # shape: (H, W, 3)
        dct_np = compute_dct_image(img_np)  # shape: (3, H, W)

        out_path = os.path.join(dct_dir, os.path.splitext(fname)[0] + '_dct.npy')
        np.save(out_path, dct_np.astype(np.float32))

if __name__ == "__main__":
    base_path = "DeepLab/dataset/new_with_masks"
    for split in ["train", "val", "test"]:
        generate_dct_for_split(base_path, split)
