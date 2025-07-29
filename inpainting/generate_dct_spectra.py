import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from tqdm import tqdm

splits = ["train", "val", "test"]
base_dir = "split"
img_subpath = "Datasets/defacto-inpainting/inpainting_img/img"
output_subpath = "Datasets/defacto-inpainting/inpainting_img/dct_spectrum"

# === Funkcja: oblicz DCT z kanału luminancji ===
def compute_dct_spectrum(img_pil):
    gray = np.array(img_pil.convert("L")).astype(np.float32)
    coeff = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    coeff = np.abs(coeff)
    coeff = np.log1p(coeff)  # log-amplituda
    coeff = (coeff - coeff.min()) / (coeff.max() - coeff.min())  # normalizacja [0,1]
    return coeff

for split in splits:
    print(f"Przetwarzanie splitu: {split}")
    img_dir = os.path.join(base_dir, split, img_subpath)
    out_dir = os.path.join(base_dir, split, output_subpath)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".tif"))]

    for fname in tqdm(files):
        img_path = os.path.join(img_dir, fname)
        img = Image.open(img_path)
        dct_map = compute_dct_spectrum(img)

        id_ = os.path.splitext(fname)[0]
        np.save(os.path.join(out_dir, f"{id_}.npy"), dct_map)

        # (Opcjonalnie) zapis jako obraz podglądowy
        # from matplotlib import pyplot as plt
        # plt.imsave(os.path.join(out_dir, f"{id_}.png"), dct_map, cmap='inferno')
