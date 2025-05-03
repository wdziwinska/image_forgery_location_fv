import os
import random
import shutil

# Ścieżki źródłowe
dataset_root = "../Datasets/defacto-inpainting"
base_dir = os.path.join(dataset_root, "inpainting_annotations")
img_dir = os.path.join(dataset_root, "inpainting_img")

graph_dir = os.path.join(base_dir, "graph")
mask_dir = os.path.join(base_dir, "inpaint_mask")
probe_mask_dir = os.path.join(base_dir, "probe_mask")
inpainted_img_dir = os.path.join(img_dir, "img")

# Katalog wyjściowy
output_root = "split"

# Zbieramy ID z plików JSON
file_ids = sorted(set(os.path.splitext(f)[0] for f in os.listdir(graph_dir) if f.endswith(".json")))

# Losowy podział
random.seed(42)
random.shuffle(file_ids)
n = len(file_ids)
splits = {
    "train": file_ids[:int(0.7 * n)],
    "val": file_ids[int(0.7 * n):int(0.85 * n)],
    "test": file_ids[int(0.85 * n):]
}

# Funkcja kopiująca plik jeśli istnieje
def copy_file(src_dir, dst_dir, id_, ext):
    src = os.path.join(src_dir, f"{id_}.{ext}")
    dst = os.path.join(dst_dir, f"{id_}.{ext}")
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

# Główna pętla podziału
for split, ids in splits.items():
    for id_ in ids:
        split_base = os.path.join(output_root, split, "Datasets", "defacto-inpainting")

        # Kopiowanie plików z zachowaniem struktury
        copy_file(graph_dir, os.path.join(split_base, "inpainting_annotations", "graph"), id_, "json")
        copy_file(mask_dir, os.path.join(split_base, "inpainting_annotations", "inpaint_mask"), id_, "tif")
        copy_file(probe_mask_dir, os.path.join(split_base, "inpainting_annotations", "probe_mask"), id_, "tif")
        copy_file(inpainted_img_dir, os.path.join(split_base, "inpainting_img", "img"), id_, "tif")
