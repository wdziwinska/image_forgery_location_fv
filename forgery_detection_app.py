#!/usr/bin/env python3
"""
Aplikacja Streamlit do detekcji i lokalizacji manipulacji obrazów:
1) Usunięcia (DeepLabV3 + defacto-inpainting)
2) Doklejenia (ConvNextUNet + CASIA2 + FFT)
Uruchom:
    streamlit run forgery_detection_app.py
"""

import os
import io
import base64
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
import timm

# Ustawienia stylu jasny motyw (białe tło, ciemny tekst)
st.set_page_config(page_title="Detekcja manipulacji", layout="centered")
# st.markdown(unsafe_allow_html=True)

# ---------------- Removal detection (DeepLabV3) ----------------
@st.cache_resource
def load_removal_model(device=torch.device('cpu')):
    # Utworzenie architektury DeepLabV3
    model = models.segmentation.deeplabv3_resnet50(
        pretrained=False, progress=True,
        num_classes=1, aux_loss=False
    )
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    ckpt_path = os.path.join('manipulation_detector_v1.pt')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

# Transformacja obrazu dla DeepLabV3
removal_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_removal(image: Image.Image, model, device, threshold=0.5):
    inp = removal_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)['out']
        prob = torch.sigmoid(out)[0, 0]
    mask_bool = prob > threshold
    mask_np = (mask_bool.cpu().numpy().astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_np, mode='L').resize(image.size)
    return mask_img, mask_bool.cpu().numpy()

# ---------------- Addition detection (ConvNextUNet) ----------------
class ConvNextUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "convnext_base", pretrained=False, in_chans=4, features_only=True
        )
        feats = self.encoder.feature_info
        chs = [f['num_chs'] for f in feats]
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(chs[2]*2, chs[2], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[2], chs[2], 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(chs[1]*2, chs[1], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[1], chs[1], 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(chs[0]*2, chs[0], 3, padding=1), nn.ReLU(),
            nn.Conv2d(chs[0], chs[0], 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        d3 = self.up3(s4); d3 = torch.cat([d3, s3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, s2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, s1], dim=1); d1 = self.dec1(d1)
        out = self.final(F.interpolate(
            d1, size=x.shape[2:], mode='bilinear', align_corners=False
        ))
        return out

@st.cache_resource
def load_addition_model(device=torch.device('cpu')):
    model = ConvNextUNet()
    ckpt = 'convnext_v10.pth'
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

add_rgb_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

def compute_fft_channel(image: Image.Image) -> torch.Tensor:
    gray = image.convert('L')
    arr = np.array(gray).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log1p(np.abs(fft))
    mag = (mag - mag.min())/(mag.max()-mag.min()+1e-8)
    return torch.from_numpy(mag).unsqueeze(0)

def predict_addition(image: Image.Image, model, device, threshold=0.5):
    rgb = add_rgb_transform(image)
    fft = compute_fft_channel(image)
    fft = F.interpolate(
        fft.unsqueeze(0), size=(128,128), mode='bilinear', align_corners=False
    )[0]
    x = torch.cat([rgb, fft], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0]
    mask_bool = prob > threshold
    mask_np = (mask_bool.cpu().numpy().astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_np, mode='L').resize(image.size)
    return mask_img, mask_bool.cpu().numpy()


# Funkcja do wyszukiwania maski w dwóch folderach
def find_mask(image_name, mask_dir1, mask_dir2, mask_dir3, mask_dir4):
    base_name, _ = os.path.splitext(image_name)
    mask_path1 = os.path.join(mask_dir1, f"{base_name}_gt.png")
    mask_path2 = os.path.join(mask_dir2, f"{base_name}.tif")
    mask_path3 = os.path.join(mask_dir3, f"{base_name}_gt.png")
    mask_path4 = os.path.join(mask_dir4, f"{base_name}.tif")

    if os.path.exists(mask_path1):
        return mask_path1
    elif os.path.exists(mask_path2):
        return mask_path2
    elif os.path.exists(mask_path3):
        return mask_path3
    elif os.path.exists(mask_path4):
        return mask_path4
    return None

# ---------------- Streamlit App ----------------
def main():
    st.markdown(
        "<h1 style='font-size:32px; text-align:center;'>Detekcja manipulacji obrazu:<br>usunięcia i doklejenia</h1>",
        unsafe_allow_html=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rem_model = load_removal_model(device)
    add_model = load_addition_model(device)

    uploaded = st.file_uploader('Wybierz zdjęcie', type=['png','jpg','jpeg','tif','tiff'])
    MAX_DIM = 1024
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        image_name = uploaded.name
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)

        col3, col4 = st.columns(2)

        # Wyświetlenie wgranego obrazu
        col3.image(img, caption="Wgrane zdjęcie", use_container_width=True)

        # Szukanie maski
        mask_path = find_mask(image_name, mask_dir1="./dataset/new_with_masks/val/groundtruth", mask_dir2="../../trainig_dataset/defacto-inpainting/split/val/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask", mask_dir3="./dataset/new_with_masks/test/groundtruth", mask_dir4="../trainig_dataset/defacto-inpainting/split/test/Datasets/defacto-inpainting/inpainting_annotations/inpaint_mask")
        if mask_path:
            mask_img = Image.open(mask_path)
            col4.image(mask_img, caption="Maska odpowiadająca zdjęciu", use_container_width=True)
        else:
            col4.warning("Nie znaleziono maski odpowiadającej wgranemu zdjęciu.")

        col1, col2 = st.columns(2)
        rem_mask, rem_bool = predict_removal(img, rem_model, device)
        if rem_bool.any():
            col1.image(rem_mask, caption='Wykryto manipulację - inpainting',  use_container_width=True)
            col1.error('Usunięto obiekty!')
        else:
            col1.success('Nie wykryto usunięć')

        add_mask, add_bool = predict_addition(img, add_model, device)
        if add_bool.any():
            col2.image(add_mask, caption='Wykryto manipulację - splicing',  use_container_width=True)
            col2.error('Doklejono obiekty!')
        else:
            col2.success('Nie wykryto doklejeń')

if __name__ == '__main__':
    main()
