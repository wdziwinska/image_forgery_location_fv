#!/usr/bin/env python3
"""
Aplikacja Streamlit do detekcji i lokalizacji manipulacji obrazów za pomocą ConvNextUNet.
Uruchom:
    streamlit run forgery_detection_app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import timm

# ---------------- UNet-like segmentation model ----------------
class ConvNextUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone z 4 kanałami (RGB + FFT)
        self.encoder = timm.create_model(
            "convnext_base", pretrained=False, in_chans=4, features_only=True
        )
        feats = self.encoder.feature_info
        chs = [f['num_chs'] for f in feats]
        # dekodery
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
        d3 = self.up3(s4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(F.interpolate(
            d1, size=x.shape[2:], mode='bilinear', align_corners=False
        ))
        return out

# Ścieżka do wytrenowanego modelu
MODEL_PATH = 'convnext_patch_segmentation_base.pth'

@st.cache_resource
def load_model(path, device=torch.device('cpu')):
    """
    Ładuje architekturę ConvNextUNet i wczytuje wytrenowane wagi.
    """
    model = ConvNextUNet()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Transformacje dla RGB i FFT
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def   (image: Image.Image) -> torch.Tensor:
    """
    Oblicza kanał FFT (log-skalowana amplituda) obrazu w skali szarości.
    Zwraca tensor o kształcie (1, H, W).
    """
    gray = image.convert('L')
    arr = np.array(gray).astype(np.float32)
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    # normalizacja do [0,1]
    mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    mag_img = Image.fromarray((mag_norm * 255).astype(np.uint8))
    mag_img = mag_img.resize((128, 128))
    tensor = transforms.ToTensor()(mag_img)
    return tensor


def predict(image: Image.Image, model, device=torch.device('cpu'), threshold=0.5):
    """
    Wykonuje prognozę: zwraca obraz maski oraz macierz boolowską detekcji.
    """
    # Przygotowanie kanałów
    rgb_t = img_transform(image)
    fft_t = compute_fft_channel(image)
    x = torch.cat([rgb_t, fft_t], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0]
    mask_bool = probs > threshold
    mask_np = mask_bool.cpu().numpy()
    mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255), mode='L')
    mask_img = mask_img.resize(image.size)
    return mask_img, mask_np


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    st.title('Detekcja i lokalizacja manipulacji obrazu (ConvNextUNet)')
    uploaded_file = st.file_uploader(
        "Wybierz zdjęcie do analizy",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        cols = st.columns(2)
        cols[0].image(image, caption='Oryginalny obraz', use_container_width=True)

        mask_img, mask_np = predict(image, model, device)
        if not mask_np.any():
            cols[1].write('Brak wykrytej manipulacji na obrazie.')
            st.success('Brak wykrytej manipulacji.')
        else:
            cols[1].image(mask_img, caption='Maska manipulacji', use_container_width=True)
            st.error('Manipulacja wykryta!')

if __name__ == "__main__":
    main()
