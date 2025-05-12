#!/usr/bin/env python3
"""
Aplikacja Streamlit do detekcji i lokalizacji manipulacji obrazów za pomocą masek.
Uruchom:
    streamlit run forgery_detection_app.py
"""

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from scipy.fftpack import dct

# === Parametry ===
MODEL_PATH = 'trained_models/manipulation_detector_v4.pt'
INPUT_SIZE = (256, 256)

# === Oblicz DCT z obrazu
def compute_dct(image_pil, size=INPUT_SIZE):
    gray = np.array(image_pil.convert("L").resize(size)).astype(np.float32)
    coeff = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    coeff = np.abs(coeff)
    coeff = np.log1p(coeff)
    coeff = (coeff - coeff.min()) / (coeff.max() - coeff.min() + 1e-8)
    return torch.tensor(coeff).unsqueeze(0).float()

# === Model z 4-kanałowym wejściem
@st.cache_resource
def load_model(path, device=torch.device('cpu')):
    """
    Ładuje model DeepLabV3-ResNet50 dostosowany do detekcji maski manipulacji,
    uwzględniając aux_loss, aby poprawnie załadować główne i pomocnicze wagi.
    """
    model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
    # Modyfikujemy wejście na 4 kanały
    new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        pretrained = deeplabv3_resnet50(pretrained=True)
        new_conv1.weight[:, :3] = pretrained.backbone.conv1.weight
        new_conv1.weight[:, 3] = pretrained.backbone.conv1.weight[:, 0] * 0.0
    model.backbone.conv1 = new_conv1
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# === Transformacja obrazu
transform_rgb = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
])

def predict(image_pil, model, device=torch.device('cpu'), threshold=0.5):
    rgb_tensor = transform_rgb(image_pil)
    dct_tensor = compute_dct(image_pil)
    combined = torch.cat([rgb_tensor, dct_tensor], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(combined)['out']
    probs = torch.sigmoid(out)[0, 0]
    mask_bool = probs > threshold
    mask_np = mask_bool.cpu().numpy()
    mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255), mode='L')
    mask_img = mask_img.resize(image_pil.size)
    return mask_img, mask_np

# === Główna aplikacja Streamlit
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    st.title('Detekcja i lokalizacja manipulacji obrazu')

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
            st.write('Manipulacja wykryta.')

if __name__ == "__main__":
    main()
