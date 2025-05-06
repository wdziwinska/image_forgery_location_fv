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

# Ścieżka do wytrenowanego modelu detekcji manipulacji (segmentacji)
MODEL_PATH = 'trained_models/manipulation_detector_v1.pt'

# Korzystamy z nowego API cache dla zasobów (modelu)
@st.cache_resource
def load_model(path, device=torch.device('cpu')):
    """
    Ładuje model DeepLabV3-ResNet50 dostosowany do detekcji maski manipulacji,
    uwzględniając aux_loss, aby poprawnie załadować główne i pomocnicze wagi.
    """
    model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Transformacje zgodne z treningiem
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict(image, model, device=torch.device('cpu'), threshold=0.5):
    """
    Wykonuje prognozę: zwraca obraz maski oraz macierz boolowską detekcji.
    """
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)['out']
    probs = torch.sigmoid(out)[0, 0]
    mask_bool = probs > threshold
    mask_np = mask_bool.cpu().numpy()
    mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255), mode='L')
    mask_img = mask_img.resize(image.size)
    return mask_img, mask_np

# Główna aplikacja Streamlit

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, device)

    st.title('Detekcja i lokalizacja manipulacji obrazu (Maska)')
    uploaded_file = st.file_uploader("Wybierz zdjęcie do analizy", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        cols = st.columns(2)
        cols[0].image(image, caption='Oryginalny obraz', use_column_width=True)

        mask_img, mask_np = predict(image, model, device)
        if not mask_np.any():
            cols[1].write('Brak wykrytej manipulacji na obrazie.')
            st.success('Brak wykrytej manipulacji.')
        else:
            cols[1].image(mask_img, caption='Maska manipulacji', use_column_width=True)
            st.write('Manipulacja wykryta.')

if __name__ == "__main__":
    main()
