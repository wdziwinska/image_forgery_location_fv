import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ścieżki do katalogów
INPUT_DIRS = ['./split/test/Datasets/defacto-inpainting/inpainting_img/img', './split/train/Datasets/defacto-inpainting/inpainting_img/img', './split/valDatasets/defacto-inpainting/inpainting_img/img']

# Obsługiwane formaty obrazów
IMAGE_EXTENSIONS = ['.png', '.jpg', '.tif']

def process_fft(image_path, output_path):
    # Wczytaj obraz w skali szarości
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'Nie można wczytać obrazu: {image_path}')
        return

    # Wykonaj FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    # spektrum amplitudy w skali logarytmicznej
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Wyświetl i zapisz spektrum
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('FFT Spectrum')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # Przetwarzanie obrazów
    for input_dir in INPUT_DIRS:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, 'split')
                    output_dir = os.path.join('fft_spectrum', relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f'{os.path.splitext(file)[0]}_fft.png')
                    process_fft(image_path, output_path)
                    print(f'Przetworzono: {file}')

    print('Przetwarzanie zakończone.')