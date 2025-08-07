# Image Manipulation Detection using CNN and FFT  

This project uses Convolutional Neural Networks (CNN) and Fast Fourier Transform (FFT) to detect image manipulation. The model focuses on detecting two types of manipulations: **inpainting** and **splicing**.

## Project Overview  
- **Goal**: Develop a deep learning model capable of detecting manipulated images using frequency analysis.
- **Dataset**: The project utilizes two datasets: **CASIA2**, which contains a variety of manipulated and authentic images — splicing manipulation — and the **Defacto-inpainting** dataset, which focuses specifically on image inpainting scenarios.
- **Techniques**:  
  - FFT (Fast Fourier Transform) to preprocess images.  
  - CNN (Convolutional Neural Network) for image classification. 

## Requirements  
Ensure you have the following packages installed to run the project:  

```bash
torch==2.7.0
torchvision==0.22.0
numpy==2.2.5
Pillow==11.2.1
opencv-python==4.8.0.74
tqdm==4.67.1
streamlit==1.45.0
```

# Apliaction results

## Splicing

<img width="790" height="1110" alt="obraz" src="https://github.com/user-attachments/assets/8f23db94-d00a-4b97-af34-3f8c9b12b45e" />

<img width="775" height="1050" alt="obraz" src="https://github.com/user-attachments/assets/ccaec2ba-bb57-43a4-855b-efa0c5f2239d" />

<img width="775" height="1052" alt="obraz" src="https://github.com/user-attachments/assets/5253e7b3-f0a4-4e42-a8ba-c00f62139668" />

<img width="771" height="777" alt="obraz" src="https://github.com/user-attachments/assets/018be733-497f-4169-b2f4-5e6d0b1739d2" />

## Inpainting

<img width="776" height="1285" alt="obraz" src="https://github.com/user-attachments/assets/8a73ad4e-bc54-446b-9a9c-f23b39ee0339" />

<img width="773" height="1326" alt="obraz" src="https://github.com/user-attachments/assets/cbb9b697-ca51-4adf-92d4-779023f15761" />

<img width="781" height="1057" alt="obraz" src="https://github.com/user-attachments/assets/ee5f43d9-901d-48e0-8147-e732579ed4a1" />

<img width="775" height="838" alt="obraz" src="https://github.com/user-attachments/assets/a142b440-c05c-4fd7-8212-aba3444048b9" />
