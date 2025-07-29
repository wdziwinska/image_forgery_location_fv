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
