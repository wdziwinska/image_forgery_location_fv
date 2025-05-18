import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Wczytaj obraz jako 2D-array (0..1)
img = plt.imread('../../Testy/inpainting-manipulated-original-the-same-pic/32ff1eb3f7df48e693a47689142619fc2DGwIiNrMiIjhPkU-27.png')[:,:,0]

# oblicz FFT i log-amplitudę
F = fftshift(fft2(img))
S = np.log(np.abs(F)+1)

# radial profile
y, x = np.indices(S.shape)
cx, cy = np.array(S.shape)/2
r = np.hypot(x-cx, y-cy).astype(int)
t = np.bincount(r.ravel(), S.ravel()) / np.bincount(r.ravel())

# wykres 1D
plt.plot(t)
plt.xlabel('Radius [px]')
plt.ylabel('Mean log-amplitude')
plt.title('Manipulated - inpainting')
# plt.title('Original')
plt.show()
