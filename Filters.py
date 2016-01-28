#filters
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fftshift, fft, ifft
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale

n=20
# resize image to next power of two for fourier analysis
# speeds up fourier and lessens artifacts
order = max(64., 2 ** np.ceil(np.log(2 * n) / np.log(2)))
# zero pad input image
# construct the fourier filter
delta = 1
l1 = (2*np.pi)**(-4/5) * (delta)**(8/5) /5
l2 = (2*np.pi)**(-4/5) * (delta)**(-2/5) *4/5

print(l1)
print(l2)

freqs = np.zeros((order, 1))

f = fftshift(abs(np.mgrid[-1:1:2 / order])).reshape(-1, 1)
w = 2 * np.pi * f

x = np.linspace(0, 2*np.pi, 32)
x = x.reshape(-1,1)
y = l2 / (l1 * ((x/np.pi)**5 / (2*np.pi)) + l2) + np.sqrt(l1*l2) * (x/np.pi)**2 * np.sqrt(l1*((x/np.pi)**5 / (2*np.pi)) + l2 - x/np.pi/(2*np.pi)) / (l1 * ((x/np.pi)**5 / (2*np.pi)) + l2)
z = l2 / (l1 * ((x)**5 / (2*np.pi)) + l2) - np.sqrt(l1*l2) * (x)**2 * np.sqrt(l1*((x)**5 / (2*np.pi)) + l2 - x/(2*np.pi)) / (l1 * ((x)**5 / (2*np.pi)) + l2)

#"tigran":
f[1:] = l2 / (l1 * ((w[1:]/np.pi)**5 / (2*np.pi)) + l2) + np.sqrt(l1*l2) * (w[1:]/np.pi)**2 * np.sqrt(l1*((w[1:]/np.pi)**5 / (2*np.pi)) + l2 - w[1:]/np.pi/(2*np.pi)) / (l1 * ((w[1:]/np.pi)**5 / (2*np.pi)) + l2)
g = l2 / (l1 * ((w/np.pi)**5 / (2*np.pi)) + l2) + np.sqrt(l1*l2) * (w/np.pi)**2 * np.sqrt(l1*((w/np.pi)**5 / (2*np.pi)) + l2 - w/np.pi/(2*np.pi)) / (l1 * ((w/np.pi)**5 / (2*np.pi)) + l2)
#'shepp-logan":
#f[1:] = f[1:] * np.sin(w[1:] / 2) / (w[1:] / 2)

#"cosine":
#f[1:] = f[1:] * np.cos(w[1:] / 2)
#d = x* np.cos(2*np.pi*x / 2)

#"hamming":
#f[1:] = f[1:] * (0.54 + 0.46 * np.cos(w[1:]))

#"hann":
#f[1:] = f[1:] * (1 + np.cos(w[1:])) / 2

#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(x, y)
#ax1.plot(x,z)
#plt.show()

plt.scatter(x,z)
plt.show()

plt.scatter(w,g)
plt.show()

print(w)
print(x)
#print(z)
