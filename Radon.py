#Radon transform
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fftshift, fft, ifft
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale


def iradon(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear"):
    """
    Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is nxm)
    output_size : int
        Number of rows and columns in the reconstruction.
    filter : str, optional (default ramp)
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann
        Assign None to use no filter.
    interpolation : str, optional (default linear)
        Interpolation method used in reconstruction.
        Methods available: nearest, linear.

    Returns
    -------
    output : ndarray
      Reconstructed image.

    Notes
    -----
    It applies the fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta == None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    th = (np.pi / 180.0) * theta
    # if output size not specified, estimate from input radon image
    if not output_size:
        output_size = int(np.floor(np.sqrt((radon_image.shape[0]) ** 2 / 2.0)))
    n = radon_image.shape[0]

    img = radon_image.copy()
    # resize image to next power of two for fourier analysis
    # speeds up fourier and lessens artifacts
    order = max(64., 2 ** np.ceil(np.log(2 * n) / np.log(2)))
    # zero pad input image
    img.resize((order, img.shape[1]))
    # construct the fourier filter
    delta = 1
    l1 = (2*np.pi)**(-4/5) * (delta)**(8/5) /5
    l2 = (2*np.pi)**(-4/5) * (delta)**(-2/5) *4/5
    freqs = np.zeros((order, 1))

    f = fftshift(abs(np.mgrid[-1:1:2 / order])).reshape(-1, 1)
    w = 2 * np.pi * f
    # start from first element to avoid divide by zero
    if filter == "ramp":
        pass
    elif filter == "tigran":
        f[1:] = l1 / (l1 * ((w[1:]/np.pi)**5 / (2*np.pi)) + l2)
    elif filter == "shepp-logan":
        f[1:] = f[1:] * np.sin(w[1:] / 2) / (w[1:] / 2)
    elif filter == "cosine":
       f[1:] = f[1:] * np.cos(w[1:] / 2)
    elif filter == "hamming":
       f[1:] = f[1:] * (0.54 + 0.46 * np.cos(w[1:]))
    elif filter == "hann":
       f[1:] = f[1:] * (1 + np.cos(w[1:])) / 2
    elif filter == None:
        f[1:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filter)

    filter_ft = np.tile(f, (1, len(theta)))
    # apply filter in fourier domain
    projection = fft(img, axis=0) * filter_ft
    radon_filtered = np.real(ifft(projection, axis=0))
    # resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    mid_index = np.ceil(n / 2.0)

    x = output_size
    y = output_size
    [X, Y] = np.mgrid[0.0:x, 0.0:y]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

    # reconstruct image by interpolation
    if interpolation == "nearest":
        for i in range(len(theta)):
            k = np.round(mid_index + xpr * np.sin(th[i]) - ypr * np.cos(th[i]))
            reconstructed += radon_filtered[
                ((((k > 0) & (k < n)) * k) - 1).astype(np.int), i]
    elif interpolation == "linear":
        for i in range(len(theta)):
          t = xpr*np.sin(th[i]) - ypr*np.cos(th[i])
          a = np.floor(t)
          b = mid_index + a
          b0 = ((((b + 1 > 0) & (b + 1 < n)) * (b + 1)) - 1).astype(np.int)
          b1 = ((((b > 0) & (b < n)) * b) - 1).astype(np.int)
          reconstructed += (t - a) * radon_filtered[b0, i] + \
                           (a - t + 1) * radon_filtered[b1, i]
    else:
        raise ValueError("Unknown interpolation: %s" % interpolation)

    return reconstructed * np.pi / (2 * len(th))

#MAIN PROGRAM	

#define delta
delta = 1
#define lambdas
l1 = (2*np.pi)**(-4/5) * (delta/np.sqrt(2))**(8/5) /5
l2 = (2*np.pi)**(-4/5) * (delta/np.sqrt(2))**(-2/5) *4/5




image = imread(data_dir + "/phantom.png", as_grey=True)
image = rescale(image, scale=0.4)

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

#ax1.set_title("Original")
#ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
#ax2.set_title("Radon transform\n(Sinogram)")
#ax2.set_xlabel("Projection angle (deg)")
#ax2.set_ylabel("Projection position (pixels)")
#ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
 #          extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

#fig.tight_layout()
#plt.show()

reconstruction_noF = iradon(sinogram, theta=theta, filter = None)
reconstruction_F = iradon(sinogram, theta=theta, filter = 'hann')
#error = reconstruction_fbp - image
#print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable': 'box-forced'})
ax1.set_title("Reconstruction without filtering")
ax1.imshow(reconstruction_noF, cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction with filtering")
ax2.imshow(reconstruction_F, cmap=plt.cm.Greys_r)#- image, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()
