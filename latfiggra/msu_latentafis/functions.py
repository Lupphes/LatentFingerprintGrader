"""functions.py
This part of the code is adapted from:
Manuel Aguado Martinez
MSU Latent Automatic Fingerprint Identification System (AFIS) -- Logarithmic Gabor filter fork
https://github.com/manuelaguadomtz/MSU-LatentAFIS (b24eb5eb57c43932e56b82336c5bd188a9a3862e)
which was adapted from:
End-to-End Latent Fingerprint Search
Cao, Kai and Nguyen, Dinh-Luan and Tymoszek, Cori and Jain, AK
MSU Latent Automatic Fingerprint Identification System (AFIS)
https://github.com/prip-lab/MSU-LatentAFIS (6dd2dab9767dce3940689150e73b072c30ec08e1)

I forked the newest adaptation and built a new structure here:
https://github.com/Lupphes/MSU-LatentAFIS
which I then restructured to be usable in LatFigGra.
- Ondřej Sloup (xsloup02)

Both licenses are MIT
"""

import math
import cv2

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import scipy.ndimage

from typing import Tuple


def local_constrast_enhancement_gaussian(img, sigma=15):
    """
    Gaussian blur is applied to normalised image
    """
    img = img.astype(np.float32)

    meanV = cv2.GaussianBlur(img, (sigma, sigma), 0)
    normalized = img - meanV
    var = abs(normalized)
    var = cv2.GaussianBlur(var, (sigma, sigma), 0)

    normalized = normalized / (var + 10) * 0.75
    normalized = np.clip(normalized, -1, 1)
    normalized = (normalized + 1) * 127.5
    return normalized


def lowpass_filtering(img, L):
    """
    Fourier transform filtering needed by Texture+Cartoon decomposition.
    Used Fourier transform insted of lowpass filter
    """
    h, w = img.shape
    h2, w2 = L.shape

    img = cv2.copyMakeBorder(img, 0, h2 - h, 0, w2 - w,
                             cv2.BORDER_CONSTANT, value=0)

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    img_fft = img_fft * L

    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h, :w]
    return rec_img


def compute_gradient_norm(input):
    input = input.astype(np.float32)

    Gx, Gy = np.gradient(input)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    return out


def fast_cartoon_texture(img, sigma=2.5, show=False):
    """
    Texture+Cartoon decomposition algorithm
    introduced in:
    A. Buades, T. M. Le, J. Morel and L. A. Vese, "Fast Cartoon + Texture Image Filters,"
    in IEEE Transactions on Image Processing,
    vol. 19, no. 8, pp. 1978-1986, Aug. 2010, doi: 10.1109/TIP.2010.2046605.
    """
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** int(math.ceil(math.log(h, 2)))
    w2 = 2 ** int(math.ceil(math.log(w, 2)))

    FFTsize = np.max([h2, w2])
    x, y = np.meshgrid(range(-int(FFTsize / 2), int(FFTsize / 2)),
                       range(-int(FFTsize / 2), int(FFTsize / 2)))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r / FFTsize

    L = 1. / (1 + (2 * math.pi * r * sigma) ** 4)
    img_low = lowpass_filtering(img, L)

    gradim1 = compute_gradient_norm(img)
    gradim1 = lowpass_filtering(gradim1, L)

    gradim2 = compute_gradient_norm(img_low)
    gradim2 = lowpass_filtering(gradim2, L)

    diff = gradim1 - gradim2
    ar1 = np.abs(gradim1)
    diff[ar1 > 1] = diff[ar1 > 1] / ar1[ar1 > 1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff - cmin) / (cmax - cmin)
    weight[diff < cmin] = 0
    weight[diff > cmax] = 1

    u = weight * img_low + (1 - weight) * img
    temp = img - u
    lim = 20

    temp1 = (temp + lim) * 255 / (2 * lim)
    temp1[temp1 < 0] = 0
    temp1[temp1 > 255] = 255
    v = temp1
    if show:
        plt.imshow(v, cmap='gray')
        plt.show()
    return v


def STFT(img, R=100):
    """
    Short time Fourier transform used to enhance the image
    introduced in:
    CHIKKERUR, Sharat, Alexander N CARTWRIGHT a Venu GOVINDARAJU.
    Fingerprint enhancement using STFT analysis.
    Pattern recognition [online]. OXFORD: Elsevier, 2007, 40(1),
    198-211 [cit. 2022-05-06]. ISSN 0031-3203.
    Available from: doi:10.1016/j.patcog.2006.05.036
    """
    patch_size = 64
    block_size = 16
    ovp_size = (patch_size - block_size) // 2
    h0, w0 = img.shape
    img = cv2.copyMakeBorder(img, ovp_size, ovp_size, ovp_size, ovp_size,
                             cv2.BORDER_CONSTANT, value=0)
    h, w = img.shape
    blkH = (h - patch_size) // block_size
    blkW = (w - patch_size) // block_size

    # -------------------------
    # Bandpass filter
    # -------------------------
    RMIN = 3  # min allowable ridge spacing
    RMAX = 18  # maximum allowable ridge spacing
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN
    patch_size2 = int(patch_size / 2)
    x, y = np.meshgrid(range(-patch_size2, patch_size2),
                       range(-patch_size2, patch_size2))
    r = np.sqrt(x * x + y * y) + 0.0001

    dRLow = 1. / (1 + (r / FHIGH)**4)  # low pass butterworth filter
    dRHigh = 1. / (1 + (FLOW / r)**4)  # high pass butterworth filter
    dBPass = dRLow * dRHigh  # bandpass

    sigma = patch_size / 3
    weight = np.exp(-(x * x + y * y) / (sigma * sigma))
    rec_img = np.zeros((h, w))
    for i in range(0, blkH):
        for j in range(0, blkW):
            patch = img[i * block_size:i * block_size + patch_size,
                        j * block_size:j * block_size + patch_size].copy()
            patch = patch - np.median(patch)
            f = np.fft.fft2(patch)
            fshift = np.fft.fftshift(f)

            filtered = dBPass * fshift
            norm = np.linalg.norm(filtered)
            filtered = filtered / (norm + 0.0001)
            f_ifft = np.fft.ifftshift(filtered)
            rec_patch = np.real(np.fft.ifft2(f_ifft)) * weight
            rec_img[i * block_size:i * block_size + patch_size,
                    j * block_size:j * block_size + patch_size] += rec_patch

    rec_img = rec_img[ovp_size:ovp_size + h0, ovp_size:ovp_size + w0]
    img = (rec_img - np.median(rec_img)) / (np.std(rec_img) + 0.000001)
    img = (img * 14 + 127)
    img[img < 0] = 0
    img[img > 255] = 255
    rec_img = (rec_img - np.min(rec_img))
    rec_img /= (np.max(rec_img) - np.min(rec_img) + 0.0001)

    return img


class LogGaborFilter():
    """
    A logarithmic Gabor filter
    introduced in:
    Wei Wang, Jianwei Li, Feifei Huang, Hailiang Feng,
    Design and implementation of Log-Gabor filter in fingerprint image enhancement
    ISSN 0167-8655
    https://doi.org/10.1016/j.patrec.2007.10.004

    Implemented by Lic. Manuel Aguado Martínez in:
    https://github.com/manuelaguadomtz/MSU-LatentAFIS for the
    Aguado Martínez, Manuel & Hernández-Palancar, José & Castillo-Rosado,
    Katy & Cupull-Gómez, Rodobaldo & Kauba, Christof & Kirchgasser, Simon & Uhl,
    Andreas. (2021). Document scanners for minutiae-based palmprint recognition:
    a feasibility study. Pattern Analysis and Applications.
    24. 1-14. 10.1007/s10044-020-00923-3.
    """
    __copyright__ = 'Copyright 2020'
    __author__ = u'Lic. Manuel Aguado Martínez'

    def __init__(self, orient_map, freq_map, curv_map=None, mask=None,
                 bsize=16, sigma=8, wsize=64, sigma_orient=0.174,
                 sigma_radio=0.4):
        """Creates a Log Gabor Filter"""
        # Maps
        self.orient_map = orient_map
        self.freq_map = freq_map
        self.curv_map = curv_map
        self.mask = mask

        # Params
        self.bsize = bsize
        self.wsize = wsize
        self.sigma = sigma
        self.sigma_orient = sigma_orient
        self.sigma_radio = sigma_radio

        if self.curv_map is None:
            self.curv_map = np.zeros_like(self.orient_map)

        if self.mask is None:
            self.mask = np.ones_like(self.orient_map)

        self.gfilters = self._get_filters()
        self.wfilter = self._get_wind_filter()

    def _get_filters(self):
        """Creates the gabor filter"""

        blk_h, blk_w = self.freq_map.shape

        # Creating grid
        x, y = np.meshgrid(range(self.wsize), range(self.wsize))

        # Centered grid
        center = self.wsize // 2
        x_c, y_c = x - center, y - center

        # Grid radius
        radius = np.sqrt(x_c * x_c + y_c * y_c) / (self.wsize - 1)

        # Filters
        g_filters = [[None] * blk_w for _ in range(blk_h)]

        for i in range(blk_h):
            for j in range(blk_w):

                if not self.mask[i, j]:
                    continue

                freq = self.freq_map[i, j]
                ori = self.orient_map[i, j]
                curv = self.curv_map[i, j]

                # Creating log gabor filter
                freq_radius = radius / freq
                freq_radius[radius == 0] = 1
                log_gabor = (-0.5 * (np.log(freq_radius)) ** 2)
                log_gabor = log_gabor / (self.sigma_radio ** 2)
                log_gabor = np.exp(log_gabor)
                log_gabor[radius == 0] = 0

                # Rotating
                x_rot = x_c * np.cos(ori) + y_c * np.sin(ori)
                y_rot = y_c * np.cos(ori) - x_c * np.sin(ori)
                d_theta = np.arctan2(y_rot, x_rot)

                # Calculating spread
                spread = (-0.5 * d_theta ** 2)
                spread = spread / ((self.sigma_orient + curv) ** 2)
                spread = np.exp(spread)

                # Final filter
                amp = 1
                log_gabor = log_gabor * spread
                g_filters[i][j] = amp * log_gabor / np.max(log_gabor)

        return g_filters

    def _get_wind_filter(self):
        """
        Creates a filter to fusion windows
        """
        # Creating grid
        x, y = np.meshgrid(range(self.wsize), range(self.wsize))

        # Centered grid
        center = self.wsize // 2
        x_c, y_c = x - center, y - center

        # Wind filter
        wfilter = np.exp(-0.5 * (x_c ** 2 + y_c ** 2) / self.sigma ** 2)

        return wfilter

    def apply(self, img) -> Tuple[npt.NDArray, np.float64]:
        """
        Apply the filter over an image
        """

        # Padding input image
        ovp_size = (self.wsize - self.bsize) // 2
        img = np.lib.pad(img, (ovp_size, ovp_size), 'symmetric')
        fimage = np.zeros_like(img, dtype=np.float64)

        for i in range(self.orient_map.shape[0]):
            for j in range(self.orient_map.shape[1]):

                if not self.mask[i, j]:
                    continue

                x0, x1 = i * self.bsize, i * self.bsize + self.wsize
                y0, y1 = j * self.bsize, j * self.bsize + self.wsize

                dwind = img[x0:x1, y0:y1]
                dwind = dwind * self.wfilter
                dwind = np.fft.fft2(dwind)
                dwind = np.fft.fftshift(dwind)
                dwind *= self.gfilters[i][j]
                dwind = np.fft.ifftshift(dwind)
                dwind = np.fft.ifft2(dwind).real

                fimage[x0:x1, y0:y1] += dwind

        fimage = fimage[ovp_size: -ovp_size, ovp_size: -ovp_size]

        minimum = np.min(fimage)
        maximum = np.max(fimage)
        fimage = (fimage - minimum) / (maximum - minimum) * 255

        thr = -minimum / (maximum - minimum) * 255

        return fimage, thr


def get_gabor_filters(angle_inc=3, fre_num=30):
    """
    Function to define orientations used in Gabor filter with a kernel
    """
    ori_num = 180 // angle_inc
    gaborfilter = np.zeros((ori_num, fre_num), dtype=object)
    for i in range(ori_num):
        ori = i * angle_inc / 180.0 * math.pi
        for j in range(fre_num):
            if j < 5:
                continue

            from skimage.filters import gabor_kernel
            kernel = gabor_kernel(j * 0.01, theta=ori, sigma_x=3, sigma_y=3)
            kernel = kernel.real

            kernel = kernel - np.mean(kernel)
            norm = np.linalg.norm(kernel)

            kernel = kernel / (norm + 0.00001)
            kernel = kernel.real * 255
            t = np.asarray(kernel, np.int16)
            gaborfilter[i, j] = t

    return gaborfilter


def gabor_filtering_pixel2(img, dir_map, fre_map, mask=None, block_size=16, angle_inc=3, gabor_filters=None):
    """
    Apply the Gabor filter to the image and enhance it
    """
    h, w = img.shape
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)

    if block_size > 1:
        cos2Theta = np.cos(dir_map * 2)
        sin2Theta = np.sin(dir_map * 2)
        cos2Theta = scipy.ndimage.interpolation.zoom(cos2Theta, block_size)
        sin2Theta = scipy.ndimage.interpolation.zoom(sin2Theta, block_size)
        frequency = scipy.ndimage.interpolation.zoom(fre_map, block_size)
        angle = np.arctan2(sin2Theta, cos2Theta) * 0.5
    else:
        angle = dir_map
        frequency = fre_map

    angle = angle / math.pi * 180
    angle = angle.astype(int)
    angle[angle < 0] = angle[angle < 0] + 180
    angle[angle == 180] = 0
    angle_ind = angle // angle_inc
    frequency_ind = np.around(frequency * 100).astype(int)

    if gabor_filters is None:
        gabor_filters = get_gabor_filters()

    img = img.astype(np.int32)
    h, w = img.shape
    enh_img = np.zeros((h, w), dtype=np.int32)

    mask[:15, :] = 0
    mask[:, :15] = 0
    mask[h-15:h, :] = 0
    mask[:, w-15:w] = 0

    candi_ind = np.where(mask > 0)

    candi_num = len(candi_ind[0])

    def apply_filter(src_arr, dst_arr, gabor_filters, idx_list, frequency_ind, angle_ind):
        for p, i, j in idx_list:
            frequency_ind_b = frequency_ind[i, j]
            if frequency_ind_b < 5 or frequency_ind_b >= 30:
                continue
            angle_ind_b = angle_ind[i, j]
            kernel = gabor_filters[angle_ind_b, frequency_ind_b]
            kh, kw = kernel.shape
            sh = kh >> 1
            sw = kw >> 1
            dst_arr[p] = np.sum(
                src_arr[i - sh:i + sh + 1, j - sw:j + sw + 1] * kernel)
        return

    from multiprocessing import Process, Array
    threads = []
    thread_num = 1
    pixels_per_thread = candi_num // thread_num
    result_array = Array('f', candi_num)

    for k in range(0, thread_num):
        idx_list = []
        for n in range(0, pixels_per_thread):
            p = k * pixels_per_thread + n
            if p >= candi_num:
                break
            idx_list.append((p, candi_ind[0][p], candi_ind[1][p]))
        t = Process(target=apply_filter, args=(img, result_array,
                    gabor_filters, idx_list, frequency_ind, angle_ind))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for k in range(candi_num):
        i = candi_ind[0][k]
        j = candi_ind[1][k]
        enh_img[i, j] = result_array[k]

    enh_img = (enh_img - np.min(enh_img) + 0.0001) / \
        (np.max(enh_img) - np.min(enh_img) + 0.0001) * 255
    return enh_img
