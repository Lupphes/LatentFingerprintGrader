"""
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

Both licenses are MIT
"""

import cv2
import math
import numpy as np

from skimage.filters import gaussian
from skimage.morphology import binary_opening, binary_closing

from .functions import fast_cartoon_texture


def construct_dictionary(ori_num=30):
    """
    Contstruct the ridge dictionary which is then used for estimation
    introduced in:
    Segmentation and enhancement of latent fingerprints: A coarse to fine ridge structure dictionary
    Cao, Kai and Liu, Eryun and Jain, Anil K
    0162-8828

    and improved in:
    End-to-End Latent Fingerprint Search
    Cao, Kai and Nguyen, Dinh-Luan and Tymoszek, Cori and Jain, AK
    """
    ori_dict = []
    s = []
    for i in range(ori_num):
        ori_dict.append([])
        s.append([])

    patch_size2 = 16
    patch_size = 32
    dict_all = []
    spacing_all = []
    ori_all = []
    Y, X = np.meshgrid(range(-patch_size2, patch_size2),
                       range(-patch_size2, patch_size2))

    for spacing in range(4, 13):
        for valley_spacing in range(max(2, spacing // 2 - 2), spacing // 2):
            ridge_spacing = spacing - valley_spacing
            for k in range(ori_num):
                theta = np.pi / 2 - k * np.pi / ori_num
                X_r = X * np.cos(theta) - Y * np.sin(theta)
                for offset in range(0, spacing - 1, 2):
                    X_r_offset = X_r + offset + ridge_spacing / 2
                    X_r_offset = np.remainder(X_r_offset, spacing)
                    Y1 = np.zeros((patch_size, patch_size))
                    Y2 = np.zeros((patch_size, patch_size))
                    Y1[X_r_offset <=
                        ridge_spacing] = X_r_offset[X_r_offset <= ridge_spacing]
                    Y2[X_r_offset > ridge_spacing] = X_r_offset[X_r_offset >
                                                                ridge_spacing] - ridge_spacing
                    element = -np.sin(2 * math.pi * (Y1 / ridge_spacing / 2)) + \
                        np.sin(2 * math.pi * (Y2 / valley_spacing / 2))
                    element = element.reshape(patch_size * patch_size,)
                    element = element - np.mean(element)
                    element = element / np.linalg.norm(element)
                    ori_dict[k].append(element)
                    s[k].append(spacing)
                    dict_all.append(element)
                    spacing_all.append(1.0 / spacing)
                    ori_all.append(theta)
    for i in range(len(ori_dict)):
        ori_dict[i] = np.asarray(ori_dict[i])
        s[k] = np.asarray(s[k])
    dict_all = np.asarray(dict_all)
    dict_all = np.transpose(dict_all)
    spacing_all = np.asarray(spacing_all)
    ori_all = np.asarray(ori_all)

    return ori_dict, s, dict_all, ori_all, spacing_all


def smooth_dir_map(dir_map, sigma=2.0, mask=None):
    cos2Theta = np.cos(dir_map * 2)
    sin2Theta = np.sin(dir_map * 2)
    if mask is not None:
        assert (dir_map.shape[0] == mask.shape[0])
        assert (dir_map.shape[1] == mask.shape[1])
        cos2Theta[mask == 0] = 0
        sin2Theta[mask == 0] = 0

    cos2Theta = gaussian(cos2Theta, sigma, channel_axis=None, mode='reflect')
    sin2Theta = gaussian(sin2Theta, sigma, channel_axis=None, mode='reflect')

    dir_map = np.arctan2(sin2Theta, cos2Theta) * 0.5

    return dir_map


def SSIM(img, temp_img, block_size=16, thr=0.65):
    """
    Calculates structural similarity between two images
    introduced in:
    Segmentation and enhancement of latent fingerprints: A coarse to fine ridge structure dictionary
    Cao, Kai and Liu, Eryun and Jain, Anil K
    0162-8828

    and improved in:
    End-to-End Latent Fingerprint Search
    Cao, Kai and Nguyen, Dinh-Luan and Tymoszek, Cori and Jain, AK
    """
    h, w = img.shape[:2]
    patch_size = 64
    blkH = h / block_size
    blkW = w / block_size
    blkH = int(blkH)
    blkW = int(blkW)
    quality = np.zeros((blkH, blkW))

    R = 500
    blocks_in_patch = int(patch_size / block_size)

    def get_weights(h, w, c, sigma=None):
        Y, X = np.mgrid[0:h, 0:w]
        x0 = w // 2
        y0 = h // 2
        if sigma is None:
            sigma = (np.max([h, w]) * 1. / 2) ** 2
        weight = np.exp(-((X - x0) * (X - x0) + (Y - y0) * (Y - y0)) / sigma)
        weight = np.stack((weight,) * c, axis=2)
        return weight

    weight = get_weights(blocks_in_patch, blocks_in_patch, 1, sigma=None)
    sigma = (patch_size / 2) ** 2
    weight_pixel = get_weights(patch_size, patch_size, 1, sigma=sigma)
    for i in range(blkH - blocks_in_patch + 1):
        for j in range(blkW - blocks_in_patch + 1):
            patch = img[i * block_size:i * block_size + patch_size,
                        j * block_size:j * block_size + patch_size]

            patch = patch - np.median(patch)
            patch = patch / (np.linalg.norm(patch) + R)
            patch = patch * weight_pixel[:, :, 0]
            patch = patch.reshape(patch_size * patch_size, )
            temp_patch = temp_img[i * block_size:i * block_size + patch_size,
                                  j * block_size:j * block_size + patch_size]
            temp_patch = temp_patch - np.median(temp_patch)
            temp_patch = temp_patch / (np.linalg.norm(temp_patch) + R)
            temp_patch = temp_patch * weight_pixel[:, :, 0]
            temp_patch = temp_patch.reshape(patch_size * patch_size, )

            simi = (np.dot(patch, temp_patch)) * weight[:, :, 0]
            quality[i:i + blocks_in_patch, j:j + blocks_in_patch] += simi

    quality = cv2.GaussianBlur(quality, (5, 5), 0)
    blkmask = quality > thr
    blkmask = binary_closing(blkmask, np.ones((3, 3))).astype(np.int64)
    blkmask = binary_opening(blkmask, np.ones((3, 3))).astype(np.int64)

    return blkmask


def get_quality_map_dict(img, dict, ori, spacing, block_size=16, process=False, R=500.0, t=0.05):
    """
    Dictionary approach to look for ridges and estimate "quality", orientation and 
    frequency
    introduced in:
    Segmentation and enhancement of latent fingerprints: A coarse to fine ridge structure dictionary
    Cao, Kai and Liu, Eryun and Jain, Anil K
    0162-8828

    and improved in:
    End-to-End Latent Fingerprint Search
    Cao, Kai and Nguyen, Dinh-Luan and Tymoszek, Cori and Jain, AK
    """
    if img.dtype == 'uint8':
        img = img.astype(np.float)
    if process:
        img = fast_cartoon_texture(img)
    h, w = img.shape

    blkH, blkW = h // block_size, w // block_size
    quality_map = np.zeros((blkH, blkW), dtype=np.float)
    dir_map = np.zeros((blkH, blkW), dtype=np.float)
    fre_map = np.zeros((blkH, blkW), dtype=np.float)

    patch_size = np.sqrt(dict.shape[0])
    patch_size = patch_size.astype(np.int64)
    pad_size = (patch_size - block_size) // 2
    img = np.lib.pad(img, (pad_size, pad_size), 'symmetric')

    patches = []
    pixel_list = []

    r = 1

    for i in range(r, blkH - r):
        for j in range(r, blkW - r):
            pixel_list.append((i, j))
            patch = img[i * block_size:i * block_size + patch_size,
                        j * block_size:j * block_size + patch_size].copy()

            patch = patch.reshape(patch_size * patch_size,)
            patch = patch - np.mean(patch)
            patch = patch / (np.linalg.norm(patch) + R)

            patch[patch > t] = 0.0
            patch[patch < -t] = -0.0
            # The above lines are a bug according to
            # https://github.com/prip-lab/MSU-LatentAFIS/issues/4
            # The lines below should fix this issue
            # patch[patch > t] = t
            # patch[patch < -t] = -t
            # I did not see the issue, but with my experiments I
            # did not see improvement or worsen in my case
            # I still let it here for possible future updates - Ondrej Sloup (Lupphes)

            patches.append(patch)

    patches = np.asarray(patches)
    simi = abs(np.dot(patches, dict))
    similar_ind = np.argmax(simi, axis=1)

    n = 0
    for i in range(r, blkH - r):
        for j in range(r, blkW - r):
            quality_map[i, j] = simi[n, similar_ind[n]]
            dir_map[i, j] = ori[similar_ind[n]]
            fre_map[i, j] = spacing[similar_ind[n]]
            n += 1

    for i in range(r):
        fre_map[i] = fre_map[r]
        dir_map[i] = dir_map[r]
        fre_map[-(r - i) - 1] = fre_map[-r - 1]
        dir_map[-(r - i) - 1] = fre_map[-r - 1]
        fre_map[:, i] = fre_map[:, r]
        dir_map[:, i] = dir_map[:, r]
        fre_map[:, -(r - i) - 1] = fre_map[:, -r - 1]
        dir_map[:, -(r - i) - 1] = fre_map[:, -r - 1]

    quality_map = cv2.GaussianBlur(quality_map, (5, 5), 0)
    dir_map = smooth_dir_map(dir_map, sigma=1.5)
    fre_map = cv2.GaussianBlur(fre_map, (3, 3), 1)
    return quality_map, dir_map, fre_map
