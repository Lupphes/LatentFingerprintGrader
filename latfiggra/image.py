"""image.py
@Author: Ondřej Sloup (xsloup02)
@Date: 07.05.2022
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import numpy.typing as npt

from pathlib import Path

from .contrast_types import ContrastTypes, ThresholdFlags

from typing import Dict


class Image:
    def __init__(self, image):
        self.image: npt.NDArray = image

    def image_to_grayscale(self) -> None:
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def image_to_color(self) -> None:
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

    def calculate_histogram(self) -> None:
        return cv2.calcHist(self.image, [0], None, [256], [0, 256])

    def calculate_threshold(self, threshold_type=ThresholdFlags.OTSU, specified_threshold=128, maxval=255) -> None:
        threshold = None
        if threshold_type == ThresholdFlags.GAUSSIAN:
            self.image = cv2.adaptiveThreshold(self.image, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == ThresholdFlags.MEAN:
            self.image = cv2.adaptiveThreshold(self.image, maxval, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == ThresholdFlags.OTSU:
            threshold, self.image = cv2.threshold(
                self.image, 0, maxval, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif threshold_type == ThresholdFlags.TRIANGLE:
            threshold, self.image = cv2.threshold(
                self.image, 0, maxval, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
        elif threshold_type == ThresholdFlags.MANUAL:
            threshold, self.image = cv2.threshold(
                self.image, specified_threshold, maxval, cv2.THRESH_BINARY)
        else:
            raise NotImplementedError(
                "This part of program was not implemented yet. Wait for updates")

        return threshold

    def invert_binary(self) -> None:
        self.image = cv2.bitwise_not(self.image)

    def apply_contrast(self, contrast_type=ContrastTypes.CLAHE) -> None:
        if contrast_type == ContrastTypes.CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.image = clahe.apply(self.image)
        elif contrast_type == ContrastTypes.WEBER:
            pass
        elif contrast_type == ContrastTypes.MICHELSON:
            min = np.uint16(np.min(self.image))
            max = np.uint16(np.max(self.image))

            contrast = (max-min) / (max+min)
            self.image = cv2.convertScaleAbs(self.image, alpha=contrast)
        elif contrast_type == ContrastTypes.RMS:
            contrast = self.image.std()
            self.image = cv2.convertScaleAbs(self.image, alpha=contrast)
        else:
            raise NotImplementedError(
                "This part of program was not implemented yet. Wait for updates")

    def apply_sobel_2d(self) -> None:
        # The Sobel kernels can also be thought of as 3×3 approximations to first-derivative-of-Gaussian kernels.
        Dxf = cv2.Sobel(self.image, cv2.CV_8UC1, 1, 0, ksize=3)  # X
        Dyf = cv2.Sobel(self.image, cv2.CV_8UC1, 0, 1, ksize=3)  # Y

        self.image = cv2.Sobel(Dxf, cv2.CV_8UC1, 0, 1, ksize=3)

        return Dxf, Dyf

    def apply_gabor(self, ksize=31, sigma=1, theta=0, lambd=1.0, gamma=0.02, psi=0) -> None:
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
        self.image = cv2.filter2D(
            self.image, cv2.CV_64F, kernel)

    def apply_median(self, ksize=3) -> None:
        self.image = cv2.medianBlur(self.image, ksize=ksize)

    def apply_gaussian(self, ksize=23, sigma=8) -> None:
        self.image = cv2.GaussianBlur(
            self.image, (ksize, ksize), cv2.BORDER_DEFAULT)

    def apply_box_filer(self, ddepth=0, ksize=(21, 21), anchor=(-1, -1), normalize=True) -> None:
        self.image = cv2.boxFilter(self.image, ddepth, ksize, anchor=anchor,
                                   normalize=normalize, borderType=cv2.BORDER_DEFAULT)

    def apply_mask(self, mask: 'Image') -> None:
        self.image = cv2.bitwise_and(self.image, self.image, mask=mask.image)

    def mask_fill(self) -> None:
        contour, _ = cv2.findContours(
            self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(self.image, [cnt], 0, 255, -1)

    def save(self, path: Path, name: str, ext: str) -> None:
        fname = os.path.join(path, name + ext)
        cv2.imwrite(fname, self.image)

    @staticmethod
    def show(name="Image", image: 'Image' = None, scale=1.0) -> None:
        if scale != 1.0:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            dimensions = (width, height)
            image = cv2.resize(
                image, dimensions, interpolation=cv2.INTER_AREA)
        cv2.imshow(name, image)

    @staticmethod
    def save_fig(dictionary: Dict, path: Path, name: str, ext: str) -> None:
        for variant in dictionary:
            for algorithm in dictionary[variant]:
                fname = os.path.join(
                    path, f"{name}_{algorithm}_{variant}{ext}")
                dictionary[variant][algorithm].tight_layout()
                dictionary[variant][algorithm].savefig(fname)

    @staticmethod
    def save_img(dictionary: Dict, path: Path, name: str, ext: str) -> None:
        for variant in dictionary:
            for algorithm in dictionary[variant]:
                dictionary[variant][algorithm].save(
                    path, f"{name}_{variant}_{algorithm}", ext)
