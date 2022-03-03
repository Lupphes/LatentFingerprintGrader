import os
import cv2
import numpy as np
import pickle

from .contrast_types import ContrastTypes, ThresholdFlags
from .exception import FileError
from .image import Image

from typing import Any


class Fingerprint:
    def __init__(self, path, name):
        self.name = name
        self.raw: Image = Image(image=self.read_raw_image(path))
        self.grayscale: Image = Image(
            image=self.raw.image)
        self.grayscale.image_to_grayscale()

    def read_raw_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileError()
        return image

    def grade_fingerprint(self):
        # texture = Image.apply_contrast().apply_sobel().apply_whatever()  # or split it up
        # binary = Image.calculate_binary(self.grayscale)
        # filteredBinary = Image.apply_gabor()  # optional

        self.filtered: Image = Image(self.grayscale.image)

        self.filtered.apply_contrast(ContrastTypes.CLAHE)

        # self.segmentation()

        self.filtered.apply_median(ksize=7)

        test = Image(self.filtered.image)

        # test.apply_box_filer(ksize=(41, 41))
        test.apply_gaussian()

        test.calculate_threshold(ThresholdFlags.GAUSSIAN)
        # test.invert_binary()

        self.filtered.calculate_threshold(ThresholdFlags.GAUSSIAN)
        # self.filtered.image = cv2.subtract(self.filtered.image, test.image)

        cv2.imshow("test", test.image)

        # self.filtered.calculate_mask(self.filtered.image)
        # self.filtered.apply_cartoon_texture(self.name + 'test', calculate=True)

        self.show()

    def segmentation(self):
        self.filtered.apply_cartoon_texture(
            self.name + "wContrast", calculate=False)

        self.filtered.calculate_threshold(
            ThresholdFlags.OTSU, specified_threshold=128, maxval=1)

        self.filtered.image = self.filtered.image * 255

        row, col = self.filtered.image.shape

        self.binary = Image(self.filtered.image)

        for i in range(row):
            for j in range(col):
                if i-1 > 0 and i+1 < row and self.filtered.image[i - 1, j] + self.filtered.image[i + 1, j] == 0:
                    self.binary.image[i, j] = 0
                elif j-1 > 0 and j+1 < col and self.filtered.image[i, j - 1] + self.filtered.image[i, j + 1] == 0:
                    self.binary.image[i, j] = 0
                elif i-1 > 0 and i+1 < row and j-1 > 0 and j+1 < col and self.filtered.image[i - 1, j - 1] + self.filtered.image[i + 1, j + 1] == 0:
                    self.binary.image[i, j] = 0
                elif i-1 > 0 and i+1 < row and j-1 > 0 and j+1 < col and self.filtered.image[i + 1, j - 1] + self.filtered.image[i - 1, j + 1] == 0:
                    self.binary.image[i, j] = 0
                else:
                    self.binary.image[i, j] = 1

        self.binary.image = self.binary.image * 255
        cv2.imshow("TEST", self.binary.image)
        pass

    def show(self):
        # Image.show(self.raw.image, "Raw", scale=0.5)
        Image.show(self.grayscale.image, "Grayscale", scale=0.5)
        Image.show(self.filtered.image, "Filtered", scale=0.5)
        # Image.show(self.binary.image, "Binary", scale=0.5)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
