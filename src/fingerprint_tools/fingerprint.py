import os
import cv2
import numpy as np
import pickle

from .contrast_types import ContrastTypes, ThresholdFlags
from .exception import FileError
from .image import Image

from typing import Any


class Fingerprint:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self.raw: Image = Image(image=self.read_raw_image(path))
        self.grayscale: Image = Image(
            image=self.raw.image)
        self.grayscale.image_to_grayscale()

    def read_raw_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileError()
        return image

    def mus_afis_segmentation(self, image_path, destination_dir, lf_latent=None):
        from .msu_latentafis.extraction_latent import main_single_image, get_feature_extractor
        if lf_latent == None:
            lf_latent = get_feature_extractor()

        print("Latent query: " + self.name)
        print("Starting feature extraction (single latent)...")
        l_template, _ = lf_latent.feature_extraction_single_latent(
            img_file=image_path, output_dir=str(os.path.abspath(destination_dir)), show_processes=False,
            minu_file=None, show_minutiae=False
        )

        self.mask = lf_latent.mask
        self.aec = lf_latent.aec
        self.bin_image = lf_latent.bin_image

        print("Exiting...")
        return lf_latent

    def grade_fingerprint(self):
        self.grade_minutiae_points()
        self.grade_contrast()
        self.grade_lines()
        self.grade_thickness()
        self.grade_sinusoidal()

    def grade_minutiae_points(self):
        pass

    def grade_contrast(self):
        pass

    def grade_lines(self):
        pass

    def grade_thickness(self):
        pass

    def grade_sinusoidal(self):
        pass

    def generate_rating(self):
        self.show()

    def show(self):
        # Image.show(self.raw.image, "Raw", scale=0.5)
        Image.show(self.grayscale.image, "Grayscale", scale=0.5)
        # Image.show(self.filtered.image, "Filtered", scale=0.5)
        # Image.show(self.binary.image, "Binary", scale=0.5)

        print("Test data:\n")
        Image.show(self.mask, scale=0.5)
        Image.show(self.aec, scale=0.5)
        Image.show(self.bin_image, scale=0.5)
        # print(lf_latent.minu_model)
        # print(lf_latent.minutiae_sets)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
