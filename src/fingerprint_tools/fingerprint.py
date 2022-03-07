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

    def mus_afis_segmentation(self, image_path, destination_dir):
        import json
        from .msu_latentafis.extraction_latent import main_single_image, get_feature_extractor
        # main_single_image(self.path, t_dir)

        lf_latent = get_feature_extractor()

        print("Latent query: " + self.name)
        print("Starting feature extraction (single latent)...")
        l_template, _ = lf_latent.feature_extraction_single_latent(
            image_path, output_dir=destination_dir, show_processes=False,
            minu_file=None, show_minutiae=False
        )

        print("Exiting...")

        print("Test data:\n")
        print(lf_latent.minu_model)
        print(lf_latent.minutiae_sets)

    def grade_fingerprint(self):
        pass

    def generate_rating(self):
        pass

    def show(self):
        # Image.show(self.raw.image, "Raw", scale=0.5)
        Image.show(self.grayscale.image, "Grayscale", scale=0.5)
        Image.show(self.filtered.image, "Filtered", scale=0.5)
        # Image.show(self.binary.image, "Binary", scale=0.5)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
