import os
import cv2
from cv2 import CV_8S
import numpy as np
import pickle
import json

from .contrast_types import ContrastTypes, ThresholdFlags
from .definitions import MinutuaeThreshold, RMSEThreshold
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
        self.json = {}

    def read_raw_image(self, path):
        def adjust_image_size(img, block_size=16):
            h, w = img.shape[:2]
            blkH = h // block_size
            blkW = w // block_size
            ovph = 0
            ovpw = 0

            img = img[ovph:ovph + blkH * block_size,
                      ovpw:ovpw + blkW * block_size]
            return img

        image = cv2.imread(path)
        if image is None:
            raise FileError()
        image = adjust_image_size(image)
        return image

    def mus_afis_segmentation(self, image_path, destination_dir, lf_latent=None):
        from .msu_latentafis.extraction_latent import get_feature_extractor
        if lf_latent == None:
            lf_latent = get_feature_extractor()

        print("Latent query: " + self.name)
        print("Starting feature extraction (single latent)...")
        l_template, _ = lf_latent.feature_extraction_single_latent(
            img_file=image_path, output_dir=str(os.path.abspath(destination_dir)), show_processes=False,
            minu_file=None, show_minutiae=True
        )

        self.common1 = lf_latent.common1
        self.common2 = lf_latent.common2
        self.common3 = lf_latent.common3
        self.common4 = lf_latent.common4
        self.virtual_des = lf_latent.virtual_des
        self.mask = Image(lf_latent.mask)
        self.aec = Image(lf_latent.aec)
        self.bin_image = Image(lf_latent.bin_image)

        print("Exiting...")
        return lf_latent

    def grade_fingerprint(self):
        self.grade_minutiae_points()
        self.grade_contrast()
        self.grade_lines()
        self.grade_thickness()
        self.grade_sinusoidal()

    def grade_minutiae_points(self):
        cm1 = len(np.array(self.common1, dtype=np.uint64))
        cm2 = len(np.array(self.common2, dtype=np.uint64))
        cm3 = len(np.array(self.common3, dtype=np.uint64))
        cm4 = len(np.array(self.common4, dtype=np.uint64))

        text_description = ""
        if cm3 > MinutuaeThreshold.BEYOND_RESONABLE_DOUBT:
            text_description = "Enough minutiae points for identification beyond reasonable doubt"
        elif cm3 <= MinutuaeThreshold.BEYOND_RESONABLE_DOUBT and cm3 > MinutuaeThreshold.TWELVE_GUIDELINE:
            text_description = "Enough minutiae points for identification with possible error"
        else:
            text_description = "Not enough minutiae points for identification"

        self.json['minutuae_points'] = {
            "quantity": cm3,
            "description": text_description
        }

    def grade_contrast(self):
        clahe_grayscale = Image(self.grayscale.image)
        clahe_grayscale.apply_contrast(contrast_type=ContrastTypes.CLAHE)

        mask_ridges = Image(cv2.bitwise_and(
            self.bin_image.image, self.bin_image.image, mask=self.mask.image))

        self.bin_image.invert_binary()

        mask_valleys = Image(cv2.bitwise_and(
            self.bin_image.image, self.bin_image.image, mask=self.mask.image))

        extracted_ridges = Image(cv2.bitwise_and(clahe_grayscale.image,
                                                 clahe_grayscale.image, mask=mask_ridges.image))

        extracted_valleys = Image(cv2.bitwise_and(clahe_grayscale.image,
                                                  clahe_grayscale.image, mask=mask_valleys.image))

        mask_ridges_wblack = clahe_grayscale.image[np.where(
            mask_ridges.image == 255)]
        mask_valleys_wblack = clahe_grayscale.image[np.where(
            mask_valleys.image == 255)]

        valleys_minimum = np.uint16(np.min(mask_valleys_wblack, axis=0))
        ridges_maximum = np.uint16(np.max(mask_ridges_wblack, axis=0))

        michelson_contrast = (ridges_maximum-valleys_minimum) / \
            (ridges_maximum+valleys_minimum)

        rmse = np.square(np.subtract(
            extracted_ridges.image, extracted_valleys.image)).mean()

        rmse_description = ""

        if rmse > RMSEThreshold.VALID:
            rmse_description = "The contrast has proven that the fingerprint is valid"
        else:
            rmse_description = "The contrast has proven that the fingerprint is not valid"

        self.json['contrast'] = {
            "rmse": str(rmse),
            "michelson_contrast": str(michelson_contrast),
            "description": rmse_description
        }

        self.bin_image.invert_binary()

    def grade_lines(self):

        # for x in range(self.bin_image.image.shape[1]):
        #     for y in self.bin_image.image[:, x]:
        #         print(y)

        count = 0
        # for x in range(int(self.bin_image.image.shape[0] / 16)):
        #     for y in range(self.bin_image.image.shape[1]):
        #         if y + 1 < self.bin_image.image.shape[1]:
        #             if self.bin_image.image[x, y] == 255 and self.bin_image.image[x, y + 1] == 0:
        #                 count += 1
        #     print(count)
        #     count = 0

        # if y == 1:  # flag -- black = 1, white = 0
        #     if self.bin_image.image[x, y] == 255:
        #         count = 0
        #     else:
        #         count = 1

    def grade_thickness(self):
        pass

    def grade_sinusoidal(self):
        pass

    def generate_rating(self, dirname):
        filename = os.path.join(dirname, 'log.json')
        if not (os.path.isfile(filename) and os.access(filename, os.R_OK)):
            with open(filename, 'w') as file:
                json.dump({}, file, indent=4)

        with open(filename, 'r+') as file:
            file_data = json.load(file)
            file_data[self.name] = self.json
            file.seek(0)
            json.dump(file_data, file, indent=4)

    def show(self):
        # Image.show(self.raw.image, "Raw", scale=0.5)
        Image.show(self.grayscale.image, "Grayscale", scale=0.5)
        # Image.show(self.filtered.image, "Filtered", scale=0.5)
        # Image.show(self.binary.image, "Binary", scale=0.5)

        print("Test data:\n")
        Image.show(self.mask, "Mask", scale=0.5)
        Image.show(self.aec, "AEC", scale=0.5)
        Image.show(self.bin_image, "Bin image", scale=0.5)
        # print(lf_latent.minu_model)
        # print(lf_latent.minutiae_sets)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
