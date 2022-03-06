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

    # Deprecated
    def filter_fingerprint(self):
        self.filtered: Image = Image(self.grayscale.image)

        # Get the texture from the library
        self.filtered.apply_cartoon_texture(self.name, calculate=False)

        # Apply histogram equalization and smooth the image
        self.filtered.apply_contrast(ContrastTypes.CLAHE)
        self.filtered.apply_median(ksize=7)

        # Create a mask which will remove some of the most noticable noise which the texture didn't catch
        test = Image(self.filtered.image)
        test.apply_gaussian()
        test.calculate_threshold(ThresholdFlags.GAUSSIAN)
        test.invert_binary()
        # cv2.imshow("test", test.image)

        self.filtered.apply_sobel_2d()
        self.filtered.calculate_threshold(ThresholdFlags.GAUSSIAN)
        self.filtered.image = cv2.subtract(self.filtered.image, test.image)
        # cv2.imshow("self.filtered", self.filtered.image)

        # lala = Image(self.filtered.image)
        # lala.invert_binary()

        self.filtered.apply_box_filer(ksize=(41, 41))

        self.filtered.calculate_threshold(ThresholdFlags.OTSU)
        self.filtered.invert_binary()
        # cv2.imshow("self.filteredBox", self.filtered.image)

        masked = cv2.bitwise_and(
            self.grayscale.image, self.grayscale.image, mask=self.filtered.image)

        self.filtered.image = masked
        # self.filtered.invert_binary()

        self.filtered.apply_median(ksize=7)
        self.filtered.apply_contrast(ContrastTypes.CLAHE)
        self.filtered.calculate_threshold(ThresholdFlags.GAUSSIAN)

        self.filtered.image = cv2.fastNlMeansDenoising(
            self.filtered.image, h=40, templateWindowSize=7, searchWindowSize=21)

        # self.filtered.calculate_mask(self.filtered.image)

        self.show()

    # Deprecated
    def texture_binarization(self):
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

    def mus_afis_segmentation(self):
        import sys
        import json
        from .msu_latentafis.descriptor_DR import template_compression_single, template_compression
        from .msu_latentafis.descriptor_PQ import encode_PQ, encode_PQ_single
        from .msu_latentafis.extraction_latent import main_single_image, parse_arguments, main

        # Parsing arguments
        args = parse_arguments(sys.argv[1:])

        # Working path
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # Loading configuration file
        with open(dir_path + '/afis.config') as config_file:
            config = json.load(config_file)

        # Setting GPUs to use
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        if args.i:  # Handling a single image

            # Setting template directory
            t_dir = args.tdir if args.tdir else config['LatentTemplateDirectory']
            template_fname = main_single_image(args.i, t_dir)

            print("Starting dimensionality reduction")
            template_compression_single(
                input_file=template_fname, output_dir=t_dir,
                model_path=config['DimensionalityReductionModel'],
                isLatent=True, config=None
            )
            print("Starting product quantization...")
            encode_PQ_single(
                input_file=template_fname,
                output_dir=t_dir, fprint_type='latent'
            )
            print("Exiting...")

        else:   # Handling a directory of images

            tdir = args.tdir if args.tdir else config['LatentTemplateDirectory']
            main(args.idir, tdir, args.edited_mnt)

            print("Starting dimensionality reduction...")
            template_compression(
                input_dir=tdir, output_dir=tdir,
                model_path=config['DimensionalityReductionModel'],
                isLatent=True, config=None
            )
            print("Starting product quantization...")
            encode_PQ(
                input_dir=tdir, output_dir=tdir, fprint_type='latent'
            )
            print("Exiting...")

    def show(self):
        # Image.show(self.raw.image, "Raw", scale=0.5)
        Image.show(self.grayscale.image, "Grayscale", scale=0.5)
        Image.show(self.filtered.image, "Filtered", scale=0.5)
        # Image.show(self.binary.image, "Binary", scale=0.5)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
