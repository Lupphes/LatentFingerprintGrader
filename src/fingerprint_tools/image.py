import os
import cv2
import numpy as np

from .contrast_types import ContrastTypes, ThresholdFlags
from .cartext import Cartext


from typing import Any

from scipy.signal import medfilt2d, medfilt


class Image:
    def __init__(self, image):
        self.image: Any = image

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    def image_to_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def calculate_histogram(self):
        return cv2.calcHist(self.image, [0], None, [256], [0, 256])

    def calculate_threshold(self, threshold_type=ThresholdFlags.OTSU, specified_threshold=128, maxval=255):
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

    def invert_binary(self):
        self.image = cv2.bitwise_not(self.image)

    def apply_contrast(self, contrast_type=ContrastTypes.CLAHE):
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
            pass
        else:
            raise NotImplementedError(
                "This part of program was not implemented yet. Wait for updates")

    def apply_sobel_2d(self):
        # The Sobel kernels can also be thought of as 3×3 approximations to first-derivative-of-Gaussian kernels.
        Dxf = cv2.Sobel(self.image, cv2.CV_8UC1, 1, 0, ksize=3)  # X
        Dyf = cv2.Sobel(self.image, cv2.CV_8UC1, 0, 1, ksize=3)  # Y

        self.image = cv2.Sobel(Dxf, cv2.CV_8UC1, 0, 1, ksize=3)

        return Dxf, Dyf

    def apply_gabor(self, ksize=31, sigma=1, theta=0, lambd=1.0, gamma=0.02, psi=0):
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
        self.image = cv2.filter2D(
            self.image, cv2.CV_64F, kernel)

    def apply_median(self, ksize=3):
        self.image = cv2.medianBlur(self.image, ksize=ksize)

    def apply_gaussian(self, ksize=23, sigma=8):
        # kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_64F)
        # kernel = [i[0] for i in kernel]
        # self.image = np.apply_along_axis(
        #     lambda x: np.convolve(x, kernel, mode='same'), 0, self.image)
        # self.image = np.apply_along_axis(
        #     lambda x: np.convolve(x, kernel, mode='same'), 1, self.image)

        self.image = cv2.GaussianBlur(
            self.image, (ksize, ksize), cv2.BORDER_DEFAULT)

        # gaussian_filter(hn, sigma=8, truncate=1.5) Performed the same
        # cv2.filter2D(self.image, cv2.CV_64F, kernel) Let's try this
        pass

    def apply_box_filer(self, ddepth=0, ksize=(21, 21), anchor=(-1, -1), normalize=True):
        self.image = cv2.boxFilter(self.image, ddepth, ksize, anchor=anchor,
                                   normalize=normalize, borderType=cv2.BORDER_DEFAULT)

    def apply_cartoon_texture(self, name, calculate=True):

        if not calculate:
            cartoon, texture = Cartext.loadTexture(name, self.image)
        else:
            cartoon, texture = Cartext.generateTexture(
                name, self.image, sigma=7)
        self.image = texture

        return cartoon, texture

    def calculate_mask_by_paper1(self, image):
        row, col = image.shape

        # ----------------------------------------------------------
        # Step 1

        # z = (Dxf + iDyf )^2 -- orientation tensor
        # Dxf, Dyf -- derivates of the image

        # https://dsp.stackexchange.com/questions/19175/sobel-vs-gaussian-derivative
        # The Sobel kernels can also be thought of as 3×3 approximations to first-derivative-of-Gaussian kernels.

        Dxf = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        cv2.imshow('Dxf', Dxf)

        Dyf = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        cv2.imshow('Dyf', Dyf)

        z = (Dxf + (Dyf * 1j)) ** 2

        # ----------------------------------------------------------
        # Step 2

        #  h_n = (x + iy)^n*g, f or n ≥ 0,
        #  h_n = (x − iy)^|n|*g, f or n < 0,
        # where g denotes a 2D Gaussian function (23 × 23 Gaussian kernel with σ = 8)

        sn = np.zeros(image.shape).astype(complex)
        hn = np.zeros(image.shape).astype(complex)

        filters_n = []

        for n in range(-2, 3):
            for x in range(row):
                for y in range(col):
                    if n >= 0:
                        hn[x, y] = ((x + (y * 1j)) ** n)

                    else:
                        hn[x, y] = ((x - (y * 1j)) ** np.absolute(n))

            # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
            # hn = gaussian_filter(hn, sigma=8, truncate=1.5)

            # https://stackoverflow.com/questions/29920114/how-to-gauss-filter-blur-a-floating-point-numpy-array
            kernel_size = 23
            sigma = 8
            kernel = cv2.getGaussianKernel(kernel_size, sigma, cv2.CV_64F)
            kernel = [i[0] for i in kernel]
            hn = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='same'), 0, hn)
            hn = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='same'), 1, hn)

            # s_n = < z, hn > / < |z|, |hn| >

            for x in range(row):
                for y in range(col):
                    if np.vdot(np.absolute(z[x, y]), np.absolute(hn[x, y])) == 0:
                        sn[x, y] = 0
                    else:
                        sn[x, y] = np.divide(np.vdot(z[x, y], hn[x, y]),
                                             np.vdot(np.absolute(z[x, y]), np.absolute(hn[x, y])))

            filters_n.append(sn)

        # sOT = s0 · ∏_k (1 − sk), k ∈ {−2, −1, 1, 2}.

        sOT = filters_n[2] * ((1 - filters_n[0]) *
                              (1 - filters_n[1]) * (1 - filters_n[3]) * (1 - filters_n[4]))

        # ----------------------------------------------------------
        # Step 3

        # divide the orientation response image, sOT , into non-overlapping blocks of size 16 × 16 pixels
        window_column = 16
        window_row = 16

        mask = np.zeros(image.shape)
        mask_real = np.zeros(image.shape)
        mask_imag = np.zeros(image.shape)

        for r in range(0, row - window_row, window_row):
            for c in range(0, col - window_column, window_column):
                array_sot = sOT[r:r+window_row, c:c+window_column]

                # array_sot = medfilt(array_sot, kernel_size=5)
                array_sot = medfilt2d(np.real(array_sot), kernel_size=5) + \
                    1j*medfilt2d(np.imag(array_sot), kernel_size=5)

                # A 5 × 5 block median filter is applied to smooth the response.
                # array_sot = cv2.medianBlur(
                #     np.real(array_sot), ksize=5) + cv2.medianBlur(np.imag(array_sot), ksize=5) * 1j

                mean = np.mean(array_sot)

                def threshold_otsu_impl(image, nbins=0.1):
                    # validate multicolored
                    if np.min(image) == np.max(image):
                        print(
                            f"the image must have multiple colors - {np.min(image)} : {np.max(image)}")
                        return np.min(image)

                    all_colors = image.flatten()
                    total_weight = len(all_colors)
                    least_variance = -1
                    least_variance_threshold = -1

                    # create an array of all possible threshold values which we want to loop through
                    color_thresholds = np.arange(
                        np.min(image)+nbins, np.max(image)-nbins, nbins)

                    # loop through the thresholds to find the one with the least within class variance
                    for color_threshold in color_thresholds:
                        bg_pixels = all_colors[all_colors < color_threshold]
                        weight_bg = len(bg_pixels) / total_weight
                        variance_bg = np.var(bg_pixels)

                        fg_pixels = all_colors[all_colors >= color_threshold]
                        weight_fg = len(fg_pixels) / total_weight
                        variance_fg = np.var(fg_pixels)

                        within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
                        if least_variance == -1 or least_variance > within_class_variance:
                            least_variance = within_class_variance
                            least_variance_threshold = color_threshold
                        print("trace:", within_class_variance, color_threshold)

                    return least_variance_threshold
                thresh = threshold_otsu_impl(array_sot)

                # thresh = pywt.threshold(array_sot, value=0.5, mode='soft')

                # otsu_threshold_real, image_result_real = cv2.threshold(
                #     np.real(array_sot).astype("uint16"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # otsu_threshold_im, image_result_im = cv2.threshold(
                #     np.imag(array_sot).astype("uint16"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # thresh = otsu_threshold_real + (1j * otsu_threshold_im)

                # mask_real[r:r+window_row, c:c +
                #           window_column] = image_result_real

                # mask_imag[r:r+window_row, c:c +
                #           window_column] = image_result_im

                if mean > thresh:
                    mask[r:r+window_row, c:c+window_column] = 1
                else:
                    mask[r:r+window_row, c:c+window_column] = 0

        # cv2.imshow('mask_real', mask_real)
        # cv2.imshow('mask_imag', mask_imag)

        self.image = mask

    @staticmethod
    def show(image, name="Image", scale=1.0):
        if scale != 1.0:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            dimensions = (width, height)
            image = cv2.resize(
                image, dimensions, interpolation=cv2.INTER_AREA)
        cv2.imshow(name, image)
