from curses import window
from statistics import variance
import cv2
from cv2 import sqrt
import numpy as np
import os
import pywt

from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter, median_filter

from scipy.signal import medfilt, medfilt2d
from scipy.special import logsumexp

from .contrast_types import ContrastTypes
from .exception import FileError
from skimage.filters import threshold_otsu


class Fingerprint:
    def __init__(self, path):
        self.image = self.fingerprintCheck(path)
        self.image_gray = self.image2grayscale()
        # self.image_gray_cont = Fingerprint.computeContrast(
        #     self.image_gray,
        #     type=ContrastTypes.CLAHE
        # )

        # self.mask, self.masked_image = Fingerprint.cannyMorphology(
        #     self.image_gray_cont
        # )

        # self.image_norm = Fingerprint.normalization(self.image_gray)

        self.image_mask = Fingerprint.generateTexture(self.image_gray)

    def fingerprintCheck(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileError()
        return image

    def image2grayscale(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('IMG2GRAY', image)
        return image

    def grade_fingerprint(self):
        # Fingerprint.show(
        #     self.image,
        #     name='Given fingerprint', scale=1.2)

        # Fingerprint.show(
        #     self.image_norm,
        #     name='Normalized', scale=1.8)

        # Fingerprint.computeContrast(type=ContrastTypes.MICHELSON)
        # Fingerprint.computeContrast(type=ContrastTypes.RMS)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pass

    @staticmethod
    def computeContrast(image, type=ContrastTypes.CLAHE):
        result = None

        if type == ContrastTypes.CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(image)
            cv2.imshow('CLAHE contrast', result)

        elif type == ContrastTypes.MICHELSON:
            min = np.uint16(np.min(image))
            max = np.uint16(np.max(image))

            # Contrast is (0,1)
            contrast = (max-min) / (max+min)
            # print('MICHELSON contrast: ', min, max, contrast)
            result = cv2.convertScaleAbs(image, alpha=contrast)
            cv2.imshow('MICHELSON contrast', result)

        elif type == ContrastTypes.WEBER:
            pass

        elif type == ContrastTypes.RMS:
            # Not normalized image
            # contrast = image2grayscale().std()
            # result = cv2.convertScaleAbs(image, alpha=contrast)
            # cv2.imshow('RMS contrast', result)
            pass

        else:
            raise NotImplementedError(
                "This part of program was not implemented yet. Wait for updates")

        return result

    @staticmethod
    def cannyMorphology(image):
        img_edge = cv2.Canny(image, 100, 200)
        cv2.imshow('Canny', img_edge)

        # Morphology
        kernel = np.ones((25, 25), np.uint8)
        img_morph = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('Morphology', img_morph)

        # Contours
        contours, _ = cv2.findContours(
            img_morph.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Sort Contors by area -> remove the largest frame contour
        n = len(contours) - 1
        contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

        copy = image.copy()

        # Just for testing
        # Iterate through contours and draw the convex hull
        thres = 750
        for c in contours:
            if cv2.contourArea(c) < thres:
                continue
            hull = cv2.convexHull(c)
            cv2.drawContours(copy, [hull], 0, (0, 255, 0), 2)

        cv2.imshow('Convex Hull', copy)

        return img_morph, None

    @staticmethod
    def show(image, name="Image", scale=1.0):
        if scale != 1.0:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            dimensions = (width, height)
            image = cv2.resize(
                image, dimensions, interpolation=cv2.INTER_AREA)

        cv2.imshow(name, image)

    @staticmethod
    def segmentation(image):
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

        # A 5 × 5 block median filter is applied to smooth the response.
        # sOT = cv2.medianBlur(sOT, 5)

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

                mean = np.mean(array_sot)
                thresh = Fingerprint.threshold_otsu_impl(array_sot)
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

        cv2.imshow('mask', mask)

    @staticmethod
    def normalization(image, M_0=0.9, VAR_0=0.7):

        row, col = image.shape

        M, _ = cv2.meanStdDev(image)
        VAR = np.var(image)
        G = np.zeros(image.shape)

        for i in range(row):
            for j in range(col):
                if image[i, j] > M:
                    G[i, j] = M_0 + \
                        np.sqrt((VAR_0 * (image[i, j] - M) ** 2) / VAR)
                else:
                    G[i, j] = M_0 - \
                        np.sqrt((VAR_0 * (image[i, j] - M) ** 2) / VAR)

        print(f'Mean: {M}, Variance: {VAR}')

        return G

    @staticmethod
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

    @staticmethod
    def segmentation4529652(image):

        row, col = image.shape
        f = image

        sigma = 5  # scale parameter

        from scipy.ndimage import gaussian_filter1d

        # Gsig = gaussian_filter1d(f, sigma)
        # Gsig = [i[0] for i in Gsig]

        n = int(4 * np.ceil(sigma) + 1)

        u = cv2.getGaussianKernel(n, sigma, cv2.CV_64F)
        u = [i[0] for i in u]
        # print(f"size: {n}")
        # print(u)

        # iteration_n = 5

        # low = np.convolve(Gsig[0], f, mode='same')
        # high = f - low

        # for i in range(iteration_n):
        #     high = high - np.convolve(Gsig, high, mode='same')
        #     low = f - high

        # # filtered = np.zeros(image.shape)

        # cv2.imshow('high', high)
        # cv2.imshow('low', low)

        pass

    @staticmethod
    def generateTexture(image):

        height, width = image.shape
        size = height * width
        sigma = 5  # scale parameter

        ksize = int(4 * np.ceil(sigma) + 1)

        kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_64F)
        kernel = [i[0] for i in kernel]

        tpGrad = np.zeros((size), dtype=np.float64)

        iHeight, iWidth = image.shape

        tpI = np.array([j for sub in image for j in sub], dtype=np.float64)

        print("width", iWidth)
        print("height", iHeight)

        for ih in range(1, iHeight - 1):
            for iw in range(1, iWidth - 1):
                xgrad = tpI[ih * iWidth + iw + 1] - tpI[ih * iWidth + iw - 1]
                ygrad = tpI[(ih-1) * iWidth + iw] - tpI[(ih+1) * iWidth + iw]

                tpGrad[ih * iWidth +
                       iw] = np.sqrt(xgrad * xgrad + ygrad * ygrad)

        # -------------------------------------------

        ratio1 = Fingerprint.fiSepConvol(tpGrad, iWidth, iHeight,
                                         kernel, ksize, kernel, ksize)

        niter = 5
        gconvolved = Fingerprint.low_pass_filter(
            tpI, sigma, niter, width, height)

        # # -------------------------------------------

        tpGrad2 = np.zeros((size), dtype=np.float64)

        for ih in range(1, iHeight - 1):
            for iw in range(1, iWidth - 1):
                xgrad = gconvolved[ih * iWidth + iw + 1] - \
                    gconvolved[ih * iWidth + iw - 1]
                ygrad = gconvolved[(ih-1) * iWidth + iw] - \
                    gconvolved[(ih+1) * iWidth + iw]

                tpGrad2[ih * iWidth +
                        iw] = np.sqrt(xgrad * xgrad + ygrad * ygrad)

        ratio2 = Fingerprint.fiSepConvol(
            tpGrad2, width, height, kernel, ksize, kernel, ksize)

        out = np.zeros((size), dtype=np.float64)

        for i in range(size):
            weight = Fingerprint.WeightingFunction(ratio1[i], ratio2[i])
            out[i] = weight * gconvolved[i] + (1.0 - weight) * tpI[i]

        cartoon_v = out
        vLim = 20
        dif_v = np.zeros((size), dtype=np.float64)

        for ii in range(size):
            fValue = tpI[ii] - cartoon_v[ii]
            fValue = (fValue + vLim) * 255.0 / (2.0 * vLim)

            if fValue < 0.0:
                fValue = 0.0
            if fValue > 255.0:
                fValue = 255.0

            dif_v[ii] = fValue

        print(f"1: {cartoon_v[0]}")
        print(f"2: {cartoon_v[1000]}")
        print(f"3: {cartoon_v[2000]}")
        print(f"4: {cartoon_v[3000]}")
        print(f"5: {cartoon_v[4000]}")
        print(f"6: {cartoon_v[5000]}")

        print(f"1: {dif_v[0]}")
        print(f"2: {dif_v[1000]}")
        print(f"3: {dif_v[2000]}")
        print(f"4: {dif_v[3000]}")
        print(f"5: {dif_v[4000]}")
        print(f"6: {dif_v[5000]}")
        print(f"7: {dif_v[6000]}")
        print(f"8: {dif_v[7000]}")
        print(f"9: {dif_v[8000]}")
        print(f"10: {dif_v[9000]}")

        cartoon_v -= cartoon_v.min()
        cartoon_v /= cartoon_v.max()

        dif_v -= dif_v.min()
        dif_v /= dif_v.max()

        cv2.imshow("cartoon_v", np.asarray(np.around(cartoon_v * 255), dtype=np.uint8).reshape(
            iHeight, iWidth))
        cv2.imshow("texture", np.asarray(np.around(dif_v * 255),
                   dtype=np.uint8).reshape(iHeight, iWidth))

        # cv2.imshow("cartoon_v", cartoon_v.reshape(iHeight, iWidth))
        # cv2.imshow("texture", dif_v.reshape(iHeight, iWidth))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def WeightingFunction(r1, r2):
        difference = r1 - r2

        ar1 = np.abs(r1, dtype=np.float64)
        if ar1 > 1.0:
            difference /= ar1
        else:
            difference = 0.0

        weight = None

        cmin = 0.25
        cmax = 0.50

        if difference < cmin:
            weight = 0.0
        elif difference > cmax:
            weight = 1.0
        else:
            weight = (difference - cmin) / (cmax - cmin)

        return weight

    @staticmethod
    def low_pass_filter(input, sigma, niter, width, height):
        size = width*height
        ksize = int(4 * np.ceil(sigma) + 1)

        kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_64F)
        kernel = [i[0] for i in kernel]

        imconvolved = np.zeros((size), dtype=np.float64)
        imdifference = np.zeros((size), dtype=np.float64)

        out = Fingerprint.fiSepConvol(
            input, width, height, kernel, ksize, kernel, ksize)

        imdifference = Fingerprint.fpCombine(
            input, 1.0, out, -1.0, width*height)

        for i in range(niter):
            imconvolved = Fingerprint.fiSepConvol(imdifference, width, height,
                                                  kernel, ksize, kernel, ksize)

            imdifference = Fingerprint.fpCombine(
                imdifference, 1.0, imconvolved, -1.0, width*height)

        out = Fingerprint.fpCombine(input, 1.0, imdifference, -1.0, size)

        return out

    @staticmethod
    def fiFloatHorizontalConvolution(u, width, height, kernel, ksize):
        v = np.zeros((width*height), dtype=np.float64)

        halfsize = int(ksize / 2)
        buffersize = width + ksize

        buffer = np.zeros((buffersize), dtype=np.float64)

        for r in range(height):
            l = r*width
            for i in range(halfsize):
                buffer[i] = u[l + halfsize - 1 - i]

            for i in range(width):
                buffer[halfsize + i] = u[l + i]

            for i in range(halfsize):
                buffer[i + width + halfsize] = u[l + width - 1 - i]

            buffer = Fingerprint.fiFloatBufferConvolution(
                buffer, kernel, width, ksize)

            for c in range(width):
                v[r*width+c] = buffer[c]

        return v

    @staticmethod
    def fiFloatVerticalVonvolution(u, width, height, kernel, ksize):
        v = np.zeros((width*height), dtype=np.float64)

        halfsize = int(ksize / 2)
        buffersize = height + ksize

        buffer = np.zeros((buffersize), dtype=np.float64)

        for c in range(width):

            for i in range(halfsize):
                buffer[i] = u[(halfsize-i-1)*width + c]

            for i in range(height):
                buffer[halfsize + i] = u[i*width + c]

            for i in range(halfsize):
                buffer[halfsize + height + i] = u[(height - i - 1)*width+c]

            buffer = Fingerprint.fiFloatBufferConvolution(
                buffer, kernel, height, ksize)

            for r in range(height):
                v[r*width+c] = buffer[r]

        return v

    @staticmethod
    def fiFloatBufferConvolution(buffer, kernel, size, ksize):
        for i in range(size):
            sum = 0.0
            k = 0

            for j in range(0, ksize - 4, 5):
                sum += buffer[i + 0 + j] * kernel[0 + j] + buffer[i + 1 + j] * kernel[1 + j] + buffer[i + 2 + j] * \
                    kernel[2 + j] + buffer[i + 3 + j] * \
                    kernel[3 + j] + buffer[i + 4 + j] * kernel[4 + j]
                k += j

            # for j in range(k, ksize, 5):
            #     sum += buffer[i + j] * (kernel[j])

            buffer[i] = sum

        return buffer

    @staticmethod
    def fiSepConvol(v, width, height, xkernel, xksize, ykernel, yksize):
        v = Fingerprint.fiFloatHorizontalConvolution(
            v, width, height, xkernel, xksize)

        v = Fingerprint.fiFloatVerticalVonvolution(
            v, width, height, ykernel, yksize)

        return v

    @staticmethod
    def fpCombine(u, a, v, b, size):
        w = np.zeros((size), dtype=np.float64)
        for i in range(size):
            w[i] = (a * u[i] + b * v[i])

        return w
