import cv2
import numpy as np
import os

from .contrast_types import ContrastTypes
from .exception import FileError


class Fingerprint:
    def __init__(self, path):
        self.image = self.fingerprintCheck(path)
        self.image_gray = self.image2grayscale()
        self.image_gray_cont = Fingerprint.computeContrast(
            self.image_gray,
            type=ContrastTypes.CLAHE
        )

        self.mask, self.masked_image = Fingerprint.segmentation(
            self.image_gray_cont
        )

    def fingerprintCheck(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileError()
        return image

    def image2grayscale(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('IMG2GRAY', image)
        return image

    @staticmethod
    def grade_fingerprints():

        image_path = 'img/012_3_1.png'
        fingerprint_image = Fingerprint(path=image_path)

        Fingerprint.show(
            fingerprint_image.image,
            name='Given fingerprint', scale=1.2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Fingerprint.computeContrast(type=ContrastTypes.MICHELSON)
        # Fingerprint.computeContrast(type=ContrastTypes.RMS)

        print("I'm here all alone")

    @staticmethod
    def computeContrast(image, type):
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
    def segmentation(image):
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
