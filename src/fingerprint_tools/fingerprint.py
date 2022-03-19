import os
from pathlib import Path
import cv2
import numpy as np
import pickle
import json
import math
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt
import logging
from datetime import datetime

from .contrast_types import ContrastTypes, ThresholdFlags
from .definitions import MinutuaeThreshold, RMSEThreshold, NumberOfRidgesThreshold
from .exception import FileError, ArrgumentError, UndefinedVariableError
from .image import Image
from .report import Report

from typing import Dict, Any, Union


class Fingerprint:
    def __init__(self, path, dpi):
        self.name: str = Path(path).name
        self.dpi: int = dpi
        self.block_size: int = 16

        self.raw: Image = Image(image=self.read_raw_image(path))

        # Grayscale
        self.grayscale: Image = Image(image=self.raw.image)
        self.grayscale.image_to_grayscale()

        # Applied CLAHE on grayscale
        self.grayscale_clahe = Image(self.grayscale.image)
        self.grayscale_clahe.apply_contrast(contrast_type=ContrastTypes.CLAHE)

        # self.json: Report = Report()
        self.json: Dict = {}

    def read_raw_image(self, path: Path, block_size=None) -> np.ndarray:
        """
        Reads the raw image from the given path and 
        parse them into 16 dividable shape and therefore 
        the MSU_AFIS package can read it
        """
        def image2blocks(image: np.ndarray, block_size: int) -> np.ndarray:
            """
            Adjusting image for MSU_AFIS package
            Making the image dividable
            """
            row, col, _ = image.shape
            blockR = row // block_size
            blockC = col // block_size

            return image[:blockR * block_size, :blockC * block_size]

        if block_size == None:
            block_size: int = self.block_size
        image = cv2.imread(path)
        if image is None:
            raise FileError()
        return image2blocks(image, block_size)

    def msu_afis(self, path_image: str, path_destination: str, extractor_class=None):
        from .msu_latentafis.extraction_latent import get_feature_extractor, FeatureExtraction_Latent
        if extractor_class == None:
            extractor_class: FeatureExtraction_Latent = get_feature_extractor()

        logging.info(f'Curently processing: "{self.name}"')
        extractor_class.feature_extraction_single_latent(
            img_file=path_image, output_dir=str(os.path.abspath(path_destination)), show_processes=False,
            minu_file=None, show_minutiae=True
        )

        self.common_minutiae: list = [
            extractor_class.common1,
            extractor_class.common2,
            extractor_class.common3,
            extractor_class.common4
        ]
        self.mask: Image = Image(extractor_class.mask)
        self.aec: Image = Image(extractor_class.aec)
        self.bin_image: Image = Image(extractor_class.bin_image)

        self.generate_helpers()

        logging.info('Feature extraction finished!')
        return extractor_class

    def generate_helpers(self) -> None:
        self.aec_masked = Image(cv2.bitwise_and(
            self.aec.image, self.aec.image, mask=self.mask.image))

        self.bin_image_masked = Image(cv2.bitwise_and(
            self.bin_image.image, self.bin_image.image, mask=self.mask.image))

        self.grayscale_masked = Image(cv2.bitwise_and(
            self.grayscale.image, self.grayscale.image, mask=self.mask.image))

        self.grayscale_clahe_masked = Image(cv2.bitwise_and(
            self.grayscale_clahe.image, self.grayscale_clahe.image, mask=self.mask.image))

    def grade_fingerprint(self) -> None:
        self.grade_minutiae_points()
        self.grade_contrast()
        self.grade_lines()

        cx, cy = self.get_center_cords()
        line_signal, _ = self.get_pependicular(cx, cy)

        self.grade_sinusoidal(line_signal)
        self.grade_thickness(line_signal)

    def grade_minutiae_points(self, thresh=2) -> None:
        if 0 > thresh or thresh > 3:
            raise ArrgumentError()

        if self.common_minutiae is None:
            raise UndefinedVariableError()

        number_of_cmp = len(
            np.array(self.common_minutiae[thresh], dtype=np.uint64))

        text_description = ""
        if number_of_cmp > MinutuaeThreshold.BEYOND_RESONABLE_DOUBT:
            text_description = "Enough minutiae points for identification beyond reasonable doubt"
        elif number_of_cmp <= MinutuaeThreshold.BEYOND_RESONABLE_DOUBT and number_of_cmp > MinutuaeThreshold.TWELVE_GUIDELINE:
            text_description = "Enough minutiae points for identification with possible error"
        else:
            text_description = "Not enough minutiae points for identification"

        # TODO: Add points to JSON
        self.json['minutuae_points'] = {
            "quantity": number_of_cmp,
            "description": text_description
        }

    def grade_contrast(self) -> None:
        # TODO: Improve Michaleson, Weber?, Calculate michelson with mask

        # Create mask with valleys and ridges
        mask_ridges = Image(self.bin_image_masked.image)

        mask_valleys = Image(self.bin_image.image)
        mask_valleys.invert_binary()
        mask_valleys.apply_mask(self.mask)

        # Extracted them from the image
        extracted_ridges = Image(self.grayscale_clahe.image)
        extracted_ridges.apply_mask(mask_ridges)

        extracted_valleys = Image(self.grayscale_clahe.image)
        extracted_valleys.apply_mask(mask_valleys)

        # Michelson contrast
        # Source: https://stackoverflow.com/questions/57256159/how-extract-contrast-level-of-a-photo-opencv
        kernel = np.ones((4, 4), np.uint8)
        reg_min = cv2.erode(self.grayscale_clahe.image, kernel, iterations=1)
        reg_max = cv2.dilate(self.grayscale_clahe.image, kernel, iterations=1)

        reg_min = reg_min.astype(np.float64)
        reg_max = reg_max.astype(np.float64)
        michelson_contrast = (reg_max - reg_min) / (reg_max + reg_min)
        michelson_contrast = 100 * np.mean(michelson_contrast)

        # Root Mean Square Error
        # Calculates the difference between colours of ridges and valleys
        rmse: np.float64 = np.mean(np.square(np.subtract(
            extracted_ridges.image, extracted_valleys.image)))

        rmse_description = ""

        if rmse > RMSEThreshold.VALID:
            rmse_description = "The contrast has proven that the fingerprint is valid"
        else:
            rmse_description = "The contrast has proven that the fingerprint is not valid"

        self.json['contrast'] = {
            "rmse": float(rmse),
            "michelson_contrast_pct": float(michelson_contrast),
            "description": rmse_description
        }

    def grade_lines(self, draw=True) -> None:
        # TODO: Calculate bounding box, Normalize count

        # Define new copy of image
        mask_ridges = Image(self.bin_image_masked.image)
        row, col = mask_ridges.image.shape
        sample_line_len = 3

        # Define properties for output images
        thickness = 2
        color = (255, 0, 0)
        mask_ridges_color = cv2.cvtColor(mask_ridges.image, cv2.COLOR_GRAY2RGB)
        mask_ridges_color_horizontal = mask_ridges_color.copy()
        mask_ridges_color_vertical = mask_ridges_color.copy()

        # Count number of horizontal lines
        horizontal = []
        horizontal_count = []
        horizontal_axis = []
        count = 0
        is_on_ridge = False
        for x in range(0, row, 16):
            for y in range(col):
                if y + 1 < col and mask_ridges.image[x, y] == 255 and mask_ridges.image[x, y + 1] == 0:
                    is_on_ridge = True
                if is_on_ridge and y + 1 < col and mask_ridges.image[x, y] == 0 and mask_ridges.image[x, y + 1] == 255:
                    count += 1
                    is_on_ridge = False

            if count != 0:
                if draw:
                    startpoint = (0, x)
                    endpoint = (col, x)
                    cv2.line(mask_ridges_color_horizontal, startpoint,
                             endpoint, color, thickness)
                    cv2.line(mask_ridges_color, startpoint,
                             endpoint, color, thickness)
                horizontal.append(count)
                if (len(horizontal_count) <= sample_line_len):
                    horizontal_count.append(count)
                    horizontal_axis.append(x)
                else:
                    if min(horizontal_count) < count:
                        index = horizontal_count.index(min(horizontal_count))
                        horizontal_count[index] = count
                        horizontal_axis[index] = x
                count = 0

        # Count number of vertical lines
        vertical = []
        vertical_count = []
        vertical_axis = []
        count = 0
        is_on_ridge = False
        for y in range(0, col, 16):
            for x in range(row):
                if x + 1 < row and mask_ridges.image[x, y] == 255 and mask_ridges.image[x + 1, y] == 0:
                    is_on_ridge = True
                if is_on_ridge and x + 1 < row and mask_ridges.image[x, y] == 0 and mask_ridges.image[x + 1, y] == 255:
                    count += 1
                    is_on_ridge = False
            if count != 0:
                if draw:
                    startpoint = (y, 0)
                    endpoint = (y, row)
                    cv2.line(mask_ridges_color_vertical, startpoint,
                             endpoint, color, thickness)
                    cv2.line(mask_ridges_color, startpoint,
                             endpoint, color, thickness)
                vertical.append(count)
                if (len(vertical_count) <= sample_line_len):
                    vertical_count.append(count)
                    vertical_axis.append(y)
                else:
                    if min(vertical_count) <= count:
                        index = vertical_count.index(min(vertical_count))
                        vertical_count[index] = count
                        vertical_axis[index] = y
                count = 0

        color_mark = (0, 255, 0)

        # Create the dictionary for output
        dicto = {}
        for i in range(sample_line_len):
            if draw:
                startpoint = (0, horizontal_axis[i])
                endpoint = (col, horizontal_axis[i])
                cv2.line(mask_ridges_color_horizontal, startpoint,
                         endpoint, color_mark, thickness)
                cv2.line(mask_ridges_color, startpoint,
                         endpoint, color_mark, thickness)
                startpoint = (vertical_axis[i], 0)
                endpoint = (vertical_axis[i], row)
                cv2.line(mask_ridges_color_vertical, startpoint,
                         endpoint, color_mark, thickness)
                cv2.line(mask_ridges_color, startpoint,
                         endpoint, color_mark, thickness)
            dicto[f"{vertical_axis[i]}:{horizontal_axis[i]}"] = [
                vertical_count[i], horizontal_count[i]]

        total_mean = np.mean(np.concatenate([horizontal, vertical]))
        description = ""

        if total_mean > NumberOfRidgesThreshold.EXCELENT:
            description = "Fingerprint has a great number of papillary ridges"
        elif total_mean > NumberOfRidgesThreshold.GOOD:
            description = "Fingerprint has a good amount of papillary ridges"
        elif total_mean > NumberOfRidgesThreshold.ENOUGH:
            description = "Fingerprint has enough papillary ridges for identification"
        else:
            description = "Fingerprint does not have enough papillary ridges for identification"

        self.json['papilary_ridges'] = {
            "higest_frequency": dicto,
            "vertical_mean": np.mean(vertical),
            "horizontal_mean": np.mean(horizontal),
            "total_mean": total_mean,
            "expected_core": [np.mean(vertical_axis), np.mean(horizontal_axis)],
            "description": description
        }

        # Images to generate later
        self.vertical_lines = Image(mask_ridges_color_vertical)
        self.horizontal_lines = Image(mask_ridges_color_horizontal)
        self.lines = Image(mask_ridges_color)

    def grade_sinusoidal(self, line_signal, draw=True) -> None:
        # TODO: Autoencoder, Expected core, Only one ridge

        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        def butter_lowpass(cutoff, nyq_freq, order=4):
            normal_cutoff = float(cutoff) / nyq_freq
            b, a = butter(order, normal_cutoff, btype='lowpass')
            return b, a

        def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
            b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
            y = filtfilt(b, a, data)
            return y

        # Normalize signal
        line_signal = line_signal.astype('float64')
        line_signal /= np.max(np.abs(line_signal), axis=0)

        low_pass = butter_lowpass_filter(line_signal, 5, 50/2)
        ridge_threshold = 0.7  # Threshold which

        # Count the number of computer ridges which uphold the threshold
        ridge = 0
        on_ridge = False
        for item in low_pass:
            if not on_ridge and item > ridge_threshold:
                ridge += 1
                on_ridge = True
            if on_ridge and item < ridge_threshold:
                on_ridge = False

        # Using the Difference of squares to find the best period aligement
        best_align = None
        index_best_a = 0
        period = np.arange(-np.pi, np.pi, 0.001)
        for i in range(len(period)):
            x = np.linspace(period[i], ridge * np.pi, len(line_signal))
            sin = np.abs(np.sin(x))

            diff_sqr = np.sum((sin - line_signal) ** 2)
            if i == 0:
                best_align = diff_sqr
            if diff_sqr < best_align:
                best_align = diff_sqr
                index_best_a = period[i]

        # Calculate the aligned sinus
        x = np.linspace(index_best_a, ridge * np.pi, len(line_signal))
        sin = np.abs(np.sin(x))

        if draw:
            fig_sinus: plt.Figure = plt.figure(figsize=(40, 10), dpi=100)
            plt.plot(line_signal, label="Ridges", figure=fig_sinus)
            plt.plot(sin, label="Sinus", figure=fig_sinus)
            plt.xlabel("Pixels (X)", figure=fig_sinus)
            plt.ylabel("Grayscale values (Y)", figure=fig_sinus)
            plt.legend()
            plt.close()
            self.fig_sin = fig_sinus

        # Calculate the integral under the discrete signal
        A_FP = simpson(line_signal)
        A_SIN = simpson(sin)

        D_D = (A_FP/A_SIN - 1) * 100

        self.json['papillary_crosscut']['sinusoidal_shape'] = {
            "ridges_low_pass_count": ridge,
            "sinus_offset": index_best_a,
            "A_FP": A_FP,
            "A_SIN": A_SIN,
            "D_D": D_D
        }

    def grade_thickness(self, line_signal, draw=True) -> None:
        # TODO: Autoencoder, Expected core
        average_thickness = 0.033  # mm

        # 18% defined by literature
        height_threshold = 18 / 100

        # Normalization
        line_signal = line_signal.astype('float64')
        line_signal /= np.max(np.abs(line_signal), axis=0)

        display_thickness = np.zeros(len(line_signal))
        on_ridge = False
        for i in range(len(line_signal)):
            if not on_ridge and line_signal[i] > height_threshold:
                on_ridge = True
            if on_ridge and line_signal[i] > height_threshold:
                display_thickness[i] = height_threshold
            if on_ridge and line_signal[i] < height_threshold:
                on_ridge = False
        if draw:
            fig_thickness = plt.figure(figsize=(40, 10), dpi=100)
            plt.plot(line_signal, label="Ridges", figure=fig_thickness)
            plt.plot(display_thickness, label="Thickness",
                     figure=fig_thickness)
            plt.xlabel("Pixels (X)", figure=fig_thickness)
            plt.ylabel("Grayscale values (Y)", figure=fig_thickness)
            plt.legend()
            plt.close()
            self.fig_thick = fig_thickness

        # Split the ridges into individual arrays
        ridges_separated = np.where(display_thickness != 0)[0]
        ridges_list = np.split(display_thickness[ridges_separated],
                               np.where(np.diff(ridges_separated) != 1)[0]+1)

        # Transform the pixels into readable format
        base = 2.54 / self.dpi
        result = []
        for item in ridges_list:
            Th = base * len(item)
            Dth = (Th/average_thickness - 1) * 100
            result.append(Dth)

        self.json['papillary_crosscut']['thickness'] = {
            "ridges_low_pass_count": len(result),
            "thickness_difference": result
        }

    def generate_rating(self, dirname: Path) -> Union[int, int]:
        filename = os.path.join(dirname, 'log.json')
        if not (os.path.isfile(filename) and os.access(filename, os.R_OK)):
            with open(filename, 'w') as file:
                json.dump({}, file, indent=4)

        with open(filename, 'r+') as file:
            file_data = json.load(file)
            file_data[self.name] = self.json
            file.seek(0)
            json.dump(file_data, file, indent=4)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_center_cords(self) -> Union[int, int]:
        mask_filled = Image(self.mask.image)
        clahe_grayscale = Image(self.grayscale.image)
        clahe_grayscale.apply_contrast(contrast_type=ContrastTypes.CLAHE)
        clahe_grayscale_color = Image(cv2.cvtColor(
            clahe_grayscale.image, cv2.COLOR_GRAY2RGB))

        contour, _ = cv2.findContours(
            mask_filled.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(mask_filled.image, [cnt], 0, 255, -1)

        blank = np.zeros(mask_filled.image.shape[:2], dtype='uint8')

        cv2.drawContours(blank, contour, -1, (255, 0, 0), 1)

        for i in contour:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.drawContours(clahe_grayscale_color.image, [
                                 i], -1, (255, 0, 0), 2)
                cv2.circle(clahe_grayscale_color.image,
                           (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(clahe_grayscale_color.image, "center", (cx - 20, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # print(f"x: {cx} y: {cy}")

        self.clahe_grayscale_color = clahe_grayscale_color
        self.mask_filled = Image(mask_filled.image)

        self.json['papillary_crosscut'] = {
            "mask_center": [cx, cy]
        }

        return cx, cy

    def get_pependicular(self, cx: int, cy: int, angle_base=1):

        def rotate_image(image, angle: int, cx: int, cy: int):
            row, col = image.shape
            squared_size: int = np.int(np.sqrt(row ** 2 + col ** 2))
            diff_col: int = squared_size - col
            diff_row: int = squared_size - row

            cx += diff_col // 2
            cy += diff_row // 2

            image = cv2.copyMakeBorder(image, round((diff_row) / 2), int((diff_row) / 2),
                                       round((diff_col) / 2), int((diff_col) / 2), 0)

            image_center = tuple((cx, cy))
            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(
                image, matrix, image.shape[1::-1], flags=cv2.INTER_NEAREST)
            return image, cx, cy

        def cut_image(image_mask, cropped_image):
            contours, _ = cv2.findContours(
                image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            return cropped_image[y:y+h, x:x+w], x, y

        def make_image(mask_ridges, index, angle_base, cx, cy, draw=False, image=True):
            rotated_image, cx_rot, cy_rot = rotate_image(
                mask_ridges.image, index*angle_base, cx, cy)
            image_mask, _, _ = rotate_image(
                self.mask_filled.image, index*angle_base, cx, cy)
            rotated_image, x_bb, y_bb = cut_image(
                image_mask, rotated_image)

            cx_rot -= x_bb
            cy_rot -= y_bb

            if draw:
                rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2RGB)
                cv2.circle(rotated_image,
                           (cx_rot, cy_rot), 7, (0, 0, 255), -1)

                cv2.line(rotated_image, (cx_rot, cy_rot),
                         (cx_rot, 0), (0, 0, 255), 1)

            if image:
                return rotated_image
            else:
                return rotated_image, cx_rot, cy_rot

        mask_ridges = Image(cv2.bitwise_and(
            self.bin_image.image, self.bin_image.image, mask=self.mask.image))

        candidate_results = []
        result_count = []

        for angle in range(0, 360, angle_base):
            rotated_image, cx_rot, cy_rot = make_image(
                mask_ridges, 1, angle, cx, cy, draw=False, image=False)

            # Count number of lines
            extracted_line = rotated_image[:cy_rot, cx_rot]
            count = 0
            is_on_ridge = False
            for x in range(len(extracted_line)):
                if x + 1 < len(extracted_line) and extracted_line[x] == 255 and extracted_line[x + 1] == 0:
                    is_on_ridge = True
                if is_on_ridge and x + 1 < len(extracted_line) and extracted_line[x] == 0 and extracted_line[x + 1] == 255:
                    count += 1
                    is_on_ridge = False

            # Blur the binary image and apply Sobel gradient in Y direction
            # throw white and black and compute mean, the greater the value
            # the better precision

            edge_detection = cv2.GaussianBlur(
                rotated_image, (21, 7), cv2.BORDER_DEFAULT)
            edge_detection = cv2.Sobel(edge_detection, 0, dx=0, dy=1)
            edge_detection = np.uint8(np.absolute(edge_detection))
            edge_detection = edge_detection[:cy_rot, :]

            # Throw black (0) and white (255) out
            edge_detection = [
                i for i in edge_detection[:cy_rot, cx_rot] if i not in [0, 255]]

            # Calculate the mean
            candidate_results.append(np.mean(edge_detection))
            result_count.append(count)

        max_sobel_value = max(candidate_results)
        angle = candidate_results.index(max_sobel_value)

        # Generate image which shows the progress
        self.pependicular = Image(make_image(
            mask_ridges, angle, angle_base, cx, cy, draw=True))

        # Apply the mask and rotation on the image
        clahe_grayscale_extracted = Image(cv2.bitwise_and(self.grayscale_clahe.image,
                                                          self.grayscale_clahe.image, mask=self.mask.image))

        self.clahe_grayscale_rotated = Image(make_image(
            clahe_grayscale_extracted, angle, angle_base, cx, cy, draw=False))

        extracted_grayscale_line = self.clahe_grayscale_rotated.image[:cy_rot, cx_rot]

        # Remove black pixels from array
        for i in range(len(extracted_grayscale_line)):
            if extracted_grayscale_line[i] != 0:
                extracted_grayscale_line = extracted_grayscale_line[i:]
                break

        self.json['papillary_crosscut'] = {
            "angle": angle * angle_base,
            "ridges_binary_count": result_count[angle]
        }

        return extracted_grayscale_line, count

    def generate_images(self, path, ext=".jpeg") -> None:
        Image.save(self.vertical_lines.image, path,
                   f"{self.name}_lines_vertical", ext)
        Image.save(self.horizontal_lines.image, path,
                   f"{self.name}_lines_horizontal", ext)
        Image.save(self.lines.image, path,
                   f"{self.name}_lines", ext)
        Image.save(self.clahe_grayscale_color.image, path,
                   f"{self.name}_center", ext)
        Image.save(self.pependicular.image, path,
                   f"{self.name}_pependicular", ext)
        Image.save(self.clahe_grayscale_rotated.image, path,
                   f"{self.name}_clahe_grayscale_rotated", ext)

        self.fig_sin.savefig(f"{self.name}_sin.png")
        self.fig_thick.savefig(f"{self.name}_thick.png")

    def show(self) -> None:
        # Image.show(self.raw.image, "Raw", scale=0.5)
        Image.show(self.grayscale.image, "Grayscale", scale=0.5)
        # Image.show(self.filtered.image, "Filtered", scale=0.5)
        # Image.show(self.binary.image, "Binary", scale=0.5)

        Image.show(self.mask, name="Mask", scale=0.5)
        Image.show(self.aec, "AEC", scale=0.5)
        Image.show(self.bin_image, "Bin image", scale=0.5)
