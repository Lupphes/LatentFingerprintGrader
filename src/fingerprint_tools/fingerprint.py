import os
import cv2
import logging
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt


from .contrast_types import ContrastTypes, ThresholdFlags
from .exception import FileError, ArrgumentError, UndefinedVariableError
from .image import Image
from .report import Report

from typing import Dict, Union, Tuple


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

        self.clahe_grayscale_color = Image(cv2.cvtColor(
            self.grayscale_clahe.image, cv2.COLOR_GRAY2RGB))

        self.report: Report = Report()

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

        self.mask_filled = Image(self.mask.image.copy())
        self.mask_filled.mask_fill()

        self.bin_image_masked_filled = Image(cv2.bitwise_and(
            self.bin_image.image, self.bin_image.image, mask=self.mask_filled.image))

        self.figure_dict = {}

    def grade_fingerprint(self) -> None:
        self.grade_minutiae_points()
        self.grade_contrast()
        self.grade_lines()

        cx, cy = self.get_center_cords()
        gray_signal, aec_signal = self.get_pependicular(cx, cy)

        self.grade_sinusoidal(gray_signal, 'gray')
        self.grade_thickness(gray_signal, 'gray')

        self.grade_sinusoidal(aec_signal, 'aec')
        self.grade_thickness(aec_signal, 'aec')

    def grade_minutiae_points(self, thresh=2) -> None:
        if 0 > thresh or thresh > 3:
            raise ArrgumentError()

        if self.common_minutiae is None:
            raise UndefinedVariableError()

        minuatue_points = np.array(
            self.common_minutiae[thresh], dtype=np.uint64)
        number_of_cmp = len(minuatue_points)

        self.report.report_minuatue(number_of_cmp, minuatue_points)

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

        self.report.report_contrast(rmse, michelson_contrast)

    def remove_black_array(self, array, black_threshold=10) -> np.ndarray:
        # Remove masked parts
        processed_extracted = []

        index = -1
        black_count = 0
        for i in range(len(array)):
            if array[i] == 0:
                black_count += 1
                if black_count == black_threshold:
                    index = 10
                elif black_count > black_threshold:
                    index += 1
            else:
                if index != -1:
                    processed_extracted = processed_extracted[:-index]
                    index = -1
                    black_count = 0
                else:
                    black_count = 0
            processed_extracted.append(array[i])

        array = np.array(processed_extracted)

        return array

    def grade_lines(self, draw=True) -> None:
        # TODO: Normalize count for length -> pixels

        # Define new copy of image
        mask_ridges, offset_x, offset_y = self.cut_image(
            self.mask_filled, self.bin_image_masked)
        row, col = mask_ridges.image.shape
        sample_line_len: int = 3

        # Define properties for output images
        thickness: int = 2
        color: Tuple[int, int, int] = (255, 0, 0)
        mask_ridges_color: np.ndarray = cv2.cvtColor(
            mask_ridges.image, cv2.COLOR_GRAY2RGB)
        mask_ridges_color_horizontal: np.ndarray = mask_ridges_color.copy()
        mask_ridges_color_vertical: np.ndarray = mask_ridges_color.copy()

        mask_ridges_color_uncut: np.ndarray = cv2.cvtColor(
            self.bin_image_masked.image, cv2.COLOR_GRAY2RGB)

        # Count number of horizontal lines
        horizontal: list = []
        horizontal_count: list = []
        horizontal_axis: list = []
        count: int = 0
        is_on_ridge = False
        for x in range(0, row, self.block_size):
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
        vertical: list = []
        vertical_count: list = []
        vertical_axis: list = []
        count: int = 0
        is_on_ridge: bool = False
        for y in range(0, col, self.block_size):
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
        dicto: Dict = {}
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
            dicto[f"{vertical_axis[i]+ offset_x}:{horizontal_axis[i]+ offset_y}"] = [
                vertical_count[i], horizontal_count[i]]

        expected_core_x: int = (np.int(np.mean(vertical_axis)))
        expected_core_y: int = (np.int(np.mean(horizontal_axis)))

        expected_core_x_off: int = expected_core_x + offset_x
        expected_core_y_off: int = expected_core_y + offset_y

        if draw:
            cv2.circle(mask_ridges_color, (expected_core_x,
                                           expected_core_y), 7, (0, 0, 255), -1)
            cv2.putText(mask_ridges_color, "center",
                        (expected_core_x - 45, expected_core_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.circle(mask_ridges_color_uncut, (expected_core_x_off,
                                                 expected_core_y_off), 7, (0, 0, 255), -1)
            cv2.putText(mask_ridges_color_uncut, "center",
                        (expected_core_x_off - 45, expected_core_y_off - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        self.report.report_lines(horizontal, vertical,
                                 vertical_axis, horizontal_axis, dicto, expected_core_x_off, expected_core_y_off)
        # Images to generate later
        self.vertical_lines = Image(mask_ridges_color_vertical)
        self.horizontal_lines = Image(mask_ridges_color_horizontal)
        self.lines = Image(mask_ridges_color)
        self.mask_ridges_color_uncut = Image(mask_ridges_color_uncut)

    def grade_sinusoidal(self, line_signal: np.ndarray, name: str, draw=True) -> None:
        # TODO: Expected core, Only one ridge

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
        ridge_threshold = 0.6  # Threshold which

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
            if not name in self.figure_dict:
                self.figure_dict[name] = {}
            self.figure_dict[name]['sin'] = fig_sinus

        # Calculate the integral under the discrete signal
        A_FP = simpson(line_signal)
        A_SIN = simpson(sin)

        D_D = (A_FP/A_SIN - 1) * 100

        self.report.report_sinusoidal(
            ridge, index_best_a, A_FP, A_SIN, D_D, name)

    def grade_thickness(self, line_signal: np.ndarray, name: str, draw=True) -> None:
        # TODO: Expected core
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
            if not name in self.figure_dict:
                self.figure_dict[name] = {}
            self.figure_dict[name]['thick'] = fig_thickness

        # Split the ridges into individual arrays
        ridges_separated = np.where(display_thickness != 0)[0]
        ridges_list = np.split(display_thickness[ridges_separated],
                               np.where(np.diff(ridges_separated) != 1)[0]+1)

        # Transform the pixels into readable format
        base = 2.54 / self.dpi
        ridge_thickness = []
        for item in ridges_list:
            Th = base * len(item)
            Dth = (Th/average_thickness - 1) * 100
            ridge_thickness.append(Dth)

        self.report.report_thickness(ridge_thickness, name)

    def generate_rating(self, dirname: Path, name='log.json') -> Union[int, int]:
        self.report.generate_report(dirname, name, self.name)

    def cut_image(self, image_mask: Image, input_image: Image) -> Union[Image, int, int]:
        contours, _ = cv2.findContours(
            image_mask.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return Image(input_image.image[y:y+h, x:x+w]), x, y

    def get_center_cords(self, draw=True) -> Union[int, int]:
        # Find countours and therfore draw the center
        contour, _ = cv2.findContours(
            self.mask_filled.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if draw:
            blank = np.zeros(self.mask_filled.image.shape[:2], dtype='uint8')
            cv2.drawContours(blank, contour, -1, (255, 0, 0), 1)

        # Calculate the center of an image
        for i in contour:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                if draw:
                    clahe_grayscale_center = Image(
                        self.clahe_grayscale_color.image)
                    cv2.drawContours(clahe_grayscale_center.image, [
                        i], -1, (255, 0, 0), 2)
                    cv2.circle(clahe_grayscale_center.image,
                               (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(clahe_grayscale_center.image, "center", (cx - 45, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.clahe_grayscale_center = clahe_grayscale_center

        self.report.report_center_cords(cx, cy)

        return cx, cy

    def get_pependicular(self, cx: int, cy: int, angle_base=1):

        def rotate_image(image: Image, angle: int, center_col: int, center_row: int) -> Tuple[Image, int, int]:
            image = image.image
            row, col = image.shape
            squared_size: int = np.int(np.sqrt(row ** 2 + col ** 2))
            diff_col: int = squared_size - col
            diff_row: int = squared_size - row

            center_col += diff_col // 2
            center_row += diff_row // 2

            image = cv2.copyMakeBorder(image, round((diff_row) / 2), int((diff_row) / 2),
                                       round((diff_col) / 2), int((diff_col) / 2), 0)

            image_center = tuple((center_col, center_row))
            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(
                image, matrix, image.shape[1::-1], flags=cv2.INTER_NEAREST)
            return Image(image), center_col, center_row

        def make_image(mask_ridges, index, angle_base, center_col, cy, draw=False, image=True) -> Union[np.ndarray, Tuple[np.ndarray, int, int]]:
            rotated_image, cx_rot, cy_rot = rotate_image(
                mask_ridges, index*angle_base, center_col, cy)
            image_mask, _, _ = rotate_image(
                self.mask_filled, index*angle_base, center_col, cy)
            rotated_image, x_bb, y_bb = self.cut_image(
                image_mask, rotated_image)

            cx_rot -= x_bb
            cy_rot -= y_bb

            if draw:
                rotated_image.image = cv2.cvtColor(
                    rotated_image.image, cv2.COLOR_GRAY2RGB)
                cv2.circle(rotated_image.image,
                           (cx_rot, cy_rot), 7, (0, 0, 255), -1)

                cv2.line(rotated_image.image, (cx_rot, cy_rot),
                         (cx_rot, 0), (0, 0, 255), 1)

            if image:
                return rotated_image.image
            else:
                return rotated_image.image, cx_rot, cy_rot

        # Prepare image variables for calculation
        mask_ridges = Image(self.bin_image_masked_filled.image)

        candidate_results = []
        for angle in range(0, 360, angle_base):
            rotated_image, cx_rot, cy_rot = make_image(
                mask_ridges, 1, angle, cx, cy, draw=False, image=False)

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

        max_sobel_value = max(candidate_results)
        angle = candidate_results.index(max_sobel_value)

        # Generate image which shows the progress
        self.pependicular = Image(make_image(
            mask_ridges, angle, angle_base, cx, cy, draw=True))

        # Count number of lines
        extracted_line, cx_rot, cy_rot = make_image(
            mask_ridges, angle, angle_base, cx, cy, draw=False, image=False)
        extracted_line = extracted_line[:cy_rot, cx_rot]
        ridge_count = 0
        is_on_ridge = False
        for x in range(len(extracted_line)):
            if x + 1 < len(extracted_line) and extracted_line[x] == 255 and extracted_line[x + 1] == 0:
                is_on_ridge = True
            if is_on_ridge and x + 1 < len(extracted_line) and extracted_line[x] == 0 and extracted_line[x + 1] == 255:
                ridge_count += 1
                is_on_ridge = False

        # Apply the mask and rotation on the image
        self.clahe_grayscale_rotated = Image(make_image(
            self.grayscale_clahe_masked, angle, angle_base, cx, cy, draw=False))

        self.aec_masked_rotated = Image(make_image(
            self.aec_masked, angle, angle_base, cx, cy, draw=False))

        extracted_grayscale_line = self.clahe_grayscale_rotated.image[:cy_rot, cx_rot]
        extracted_aec_line = self.aec_masked_rotated.image[:cy_rot, cx_rot]

        # Remove black pixels from array
        for i in range(len(extracted_grayscale_line)):
            if extracted_grayscale_line[i] != 0:
                extracted_grayscale_line = extracted_grayscale_line[i:]
                break

        for i in range(len(extracted_aec_line)):
            if extracted_aec_line[i] != 0:
                extracted_aec_line = extracted_aec_line[i:]
                break

        # Remove masked parts
        extracted_grayscale_line = self.remove_black_array(
            extracted_grayscale_line)

        extracted_aec_line = self.remove_black_array(
            extracted_aec_line)
        self.report.report_pependicular(angle, angle_base, ridge_count)

        return extracted_grayscale_line, extracted_aec_line

    def generate_images(self, path: Path, ext=".jpeg") -> None:
        self.vertical_lines.save(path, f"{self.name}_lines_vertical", ext)
        self.horizontal_lines.save(path, f"{self.name}_lines_horizontal", ext)
        self.lines.save(path, f"{self.name}_lines", ext)
        self.mask_ridges_color_uncut.save(
            path, f"{self.name}_lines_uncut", ext)
        self.clahe_grayscale_center.save(path, f"{self.name}_center", ext)
        self.pependicular.save(path, f"{self.name}_pependicular", ext)
        self.clahe_grayscale_rotated.save(
            path, f"{self.name}_clahe_grayscale_rotated", ext)
        self.aec_masked_rotated.save(
            path, f"{self.name}_aec_masked_rotated", ext)

        Image.save_fig(self.figure_dict, path, f"{self.name}", ext)

    def show(self) -> None:
        # Image.show(self.raw.image, "Raw", scale=0.5)
        Image.show(self.grayscale.image, "Grayscale", scale=0.5)
        # Image.show(self.filtered.image, "Filtered", scale=0.5)
        # Image.show(self.binary.image, "Binary", scale=0.5)

        Image.show(self.mask, name="Mask", scale=0.5)
        Image.show(self.aec, "AEC", scale=0.5)
        Image.show(self.bin_image, "Bin image", scale=0.5)
