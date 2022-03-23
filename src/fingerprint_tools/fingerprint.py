import os
import cv2
import logging
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt, argrelextrema

from .contrast_types import ContrastTypes
from .exception import FileError, ArrgumentError, UndefinedVariableError
from .image import Image
from .report import Report

from typing import Dict, Union, Tuple


class Fingerprint:
    def __init__(self, path, ppi):
        self.name: str = Path(path).name
        self.ppi: int = ppi
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
        if self.ppi != 500:
            image = cv2.resize(image, (0, 0), fx=500.0 /
                               self.ppi, fy=500.0 / self.ppi)

        return image2blocks(image, block_size)

    def msu_afis(self, path_image: str, path_destination: str, extractor_class=None):
        from .msu_latentafis.extraction_latent import get_feature_extractor, FeatureExtraction_Latent
        if extractor_class == None:
            extractor_class: FeatureExtraction_Latent = get_feature_extractor()

        logging.info(f'Curently processing: "{self.name}"')
        extractor_class.feature_extraction_single_latent(
            img_file=path_image, output_dir=str(os.path.abspath(path_destination)), ppi=self.ppi, show_processes=False,
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

        logging.info(f'Feature extraction of {self.name} finished!')
        return extractor_class

    def generate_helpers(self) -> None:
        self.aec_masked = Image(self.aec.image)
        self.aec_masked.apply_mask(mask=self.mask)

        self.bin_image_masked = Image(self.bin_image.image)
        self.bin_image_masked.apply_mask(mask=self.mask)

        self.grayscale_masked = Image(self.grayscale.image)
        self.grayscale_masked.apply_mask(mask=self.mask)

        self.grayscale_clahe_masked = Image(self.grayscale_clahe.image)
        self.grayscale_clahe_masked.apply_mask(mask=self.mask)

        self.mask_filled = Image(self.mask.image.copy())
        self.mask_filled.mask_fill()

        self.bin_image_masked_filled = Image(self.bin_image.image)
        self.bin_image_masked_filled.apply_mask(mask=self.mask_filled)

        self.figure_dict = {}
        self.image_dict = {}

    def grade_fingerprint(self) -> None:
        logging.info(f'Grading {self.name}!')

        self.grade_minutiae_points()
        self.grade_contrast()
        exp_x, exp_y = self.grade_lines()
        cent_x, cent_y = self.get_center_cords('mask_center')

        # With mask center
        gray_signal, aec_signal = self.get_pependicular(
            cent_x, cent_y, name='mask_center')

        if len(gray_signal) == 0:
            self.report.report_error(
                'gray', 'The returned empty array. No ridges found.')
        else:
            self.grade_sinusoidal(gray_signal, 'gray')
            self.grade_thickness(gray_signal, 'gray')

        if len(aec_signal) == 0:
            self.report.report_error(
                'aec', 'The returned empty array. No ridges found.')
        else:
            self.grade_sinusoidal(aec_signal, 'aec')
            self.grade_thickness(aec_signal, 'aec')

        # With estimated core
        gray_signal_exp, aec_signal_exp = self.get_pependicular(
            exp_x, exp_y, name='estimated_core')

        if len(gray_signal_exp) == 0:
            self.report.report_error(
                'gray_core', 'The returned empty array. No ridges found.')
        else:
            self.grade_sinusoidal(gray_signal_exp, 'gray_core')
            self.grade_thickness(gray_signal_exp, 'gray_core')

        if len(aec_signal_exp) == 0:
            self.report.report_error(
                'aec_core', 'The returned empty array. No ridges found.')
        else:
            self.grade_sinusoidal(aec_signal_exp, 'aec_core')
            self.grade_thickness(aec_signal_exp, 'aec_core')

        logging.info(f'Grading {self.name} finished!')

    def grade_minutiae_points(self, name='default', thresh=2) -> None:
        if 0 > thresh or thresh > 3:
            raise ArrgumentError()

        if self.common_minutiae is None:
            raise UndefinedVariableError()

        minuatue_points = np.array(
            self.common_minutiae[thresh], dtype=np.uint64)
        number_of_cmp = len(minuatue_points)

        self.report.report_minuatue(number_of_cmp, minuatue_points)

    def grade_contrast(self, name='default') -> None:
        # TODO: Weber?
        # https://stackoverflow.com/questions/68145161/using-python-opencv-to-calculate-vein-leaf-density

        # Create mask with valleys and ridges
        mask_ridges = Image(self.bin_image_masked.image)

        mask_valleys = Image(self.bin_image.image)
        mask_valleys.invert_binary()
        mask_valleys.apply_mask(self.mask)

        # Extracted them from the image
        extracted_ridges: Image = Image(self.grayscale_clahe.image)
        extracted_ridges.apply_mask(mask_ridges)

        extracted_valleys: Image = Image(self.grayscale_clahe.image)
        extracted_valleys.apply_mask(mask_valleys)

        # Michelson contrast
        # Source: https://stackoverflow.com/questions/57256159/how-extract-contrast-level-of-a-photo-opencv
        kernel = np.ones((4, 4), np.uint8)

        cut_gray, _, _ = self.cut_image(self.mask, self.grayscale_clahe)

        reg_min = cv2.erode(cut_gray.image, kernel, iterations=1)
        reg_max = cv2.dilate(cut_gray.image, kernel, iterations=1)

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
        if index != -1:
            processed_extracted = processed_extracted[:-index]
            index = -1
            black_count = 0

        array = np.array(processed_extracted)

        return array

    def grade_lines(self, name='lines', draw=True) -> None:
        # TODO: Does density and count match have relevance
        def create_candidates(dicto, name, value, cord, sample):
            if (len(dicto[name]['candidates']) < sample):
                dicto[name]['candidates'].append(value)
                dicto[name]['axis'].append(cord)
            else:
                if min(dicto[name]['candidates']) < value:
                    index = dicto[name]['candidates'].index(
                        min(dicto[name]['candidates']))
                    dicto[name]['candidates'][index] = value
                    dicto[name]['axis'][index] = cord

        # Define new copy of image
        mask_ridges, offset_x, offset_y = self.cut_image(
            self.mask_filled, self.bin_image_masked)
        row, col = mask_ridges.image.shape
        sample_line_len: int = 3

        # Define properties for output images
        thickness: int = 2
        color: Tuple[int, int, int] = (255, 0, 0)
        mask_ridges_color_count: np.ndarray = cv2.cvtColor(
            mask_ridges.image, cv2.COLOR_GRAY2RGB)
        mask_ridges_color_horizontal_count: np.ndarray = mask_ridges_color_count.copy()
        mask_ridges_color_vertical_count: np.ndarray = mask_ridges_color_count.copy()

        mask_ridges_color_density: np.ndarray = mask_ridges_color_count.copy()
        mask_ridges_color_horizontal_density: np.ndarray = mask_ridges_color_count.copy()
        mask_ridges_color_vertical_density: np.ndarray = mask_ridges_color_count.copy()

        mask_ridges_color_uncut: np.ndarray = cv2.cvtColor(
            self.bin_image_masked.image, cv2.COLOR_GRAY2RGB)

        # Count number of horizontal lines
        horizontal = {
            "count": {
                "array": [],
                "candidates": [],
                "axis": []
            },
            "density": {
                "array": [],
                "candidates": [],
                "axis": []
            }
        }
        count: int = 0
        is_on_ridge: bool = False
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
                    cv2.line(mask_ridges_color_horizontal_count, startpoint,
                             endpoint, color, thickness)
                    cv2.line(mask_ridges_color_count, startpoint,
                             endpoint, color, thickness)

                extracted = self.remove_black_array(mask_ridges.image[x, :])
                density = count/len(extracted)

                horizontal['count']['array'].append(count)
                horizontal['density']['array'].append(density)

                create_candidates(horizontal, 'count',
                                  count, x, sample_line_len)

                create_candidates(horizontal, 'density',
                                  density, x, sample_line_len)
                count = 0

        # Count number of vertical lines
        vertical = {
            "count": {
                "array": [],
                "candidates": [],
                "axis": []
            },
            "density": {
                "array": [],
                "candidates": [],
                "axis": []
            }
        }
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
                    cv2.line(mask_ridges_color_vertical_count, startpoint,
                             endpoint, color, thickness)
                    cv2.line(mask_ridges_color_count, startpoint,
                             endpoint, color, thickness)

                extracted = self.remove_black_array(mask_ridges.image[:, y])
                density = count/len(extracted)

                vertical['count']['array'].append(count)
                vertical['density']['array'].append(density)

                create_candidates(vertical, 'count',
                                  count, y, sample_line_len)

                create_candidates(vertical, 'density',
                                  density, y, sample_line_len)
                count = 0

        color_mark = (0, 255, 0)

        # Create the dictionary for output
        cord_count: Dict = {}
        cord_density: Dict = {}
        for i in range(sample_line_len):
            if draw:
                startpoint = (0, horizontal['count']['axis'][i])
                endpoint = (col, horizontal['count']['axis'][i])
                cv2.line(mask_ridges_color_horizontal_count, startpoint,
                         endpoint, color_mark, thickness)
                cv2.line(mask_ridges_color_count, startpoint,
                         endpoint, color_mark, thickness)
                startpoint = (vertical['count']['axis'][i], 0)
                endpoint = (vertical['count']['axis'][i], row)
                cv2.line(mask_ridges_color_vertical_count, startpoint,
                         endpoint, color_mark, thickness)
                cv2.line(mask_ridges_color_count, startpoint,
                         endpoint, color_mark, thickness)

                startpoint = (0, horizontal['density']['axis'][i])
                endpoint = (col, horizontal['density']['axis'][i])
                cv2.line(mask_ridges_color_horizontal_density, startpoint,
                         endpoint, color_mark, thickness)
                cv2.line(mask_ridges_color_density, startpoint,
                         endpoint, color_mark, thickness)

                startpoint = (vertical['density']['axis'][i], 0)
                endpoint = (vertical['density']['axis'][i], row)
                cv2.line(mask_ridges_color_vertical_density, startpoint,
                         endpoint, color_mark, thickness)
                cv2.line(mask_ridges_color_density, startpoint,
                         endpoint, color_mark, thickness)

            cord_count[f"{vertical['count']['axis'][i]+ offset_x}:{horizontal['count']['axis'][i]+ offset_y}"] = [
                vertical['count']['candidates'][i], horizontal['count']['candidates'][i]]

            cord_density[f"{vertical['density']['axis'][i]+ offset_x}:{horizontal['density']['axis'][i]+ offset_y}"] = [
                vertical['density']['candidates'][i], horizontal['density']['candidates'][i]]

        expected_core_x: int = (np.int(np.mean(vertical['density']['axis'])))
        expected_core_y: int = (np.int(np.mean(horizontal['density']['axis'])))

        expected_core_x_off: int = expected_core_x + offset_x
        expected_core_y_off: int = expected_core_y + offset_y

        if draw:
            cv2.circle(mask_ridges_color_count, (expected_core_x,
                                                 expected_core_y), 7, (0, 0, 255), -1)
            cv2.putText(mask_ridges_color_count, "core",
                        (expected_core_x - 35, expected_core_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.circle(mask_ridges_color_uncut, (expected_core_x_off,
                                                 expected_core_y_off), 7, (0, 0, 255), -1)
            cv2.putText(mask_ridges_color_uncut, "core",
                        (expected_core_x_off - 35, expected_core_y_off - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.circle(mask_ridges_color_density, (expected_core_x,
                                                   expected_core_y), 7, (0, 0, 255), -1)
            cv2.putText(mask_ridges_color_density, "core",
                        (expected_core_x - 35, expected_core_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        lines_dict = {
            "vertical": vertical,
            "horizontal": horizontal
        }

        lines_append = {
            "expected_core": [expected_core_x_off, expected_core_y_off],
            "higest_frequency": cord_count,
            "heigest_density": cord_density
        }

        self.report.report_lines(lines_dict, lines_append)

        # Images to generate later
        if not name in self.image_dict:
            self.image_dict[name] = {}
        self.image_dict[name]["vertical"] = Image(
            mask_ridges_color_vertical_count)
        self.image_dict[name]["horizontal"] = Image(
            mask_ridges_color_horizontal_count)
        self.image_dict[name]["full_vh"] = Image(mask_ridges_color_count)
        self.image_dict[name]["uncut"] = Image(mask_ridges_color_uncut)

        self.image_dict[name]["density_vertical"] = Image(
            mask_ridges_color_vertical_density)
        self.image_dict[name]["density_horizontal"] = Image(
            mask_ridges_color_horizontal_density)
        self.image_dict[name]["density_full_vh"] = Image(
            mask_ridges_color_density)

        return expected_core_x_off, expected_core_y_off

    def grade_sinusoidal(self, line_signal: np.ndarray, name: str, draw=True, really=False) -> None:
        # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
        def butter_lowpass(cutoff, nyq_freq, order=4):
            normal_cutoff = float(cutoff) / nyq_freq
            b, a = butter(order, normal_cutoff, btype='lowpass',
                          analog=False, fs=None)
            return b, a

        def butter_lowpass_filter(x, cutoff_freq, nyq_freq, order=4):
            b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
            y = filtfilt(b, a, x, axis=0)
            return y

        def square_diff_match(line_signal, sin_array, sin_one) -> np.ndarray:
            # Using the Difference of squares to find the best period aligement
            best_diff = None
            best_sin = None
            stack_array = np.array([])
            period = sin_one
            sin_array_shift = np.append(sin_array, period)
            for _ in range(len(period)):
                stack_array = sin_array_shift[:-len(period)]
                diff_sqr = np.sum((stack_array - line_signal) ** 2)

                if best_diff == None:
                    best_diff = diff_sqr
                    best_sin = stack_array
                if diff_sqr < best_diff:
                    best_diff = diff_sqr
                    best_sin = stack_array

                last, sin_array_shift = sin_array_shift[-1], sin_array_shift[:-1]
                sin_array_shift = np.append(last, sin_array_shift)
            return np.array(best_sin)

        # Normalize signal to <0;1>
        line_signal = line_signal.astype('float64')
        line_signal /= np.max(np.abs(line_signal), axis=0)

        if len(line_signal) < 16:
            self.report.report_error('pependicular_low_pass',
                                     'The pependicular line is too short to analyze')
            return
        low_pass = butter_lowpass_filter(line_signal, 5, 50/2)

        local_min = argrelextrema(low_pass, np.less)[0]
        local_max = argrelextrema(low_pass, np.greater)[0]
        points_y = np.full(len(local_max), 1.9, dtype=np.float16)

        # Normalize signal to <0;2>
        line_signal *= 2

        ridge_count = len(local_max)
        signal_lenght = len(line_signal)

        if ridge_count == 0:
            self.report.report_error('pependicular_low_pass',
                                     'The algorithm didn\'t found any ridges to analyze.')
            return

        ridge_per_part = signal_lenght // ridge_count
        rest = signal_lenght % ridge_count

        # Calculate the aligned sinus
        x_one = np.linspace((-np.pi)/2, 3 * np.pi / 2, ridge_per_part)
        sin_one = np.sin(x_one) + 1  # + 1 normalization
        sin_array = np.array([])
        for _ in range(ridge_count):
            sin_array = np.append(sin_array, sin_one)

        # Rest compensation
        sin_rest = np.array([])
        while rest > len(sin_rest):
            sin_rest = np.append(sin_rest, sin_one)

        sin_last = sin_rest[:rest]

        sin_array = np.append(sin_array, sin_last)

        # Using the Difference of squares to find the best period aligement
        best_sin = square_diff_match(line_signal, sin_array, sin_one)

        if draw:
            fig_sinus_all: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)
            plt.title('Ridges compared to a partial sinusoid')
            plt.plot(line_signal, label="Ridges all", figure=fig_sinus_all)
            plt.plot(best_sin, label="Sinus all", figure=fig_sinus_all)
            plt.plot(local_max, points_y, 'o', color="green")
            for item in local_min:
                plt.axvline(x=item, color='red', linestyle='--',
                            figure=fig_sinus_all)
            plt.xlabel("Pixels (X)", fontsize='small', figure=fig_sinus_all)
            plt.ylabel("Grayscale values (Y)",
                       fontsize='small', figure=fig_sinus_all)
            plt.legend(fontsize='small', loc='upper right')
            plt.rc('font', size=14)
            plt.close()
            if not name in self.figure_dict:
                self.figure_dict[name] = {}
            self.figure_dict[name]['sin_all'] = fig_sinus_all

            fig_sin_overview: plt.Figure = plt.figure(
                figsize=(22, 5), dpi=150)
            plt.title('Overview of each ridge')
            plt.plot(line_signal, label="Ridges", figure=fig_sin_overview)
            plt.plot(local_max, points_y, 'o', color="green")
            for item in local_min:
                plt.axvline(x=item, color='red', linestyle='--',
                            figure=fig_sin_overview)
            plt.xlabel("Pixels (X)", fontsize='small',
                       figure=fig_sin_overview)
            plt.ylabel("Grayscale values (Y)", fontsize='small',
                       figure=fig_sin_overview)
            plt.legend(fontsize='small', loc='upper right')
            plt.rc('font', size=14)
            plt.close()
            self.figure_dict[name]['sin_overview'] = fig_sin_overview

        # Calculate the integral under the discrete signal
        A_FP = simpson(line_signal, axis=0)
        A_SIN = simpson(best_sin, axis=0)

        D_D = (A_FP/A_SIN - 1) * 100

        # Calculate D_D for each ridge
        ridges = np.split(line_signal, local_min)[1:-1]
        sin_all_count = []
        ridges_all_count = []
        D_D_ridge = []
        for i in range(len(ridges)):
            x_one = np.linspace((-np.pi)/2, 3 * np.pi / 2, len(ridges[i]))
            sin_one = np.sin(x_one) + 1  # + 1 normalization
            best_sin = square_diff_match(ridges[i], sin_one, sin_one)

            A_FP_ridge = simpson(ridges[i], axis=0)
            A_SIN_ridge = simpson(best_sin, axis=0)

            D_D_ridge.append((A_FP_ridge/A_SIN_ridge - 1) * 100)

            sin_all_count = np.append(sin_all_count, best_sin)
            ridges_all_count = np.append(ridges_all_count, ridges[i])

            if draw and really:
                fig_sin_one: plt.Figure = plt.figure(
                    figsize=(10, 5), dpi=150)
                plt.title('EACH ridge ptimized to its sinus DEBUG')
                plt.plot(ridges[i], label="One ridge", figure=fig_sin_one)
                plt.plot(best_sin, label="One sinus", figure=fig_sin_one)
                plt.xlabel("Pixels (X)", fontsize='small',
                           figure=fig_sin_one)
                plt.ylabel("Grayscale values (Y)",
                           fontsize='small', figure=fig_sin_one)
                plt.legend(fontsize='small', loc='upper right')
                plt.rc('font', size=14)
                fig_sin_one.savefig(f"{self.name}_one_{i}.png")
                plt.close()

        if draw:
            fig_sin_optimalised: plt.Figure = plt.figure(
                figsize=(22, 5), dpi=150)
            plt.title('Optimized ridges each to match sinus')
            plt.plot(ridges_all_count, label="Ridges optimalised",
                     figure=fig_sin_optimalised)
            plt.plot(sin_all_count, label="Sinus optimalised",
                     figure=fig_sin_optimalised)
            plt.xlabel("Pixels (X)", fontsize='small',
                       figure=fig_sin_optimalised)
            plt.ylabel("Grayscale values (Y)", fontsize='small',
                       figure=fig_sin_optimalised)
            plt.legend(fontsize='small', loc='upper right')
            plt.rc('font', size=14)
            plt.close()
            self.figure_dict[name]['sin_optimalised'] = fig_sin_optimalised

        self.report.report_sinusoidal(
            ridge_count, A_FP, A_SIN, D_D, D_D_ridge, name)

    def grade_thickness(self, line_signal: np.ndarray, name: str, draw=True) -> None:
        # TODO: Check thickness DPI but should be alright
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
            fig_thickness = plt.figure(figsize=(22, 5), dpi=150)
            plt.title('Thickness of ridges')
            plt.plot(line_signal, label="Ridges", figure=fig_thickness)
            plt.plot(display_thickness, label="Thickness",
                     figure=fig_thickness)
            plt.xlabel("Pixels (X)", fontsize='small', figure=fig_thickness)
            plt.ylabel("Grayscale values (Y)",
                       fontsize='small', figure=fig_thickness)
            plt.legend(fontsize='small', loc='upper right')
            plt.rc('font', size=14)
            plt.close()
            if not name in self.figure_dict:
                self.figure_dict[name] = {}
            self.figure_dict[name]['thick'] = fig_thickness

        # Split the ridges into individual arrays
        ridges_separated = np.where(display_thickness != 0)[0]
        ridges_list = np.split(display_thickness[ridges_separated],
                               np.where(np.diff(ridges_separated) != 1)[0]+1)

        # Transform the pixels into readable format
        base = 2.54 / self.ppi
        # TODO: Evaluate if PPI is good
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

    def get_center_cords(self, name: str, draw=True) -> Union[int, int]:
        # Find countours and therfore draw the center
        contour, _ = cv2.findContours(
            self.mask_filled.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clahe_grayscale_center = Image(self.clahe_grayscale_color.image)
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
                    cv2.drawContours(clahe_grayscale_center.image, [
                        i], -1, (255, 0, 0), 2)
                    cv2.circle(clahe_grayscale_center.image,
                               (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(clahe_grayscale_center.image, "center", (cx - 45, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.report.report_center_cords(cx, cy, name)

        if not name in self.image_dict:
            self.image_dict[name] = {}
        self.image_dict[name]["center_cords"] = clahe_grayscale_center
        return cx, cy

    def get_pependicular(self, cx: int, cy: int, name: str, angle_base=1):
        # TODO: Optimalisation (read in 4 directions) - rotate just till 90
        def rotate_image(image: Image, angle: int, center_col: int, center_row: int) -> Tuple[Image, int, int]:
            image = image.image
            row, col = image.shape
            diff_col = col - center_col
            diff_row = row - center_row

            dimensions = np.array([row, col, diff_row, diff_col])
            dim_max = np.max(dimensions)

            top_add = dim_max - center_row
            right_add = dim_max - diff_col
            bottom_add = dim_max - diff_row
            left_add = dim_max - center_col

            center_col += left_add
            center_row += top_add

            image = cv2.copyMakeBorder(
                image, top=top_add, bottom=bottom_add, right=right_add, left=left_add, borderType=0)

            image_center = tuple((int(center_col), int(center_row)))
            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(
                image, matrix, image.shape[1::-1], flags=cv2.INTER_NEAREST)
            return Image(image), center_col, center_row

        def make_image(mask_ridges, angle, center_col, cy, draw=False, image=True) -> Union[np.ndarray, Tuple[np.ndarray, int, int]]:
            rotated_image, cx_rot, cy_rot = rotate_image(
                mask_ridges, angle, center_col, cy)
            image_mask, _, _ = rotate_image(
                self.mask_filled, angle, center_col, cy)
            rotated_image, x_bb, y_bb = self.cut_image(
                image_mask, rotated_image)

            cx_rot -= x_bb
            cy_rot -= y_bb

            # If coordinate is out from image,
            # set in on the most bottom position
            row, _ = rotated_image.image.shape
            if row < cy_rot:
                cy_rot = row

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
        mask_ridges = Image(self.bin_image_masked.image)

        candidate_results = []
        count_results = []
        for angle in range(0, 360, angle_base):
            rotated_image, cx_rot, cy_rot = make_image(
                mask_ridges, angle, cx, cy, draw=False, image=False)

            # If coordinate is not on the picture
            # skip it
            if cx_rot < 0 or cy_rot < 0:
                candidate_results.append(0)
                count_results.append(None)
                continue
            row, col = rotated_image.shape
            if col < cx_rot:  # or row < cx_rot
                candidate_results.append(0)
                count_results.append(None)
                continue

            # Count number of lines
            extracted_line = rotated_image[:cy_rot, cx_rot:cx_rot+1]
            ridge_count = 0
            is_on_ridge = False
            for x in range(len(extracted_line)):
                if x + 1 < len(extracted_line) and extracted_line[x] == 255 and extracted_line[x + 1] == 0:
                    is_on_ridge = True
                if is_on_ridge and x + 1 < len(extracted_line) and extracted_line[x] == 0 and extracted_line[x + 1] == 255:
                    ridge_count += 1
                    is_on_ridge = False
            if ridge_count == 0:
                candidate_results.append(0)
                count_results.append(None)
                continue

            # Blur the binary image and apply Sobel gradient in Y direction
            # throw white and black and compute mean, the greater the value
            # the better precision

            edge_detection = cv2.GaussianBlur(
                rotated_image, (21, 7), cv2.BORDER_DEFAULT)
            edge_detection = cv2.Sobel(edge_detection, 0, dx=0, dy=1)
            edge_detection = np.uint8(np.absolute(edge_detection))

            # Throw black (0) and white (255) out
            edge_detection = edge_detection[:cy_rot, cx_rot:cx_rot+1]
            edge_detection = [
                i for i in edge_detection if i not in [0, 255]]

            # If nothing was found
            if len(edge_detection) == 0:
                candidate_results.append(0)
                count_results.append(None)
            else:
                # Calculate the mean
                candidate_results.append(np.mean(edge_detection))
                count_results.append(ridge_count)

        max_sobel_value = np.max(candidate_results)
        angle = candidate_results.index(max_sobel_value)
        ridge_count = count_results[angle]
        angle *= angle_base

        # Apply the mask and rotation on the image
        # and get correct coordinations
        self.clahe_grayscale_rotated, cx_rot, cy_rot = make_image(
            self.grayscale_clahe_masked, angle, cx, cy, draw=False, image=False)

        self.clahe_grayscale_rotated = Image(self.clahe_grayscale_rotated)

        self.aec_masked_rotated = Image(make_image(
            self.aec_masked, angle, cx, cy, draw=False))

        extracted_grayscale_line = self.clahe_grayscale_rotated.image[:cy_rot, cx_rot:cx_rot+1]
        extracted_aec_line = self.aec_masked_rotated.image[:cy_rot,
                                                           cx_rot:cx_rot+1]

        extracted_grayscale_line = np.rot90(extracted_grayscale_line)[0]
        extracted_aec_line = np.rot90(extracted_aec_line)[0]
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

        self.report.report_pependicular(angle, ridge_count, name)

        # Generate image which shows the progress
        if not name in self.image_dict:
            self.image_dict[name] = {}
        self.image_dict[name]["pependicular"] = Image(make_image(
            mask_ridges, angle, cx, cy, draw=True)
        )
        self.image_dict[name]["pependicular_clahe"] = Image(make_image(
            self.grayscale_clahe_masked, angle, cx, cy, draw=True)
        )
        self.image_dict[name]["pependicular_aec"] = Image(make_image(
            self.aec_masked, angle, cx, cy, draw=True)
        )

        return extracted_grayscale_line, extracted_aec_line

    def generate_images(self, path: Path, ext=".jpeg") -> None:
        Image.save_img(self.image_dict, path, self.name, ext)
        Image.save_fig(self.figure_dict, path, self.name, ext)

    def show(self) -> None:
        # Image.show("Raw", self.raw.image, scale=0.5)
        Image.show("Grayscale", self.grayscale.image, scale=0.5)
        # Image.show("Filtered", self.filtered.image, scale=0.5)
        # Image.show("Binary", self.binary.image, scale=0.5)

        Image.show(name="Mask", image=self.mask, scale=0.5)
        Image.show("AEC", self.aec, scale=0.5)
        Image.show("Bin image", self.bin_image, scale=0.5)
