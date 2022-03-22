import os
import json
import numpy as np
from pathlib import Path

from typing import Dict

from .definitions import MinutuaeThreshold, RMSEThreshold, NumberOfRidgesThreshold


class Report:
    def __init__(self):
        self.report: Dict = {}

    def report_minuatue(self, number_of_cmp: int, minuatue_points: np.ndarray) -> None:
        text_description = ""
        if number_of_cmp > MinutuaeThreshold.BEYOND_RESONABLE_DOUBT:
            text_description = "Enough minutiae points for identification beyond reasonable doubt"
        elif number_of_cmp <= MinutuaeThreshold.BEYOND_RESONABLE_DOUBT and number_of_cmp > MinutuaeThreshold.TWELVE_GUIDELINE:
            text_description = "Enough minutiae points for identification with possible error"
        else:
            text_description = "Not enough minutiae points for identification"

        # TODO: Add points to JSON
        self.report['minutuae_points'] = {
            "quantity": number_of_cmp,
            "description": text_description
        }

    def report_contrast(self, rmse: np.float64, michelson_contrast: np.float64) -> None:
        rmse_description = ""

        if rmse > RMSEThreshold.VALID:
            rmse_description = "The contrast has proven that the fingerprint is valid"
        else:
            rmse_description = "The contrast has proven that the fingerprint is not valid"

        self.report['contrast'] = {
            "rmse": float(rmse),
            "michelson_contrast_pct": float(michelson_contrast),
            "description": rmse_description
        }

    def report_lines(self, lines_dict: Dict, lines_append: Dict) -> None:
        total_mean = np.mean(np.concatenate(
            [
                lines_dict['horizontal']['count']['array'],
                lines_dict['vertical']['count']['array']
            ]
        ))
        description = ""

        if total_mean > NumberOfRidgesThreshold.EXCELENT:
            description = "Fingerprint has a great number of papillary ridges"
        elif total_mean > NumberOfRidgesThreshold.GOOD:
            description = "Fingerprint has a good amount of papillary ridges"
        elif total_mean > NumberOfRidgesThreshold.ENOUGH:
            description = "Fingerprint has enough papillary ridges for identification"
        else:
            description = "Fingerprint does not have enough papillary ridges for identification"

        self.report['papilary_ridges'] = {
            "vertical_mean": np.mean(lines_dict['horizontal']['count']['array']),
            "horizontal_mean": np.mean(lines_dict['vertical']['count']['array']),
            "total_mean": total_mean,
            "description": description
        }

        self.report['papilary_ridges'] = {
            **self.report['papilary_ridges'], **lines_append}

    def report_sinusoidal(self, ridge: int, A_FP: np.float64, A_SIN: np.float64, D_D: np.float64, D_D_ridge: np.ndarray, name: str) -> None:
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        if not 'sinusoidal_shape' in self.report['papillary_crosscut']:
            self.report['papillary_crosscut']['sinusoidal_shape'] = {}

        self.report['papillary_crosscut']['sinusoidal_shape'][name] = {
            "ridges_low_pass_count": ridge,
            "A_FP": A_FP,
            "A_SIN": A_SIN,
            "D_D": D_D,
            "D_D_ridges": D_D_ridge
        }

    def report_thickness(self, ridge_thickness: np.ndarray, name: str) -> None:
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        if not 'thickness' in self.report['papillary_crosscut']:
            self.report['papillary_crosscut']['thickness'] = {}

        self.report['papillary_crosscut']['thickness'][name] = {
            "ridges_low_pass_count": len(ridge_thickness),
            "thickness_difference": ridge_thickness
        }

    def report_center_cords(self, cx: int, cy: int, name: str):
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        self.report['papillary_crosscut'][name] = {
            "mask_center": [cx, cy]
        }

    def report_pependicular(self, angle: int, angle_base: int, ridge_count: int, name: str):
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        self.report['papillary_crosscut'][name] = {
            "angle": angle * angle_base,
            "ridges_binary_count": ridge_count
        }

    def report_error(self, name: str, error_msg: str):
        if not 'error' in self.report:
            self.report['error'] = {}
        self.report['error'][name] = {
            "description": error_msg
        }

    def add(self, name: str, input: Dict) -> None:
        self.report[name] = input

    def generate_report(self, dirname: Path, name: str, fingerprint: str) -> None:
        filename = os.path.join(dirname, name)
        if not (os.path.isfile(filename) and os.access(filename, os.R_OK)):
            with open(filename, 'w') as file:
                json.dump({}, file, indent=4)

        with open(filename, 'r+') as file:
            file_data = json.load(file)
            file_data[fingerprint] = self.report
            file.seek(0)
            json.dump(file_data, file, indent=4)
