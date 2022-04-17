import os
import json
import numpy as np
import numpy.typing as npt
from pathlib import Path

from typing import Dict

from .definitions import MinutiaeThreshold, ColorDifferenceThreshold, NumberOfRidgesThreshold
from .string_database import StringDatabase


class Report:
    def __init__(self):
        self.report: Dict = {}

    def report_minutiae(self, number_of_cmp: int, minutiae_points: npt.NDArray) -> None:
        text_description = ''
        if number_of_cmp > MinutiaeThreshold.BEYOND_RESONABLE_DOUBT:
            text_description = StringDatabase.MINUTIAE_POINT_NO_DOUBT
        elif number_of_cmp <= MinutiaeThreshold.BEYOND_RESONABLE_DOUBT and number_of_cmp > MinutiaeThreshold.TWELVE_GUIDELINE:
            text_description = StringDatabase.MINUTIAE_POINT_ENOUGH
        else:
            text_description = StringDatabase.MINUTIAE_POINT_NOT_ENOUGH

        self.report['minutiae_points'] = {
            'quantity': number_of_cmp,
            'description': text_description
            # 'minutiae_points': minutiae_points
        }

    def report_contrast(self, rmse_ridge: np.float64, rmse_valley: np.float64, rmse_ratio: np.float64, color_ration: np.float64, michelson_contrast: np.float64) -> None:
        col_diff_description = ''

        if color_ration > ColorDifferenceThreshold.VALID:
            col_diff_description = StringDatabase.COL_DIFF_VALID
        else:
            col_diff_description = StringDatabase.COL_DIFF_INVALID

        self.report['contrast'] = {
            'rmse_ridge': float(rmse_ridge),
            'rmse_valley': float(rmse_valley),
            'rmse_ratio': float(rmse_ratio),
            'color_ration': float(color_ration),
            'michelson_contrast_pct': float(michelson_contrast),
            'description': col_diff_description
        }

    def report_lines(self, lines_dict: Dict, lines_append: Dict) -> None:
        total_mean = np.mean(np.concatenate(
            [
                lines_dict['horizontal']['count']['array'],
                lines_dict['vertical']['count']['array']
            ]
        ))
        description = ''

        if total_mean > NumberOfRidgesThreshold.EXCELENT:
            description = StringDatabase.RIDGES_EXCELENT
        elif total_mean > NumberOfRidgesThreshold.GOOD:
            description = StringDatabase.RIDGES_GOOD
        elif total_mean > NumberOfRidgesThreshold.ENOUGH:
            description = StringDatabase.RIDGES_ENOUGH
        else:
            description = StringDatabase.RIDGES_POOR

        self.report['papilary_ridges'] = {
            'vertical_mean': np.mean(lines_dict['horizontal']['count']['array']),
            'horizontal_mean': np.mean(lines_dict['vertical']['count']['array']),
            'total_mean': total_mean,
            'description': description
        }

        self.report['papilary_ridges'] = {
            **self.report['papilary_ridges'], **lines_append}

    def report_sinusoidal(self, ridge: int, A_FP: np.float64, A_SIN: np.float64, D_D: np.float64, D_D_ridge: npt.NDArray, name: str) -> None:
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        if not 'sinusoidal_shape' in self.report['papillary_crosscut']:
            self.report['papillary_crosscut']['sinusoidal_shape'] = {}

        self.report['papillary_crosscut']['sinusoidal_shape'][name] = {
            'ridges_low_pass_count': ridge,
            'A_FP': A_FP,
            'A_SIN': A_SIN,
            'D_D': D_D,
            'D_D_ridges': D_D_ridge
        }

    def report_thickness(self, ridge_thickness: npt.NDArray, name: str) -> None:
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        if not 'thickness' in self.report['papillary_crosscut']:
            self.report['papillary_crosscut']['thickness'] = {}

        self.report['papillary_crosscut']['thickness'][name] = {
            'ridges_low_pass_count': len(ridge_thickness),
            'thickness_difference': ridge_thickness
        }

    def report_center_cords(self, cx: int, cy: int, name: str):
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        self.report['papillary_crosscut'][name] = {
            'mask_center': [cx, cy]
        }

    def report_perpendicular(self, angle: int, ridge_count: int, name: str):
        if not 'papillary_crosscut' in self.report:
            self.report['papillary_crosscut'] = {}
        self.report['papillary_crosscut'][name] = {
            'angle': angle,
            'ridges_binary_count': ridge_count
        }

    def report_error(self, name: str, error_msg: str):
        if not 'error' in self.report:
            self.report['error'] = {}
        self.report['error'][name] = {
            'description': error_msg
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
