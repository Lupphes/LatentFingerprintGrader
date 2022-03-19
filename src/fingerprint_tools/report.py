import os
import json

from typing import Dict


class Report:
    def __init__(self):
        self.report: Dict = {}

    def report_minuatue(self):
        pass

    def report_contrast(self):
        pass

    def report_lines(self):
        pass

    def report_sinusoidal():
        pass

    def report_thickness(self):
        pass

    def add(self, name, input: Dict) -> None:
        self.report[name] = input
