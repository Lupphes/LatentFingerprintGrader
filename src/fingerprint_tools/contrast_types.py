from enum import Enum


class ContrastTypes(Enum):
    WEBER = 1
    MICHELSON = 2
    RMS = 3
    CLAHE = 4


class ThresholdFlags(Enum):
    MEAN = 1
    GAUSSIAN = 2
    OTSU = 3
    TRIANGLE = 4
    MANUAL = 5
