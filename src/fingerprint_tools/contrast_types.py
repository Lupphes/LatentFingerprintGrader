from enum import Enum


class ContrastTypes(Enum):
    """
    All contrast types which are available with the 
    contrasting function
    """
    WEBER = 1
    MICHELSON = 2
    RMS = 3
    CLAHE = 4


class ThresholdFlags(Enum):
    """
    All threshold types which are available with the 
    thresholding function
    """
    MEAN = 1
    GAUSSIAN = 2
    OTSU = 3
    TRIANGLE = 4
    MANUAL = 5
