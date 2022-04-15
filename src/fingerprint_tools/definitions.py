from enum import IntEnum, Enum


class MinutiaeThreshold(IntEnum):
    """
    The threshold for Minutiae points specified by 
    12-guidelines and additional margin for error

    Anil K. Jain and David Maltoni. 2003. 
    Handbook of Fingerprint Recognition. 
    Springer-Verlag, Berlin, Heidelberg.
    Page 361 - 364
    """
    TWELVE_GUIDELINE = 12
    BEYOND_RESONABLE_DOUBT = 20


class RMSEThreshold(float, Enum):
    """
    The threshold for Root Mean Square Error.
    This value was selected based on experiments with the 
    dataset, and since it is a new method making it more 
    accurate is desirable
    """
    VALID = 0.00901960784314


class NumberOfRidgesThreshold(IntEnum):
    """
    Thresholds that grade the number of ridges
    This value was selected based on readings from the dataset
    and from general average sensor readings
    """
    EXCELENT = 30
    GOOD = 20
    ENOUGH = 15
    POOR = 5
