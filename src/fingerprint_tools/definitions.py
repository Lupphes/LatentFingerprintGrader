from enum import IntEnum, Enum


class MinutuaeThreshold(IntEnum):
    # Anil K. Jain and David Maltoni. 2003. Handbook of Fingerprint Recognition. Springer-Verlag, Berlin, Heidelberg.
    # Page 361 - 364
    TWELVE_GUIDELINE = 12
    BEYOND_RESONABLE_DOUBT = 20


class RMSEThreshold(float, Enum):
    # This value was picked based on experiments with the dataset.
    VALID = 2.3


class NumberOfRidgesThreshold(IntEnum):
    EXCELENT = 30
    GOOD = 20
    ENOUGH = 15
    POOR = 5
