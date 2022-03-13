from enum import IntEnum, Enum


class MinutuaeThreshold(IntEnum):
    TWELVE_GUIDELINE = 12
    BEYOND_RESONABLE_DOUBT = 20


class RMSEThreshold(float, Enum):
    VALID = 1.5


class NumberOfRidgesThreshold(IntEnum):
    EXCELENT = 30
    GOOD = 20
    ENOUGH = 15
    POOR = 5
