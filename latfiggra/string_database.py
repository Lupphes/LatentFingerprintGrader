from enum import Enum


class StringDatabase(str, Enum):
    """
    String dataset for every string line exported to the JSON.
    Describes errors or statuses which occurred.
    """
    ERR_NO_FINGERPRINT = 'There is no fingerprint on the image. Please check your input.'
    ERR_PERPENDICULAR_GENERAL = 'No ridges were found on the perpendicular lines, or it was too short (<15px). Further analysis is pointless.'
    ERR_PERPENDICULAR_TOO_SHORT = 'The perpendicular line is too short for analysis.'
    ERR_PERPENDICULAR_NO_RIDGE = 'No ridges were found on the perpendicular line.'
    EXCEPT_MASK_BLACK = 'The mask image is black. No fingerprint was found. This should\'ve been detected sooner. Check the output.'

    MINUTIAE_POINT_NO_DOUBT = 'Enough minutiae points for identification beyond reasonable doubt'
    MINUTIAE_POINT_ENOUGH = 'Enough minutiae points for identification with possible error'
    MINUTIAE_POINT_NOT_ENOUGH = 'Not enough minutiae points for identification'

    COL_DIFF_VALID = 'The contrast difference is enough to differentiate between ridges and valleys'
    COL_DIFF_INVALID = 'The contrast difference is not enough to differentiate between ridges and valleys'

    RIDGES_EXCELENT = 'The fingerprint has a great number of papillary ridges'
    RIDGES_GOOD = 'The fingerprint has a good amount of papillary ridges'
    RIDGES_ENOUGH = 'The fingerprint has enough papillary ridges for identification'
    RIDGES_POOR = 'The fingerprint does not have enough papillary ridges for identification'
