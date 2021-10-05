

import os
import cv2

from fingerprint_tools import Fingerprint, FingerprintTools, ContrastTypes

test = Fingerprint('/mnt/c/Users/Luppo/Desktop/BP/FingerprintQualityGrader/img/012_3_1.png', contrast_type=ContrastTypes.CLAHE)
test.show(name='Test', scale=1.2)

cv2.waitKey(0)
cv2.destroyAllWindows()