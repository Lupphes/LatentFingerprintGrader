

import os
import cv2

from fingerprint_tools import Fingerprint, FingerprintTools, ContrastTypes

image_path = 'img/012_3_1.png'
fingerprint_image = Fingerprint(path=image_path, contrast_type=ContrastTypes.CLAHE)
fingerprint_image.show(name='Given fingerprint', scale=1.2)

cv2.waitKey(0)
cv2.destroyAllWindows()