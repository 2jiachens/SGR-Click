"""OpenCV decoding that also supports Unicode paths on Windows."""
import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
