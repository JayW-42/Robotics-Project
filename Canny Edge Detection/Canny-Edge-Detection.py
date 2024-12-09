import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
from scipy import signal
from numpy.linalg import svd, inv
import os
#from lolviz import matrixviz

os.chdir(os.path.join(os.path.dirname(__file__)))

print(r'driving.jpg' in os.listdir("."))
img = cv2.imread(r'driving.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('driving',img)

# Use a Box Filter to blur the image
def noise_reduction(img):
    blurred = cv2.blur(img,(4,4))
    return blurred

