'''
ROBO 5000 Final Project - 1.3 Canny Edge Detection
Nikhik Sawane & Jay Warren

Collaborators: Allie Everett

Some code reused from Jay's CSCI 5722 Quiz 3 - Canny Edge Detection
'''

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

# Use a Box Filter to blur the image
def noise_reduction(img):
    blurred = cv2.blur(img,(4,4))
    return blurred


# TODO - rework
def plot_step(img_a, img_b, title_a, title_b):
    '''
    Input
      - img_a (numpy array): the left image
      - img_b (numpy array): the right image
      - title_a (string): the left title
      - title_b (string): the right title
    Output
      - No variables are returned
      - Displays a plot with two side-by-side images, to visualize what each step of the algorithm does
    '''
    plt.subplot(121),plt.imshow(img_a, cmap = 'gray')
    plt.title(title_a), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_b, cmap = 'gray')
    plt.title(title_b), plt.xticks([]), plt.yticks([])
    plt.show()

def gradient(img):
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    # TODO - sobel function?
    #gradient_x = cv2.Sobel(img, -1, 1, 0);
    #gradient_y = cv2.Sobel(img, -1, 0, 1);
    gradient_x = cv2.filter2D(img, ddepth =-1, kernel=sobelx)
    gradient_y = cv2.filter2D(img, ddepth =-1, kernel=sobely)

    return gradient_x, gradient_y

def noise_reduction(img):
    blurred = cv2.GaussianBlur(img,(1,1),0)
    return blurred

noise_reduced_img = noise_reduction(img)

# TODO - plotting?
plot_step(img, noise_reduced_img, 'Original Image', 'Noise-Reduced Image')

def scale(x):
    normX = ((x - np.min(x))/(np.max(x)-np.min(x))) * 255
    return normX

gradient_x, gradient_y = gradient(noise_reduced_img)
scaled_img = scale(np.hypot(gradient_x, gradient_y))
theta = np.arctan2(gradient_y, gradient_x)
plot_step(gradient_x, gradient_y, 'Gradient X', 'Gradient Y')
plot_step(noise_reduced_img, scaled_img, 'Noise-Reduced Image', 'Gradient Image')

def non_max_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32) 
    angle = theta * 180.0 / np.pi  # max -> 180, min -> -180
    angle[angle < 0] += 180  # max -> 180, min -> 0

    # loop through all the image pixels and determine if the current pixel is a local maximum or not
    for i in range(M-1):
      for j in range(N-1):
        #print(angle[i,j])
        # if right-left
        if (angle[i,j] < 36):
          n1 = G[i,j-1]
          n2 = G[i,j+1]
        # if diagonal
        elif ( 36 <= angle[i,j] < 72):
          n1 = G[i-1,j+1]
          n2 = G[i+1,j-1]
        # if up-down
        elif (72 < angle[i,j] <= 108):
          n1 = G[i-1,j]
          n2 = G[i+1,j]
        elif (108 < angle[i,j] <= 144):
          n1 = G[i-1,j-1]
          n2 = G[i+1,j+1]
        elif (144 < angle[i,j] <= 180):
          n1 = G[i,j-1]
          n2 = G[i,j+1]

        if (n1 <= G[i,j] >= n2):
          Z[i,j] = G[i,j]
        else:
          Z[i,j] = 0

    return Z

max_img = non_max_suppression(scaled_img, theta) 

plot_step(scaled_img, max_img, 'Gradient Image', 'Non-Max Suppressed Image')

# plot_step(scaled_img[200:230,10:40], max_img[200:230,10:40], 'Gradient Image', 'Non-Max Suppressed Image')

# window = theta[200:230,10:40]

# x = np.arange(200, 230)
# y = np.arange(10, 40)
# u = 2 * np.cos(window)
# v = 2 * np.sin(window)

# plt.quiver(x, y, u, v)

def threshold(img, lowThresholdRatio=0.03, highThresholdRatio=.1, weak=25, strong=255):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    M,N = img.shape

    # create empty image of zeros to store threshold values
    Z = np.zeros([M, N])

    for i in range(M):
      for j in range(N):
        # find indices where img is above highThreshold
        if (img[i,j] >= highThreshold):
          Z[i,j] = strong
        # find indices where img is above lowThreshold and below highThreshold
        elif (img[i,j] >= lowThreshold) and (img[i,j] < highThreshold):
          Z[i,j] = weak
        else:
          Z[i,j] = 0
    print(Z)

    return Z

thresholded_img = threshold(max_img)

plot_step(max_img, thresholded_img, 'Non-Max Suppressed Image', 'Thresholded Image')

def hysteresis(img, weak=25, strong=255):
    M,N = img.shape

    # create empty image of zeros to store threshold values
    Z = np.zeros([M, N])

    for i in range(1, M-1):
      for j in range(1, N-1):
        pixel = img[i,j]

        if (pixel == weak):
          neighbors = np.array([img[i+1,j],img[i+1,j+1],img[i,j+1],img[i-1,j+1],
                                img[i-1,j],img[i-1,j-1],img[i,j-1],img[i+1,j-1]])
          if strong in neighbors:
            Z[i,j] = 255
          else:
            Z[i,j] = 0
        elif(pixel == strong):
          Z[i,j] = 255

    return Z

hysteresis_img = hysteresis(thresholded_img)

plot_step(thresholded_img, hysteresis_img, 'Thresholded Image', 'Final Image')

minVal = 75
maxVal = 100

edges = cv2.Canny(img, minVal, maxVal)
plot_step(img, edges, 'Original Image', 'Edge Detection with OpenCV')