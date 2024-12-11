'''
ROBO 5000 Final Project - 1.3 Canny Edge Detection with DFS Hysteresis
Nikhil Sawane & Jay Warren
Collaborator: Allie Everett - assisted debugging non_max_supression()

Some code reused from CSCI 5722 Quiz 3 - Canny Edge Detection
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import svd, inv
import os

os.chdir(os.path.join(os.path.dirname(__file__)))

# read grayscale image data
img = cv2.imread(r'KeyDragon.png', cv2.IMREAD_GRAYSCALE)
colorImg = cv2.cvtColor(cv2.imread(r'KeyDragon.png'), cv2.COLOR_BGR2RGB)

'''
Display the initial, final, and OpenCV images
Input   -   Initial Image
            Final Imagec
            OpenCV Image
An expanded and slightly modified plot_step()
'''
def displayResults(imgI, imgF, imgOCV):
    plt.subplot(131),plt.imshow(imgI)
    plt.title("Initial Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(imgF, cmap = 'gray')
    plt.title("Final Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(imgOCV, cmap = 'gray')
    plt.title("OpenCV"), plt.xticks([]), plt.yticks([])
    plt.show()

'''
Display any 2 images and their labels
Input   -   Image A
            Image B
            Label A
            Label B
Pre-coded helper function from CSCI 5722 Quiz 3. 
Used for debugging. 
'''
def plot_step(img_a, img_b, title_a, title_b):
    plt.subplot(121),plt.imshow(img_a, cmap = 'gray')
    plt.title(title_a), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_b, cmap = 'gray')
    plt.title(title_b), plt.xticks([]), plt.yticks([])
    plt.show()

'''
Recursive DFS for hysteresis
Input   -   Thresholded Image
            Hysteresis Output image
            Weak neighbors of previous pixel
            Weak pixel value
            Strong pixel value
'''
def pixelSearch(img, Z, neighbors, weak, strong):
    M,N = Z.shape

    # for each of the weak neighbors
    for n in neighbors:
        # copy over to output as strong
        Z[n[0],n[1]] = strong

        # # collect indices of neighbors 
        nextNeighbors = np.array([[n[0]+1,n[1]],[n[0]+1,n[1]+1],[n[0],n[1]+1],[n[0]-1,n[1]+1],
                                  [n[0]-1,n[1]],[n[0]-1,n[1]-1],[n[0],n[1]-1],[n[0]+1,n[1]-1]])
        
        # identify weak neighbors (if not on the img border)
        dfsN = []
        for n in nextNeighbors:
            if n[0] != M and n[1] != N:
              # if the pixel is weak and not yet copied
              if img[n[0],n[1]] == weak and Z[n[0],n[1]] != strong:
                dfsN.append(n)
        
        # recurse, so long as weak neighbors remain
        if len(dfsN) > 0:
          pixelSearch(img, Z, dfsN, weak, strong)

'''
Apply Blur
Input   -   Grayscale image
Output  -   Blurred image
'''
def noise_reduction(img):
    # apply Gaussian Blur (3x3 kernel size)
    blurred = cv2.GaussianBlur(img,(3,3),0)
    return blurred

'''
Gradient Calculation
Input   -   Blurred image
Output  -   Intensity gradient in x direction
            Intensity gradient in y direction
'''
def gradient(img):
    # Define sobel kernels
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    # Convolve to calculate Gradient
    gradient_x = cv2.filter2D(img, ddepth =-1, kernel=sobelx)
    gradient_y = cv2.filter2D(img, ddepth =-1, kernel=sobely)

    return gradient_x, gradient_y

'''
Normalize gradient magnitudes
Input   -   Image gradient magnitudes
Output  -   Normalized magnitudes
'''
def scale(x):
    normX = ((x - np.min(x))/(np.max(x)-np.min(x))) * 255
    return normX

'''
Apply Non-Maximum Suppression to the image
Inputs  -   Normalized image gradients
            Gradient Angles (rad)
Output  -   Supressed image
'''
def non_max_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32) 

    # convert & scale angles to degrees
    angle = theta * 180.0 / np.pi  # max -> 180, min -> -180
    angle[angle < 0] += 180  # max -> 180, min -> 0

    # loop through all the image pixels and determine if the current pixel is a local maximum or not
    for i in range(M-1):
      for j in range(N-1):
        # collect adjacent pixels, based on gradient angle

        # right-left
        if (angle[i,j] < 36):
          n1 = G[i,j-1]
          n2 = G[i,j+1]
        # upper right, lower left diagonal 
        elif ( 36 <= angle[i,j] < 72):
          n1 = G[i-1,j+1]
          n2 = G[i+1,j-1]
        # up-down
        elif (72 < angle[i,j] <= 108):
          n1 = G[i-1,j]
          n2 = G[i+1,j]
        # lower right, upper left diagonal 
        elif (108 < angle[i,j] <= 144):
          n1 = G[i-1,j-1]
          n2 = G[i+1,j+1]
        # left-right
        elif (144 < angle[i,j] <= 180):
          n1 = G[i,j-1]
          n2 = G[i,j+1]

        # keep if maximum
        if (n1 <= G[i,j] >= n2):
          Z[i,j] = G[i,j]
        else:
          Z[i,j] = 0

    return Z

'''
Keep significant pixels through thresholding
Input   -   Non-max suppressed image
            Low threshold ratio
            High threshold ratio
            New "weak" pixel value (pass low threshold)
            New "strong" pixel value (pass high threshold)
Output  -   Thresholded image
Low & High ratios determined through trial and error. 
'''
def threshold(img, lowThresholdRatio=0.1, highThresholdRatio=.25, weak=125, strong=255):
    # calc threshold values based on dictated ratios
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

    return Z

'''
Apply hysteresis to strong pixel to propagate strong edges and
  discard weak pixels
Input   -   Thresholded Image
            "Weak" pixel value
            "Strong" pixel value
Output  -   Final Image
'''
def hysteresis(img, weak=125, strong=255):
    M,N = img.shape

    # create empty image of zeros to store final image
    Z = np.zeros([M, N])

    # for each pixel
    for i in range(1, M-1):
      for j in range(1, N-1):
        pixel = img[i,j]

        if (pixel == strong):
          Z[i,j] = strong

          # collect indices of neighbors
          neighbors = np.array([[i+1,j],[i+1,j+1],[i,j+1],[i-1,j+1],
                                [i-1,j],[i-1,j-1],[i,j-1],[i+1,j-1]])
          
          # identify weak neighbors
          dfsN = []
          for n in neighbors:
             if img[n[0],n[1]] == weak:
                dfsN.append(n)

          # initiate recursive search on weak neighbors
          pixelSearch(img, Z, dfsN, weak, strong)

    return Z

# Unused Brute Force Hysteresis, unused. 
def brute_hysteresis(img, weak=25, strong=255):
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

# Structural code from CSCI 5722 Quiz 3
noise_reduced_img = noise_reduction(img)
gradient_x, gradient_y = gradient(noise_reduced_img)
scaled_img = scale(np.hypot(gradient_x, gradient_y))
theta = np.arctan2(gradient_y, gradient_x)
max_img = non_max_suppression(scaled_img, theta) 
thresholded_img = threshold(max_img)
hysteresis_img = hysteresis(thresholded_img)
minVal = 75
maxVal = 100
edges = cv2.Canny(img, minVal, maxVal)


displayResults(colorImg, hysteresis_img, edges)