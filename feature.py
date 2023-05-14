import cv2
import numpy as np
import mahotas as mt

# Load the fundus image
img = cv2.imread('19_left.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply CLAHE contrast enhancement to improve the visibility of features
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# Apply Gaussian blur to reduce noise and smooth the image
gray = cv2.GaussianBlur(gray, (5,5), 0)

# Compute the Canny edges of the image
edges = cv2.Canny(gray, 50, 150)

# Compute the Hu moments of the image
hu_moments = cv2.HuMoments(cv2.moments(edges)).flatten()

# Compute Haralick texture features using the GLCM matrix
glcm = mt.features.haralick(gray)
haralick = np.mean(glcm, axis=0)

# Compute HOG (Histogram of Oriented Gradients) features
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
h = hog.compute(gray)

# Print out the extracted features
print("Hu moments:", hu_moments)
print("Haralick texture features:", haralick)
print("HOG features:", h.flatten())
