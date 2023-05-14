import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from scipy import stats

# Load the image
image = cv2.imread("19_left.jpg")

# Resize the image to 224x224
image = cv2.resize(image, (224, 224))

# Apply intensity adjustment
C = 1.3
adjusted_image = cv2.multiply(image, np.array([C]))

# Apply bilateral filtering
diameter = 9
sigmaColor = sigmaSpace = 75
filtered_image = cv2.bilateralFilter(adjusted_image, diameter, sigmaColor, sigmaSpace)

# Split the image into its red, green, and blue channels
b, g, r = cv2.split(filtered_image)
# Apply morphological operations on the red channel to highlight the OD
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
tophat = cv2.morphologyEx(r, cv2.MORPH_TOPHAT, kernel)

# Threshold the image to extract the OD
thresh = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Invert the mask to remove the OD
mask = cv2.bitwise_not(closed)
result = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)


segments = slic(result, n_segments=100, compactness=10)

# Visualize superpixel segmentation
fig, ax = plt.subplots()
ax.imshow(mark_boundaries(result, segments))
ax.set_xticks([])
ax.set_yticks([])
plt.show()

green_channel = result[:,:,1]

mean_value = np.mean(green_channel)
median_value = np.median(green_channel)
mode_value = stats.mode(green_channel, axis=None)[0]
max_value = np.max(green_channel)
min_value = np.min(green_channel)

# Print the extracted statistical features
print("Mean pixel value:", mean_value)
print("Median pixel value:", median_value)
print("Mode pixel value:", np.max(mode_value))
print("Maximum pixel value:", max_value)
print("Minimum pixel value:", min_value)

# Display the original and processed images
cv2.imshow("Original Image", image)
cv2.imshow("Processed Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
