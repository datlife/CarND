# POLAR representation for lines
# Reason: avoid infinity
# d     : perpendicular distance from line to origin
# theta : angle of the perpendicular line with the X-ASIS
# POLAR LINE: x*cos(theta) + y*sin(theta) = d
# For example: point P(a,b)
#  a*cos(theta) + b*sin(theta) = d is a sinusoid HOUGH TRANSFORMATION: (y-axis: d / x-axis: theta)
import cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Open image
image = (mpimg.imread('exit-ramp.png')*255).astype('uint8')

# Convert to gray scale
image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

# Create Gaussian Smoothed Image, kernel size k (matrix 3:3)
kernel_size = 5
gaussian_blur = cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)

# Canny Edge Detection
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gaussian_blur, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(canny_edges)
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
imshape = image.shape
xsize = imshape[0]
ysize = imshape[1]

# Points of polygon
left_top = (ysize*0.45, xsize*0.40)
right_top = (ysize*0.5, xsize*0.40)
left_bottom = (0, xsize)
right_bottom = (ysize, xsize)

vertices = np.array([[left_top, right_top, left_bottom, right_bottom]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(canny_edges, mask)

x_mask = [left_top[0], right_top[0], right_bottom[0], left_bottom[0], left_top[0]]
y_mask = [left_top[1], right_top[1], right_bottom[1], left_bottom[1], left_top[1]]

plt.figure(3)
plt.imshow(masked_edges, cmap='gray')
plt.plot(x_mask, y_mask, 'b--', lw=3)
# Hough Transformation
# http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlines

rho = 1             # perpendicular distance from origin to line
theta = np.pi/180   # theta angle between line and X-axis
threshold = 12      # Minimum votes to get returned ( =1 now )
min_line_length = 200
max_line_gap = 13
line_image = np.copy(image)*0
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

line_color = (255, 0, 0)
line_thickness = 3
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_thickness)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((canny_edges, canny_edges, canny_edges))


# Blending two images
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

# Original image
f = plt.figure(1)
f.add_subplot(221)
plt.imshow(image)
# Gray image
f.add_subplot(222)
plt.imshow(image_gray, cmap='gray')
# Gaussian Image
f.add_subplot(223)
plt.imshow(canny_edges, cmap='gray')
f.add_subplot(224)
plt.imshow(combo)

# Canny Edge Image

plt.show()
