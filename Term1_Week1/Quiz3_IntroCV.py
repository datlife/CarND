# Do all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Read in the image and convert to grayscale
# Note: in the previous example we were reading a .jpg
# Here we read a .png and convert to 0,255 bytescale

image = (mpimg.imread('exit-ramp.png')*255).astype('uint8')

# crop the image
# image = image[0:image.shape[0]*0.8, 35:image.shape[1]*0.95]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 5 # Must be an odd number (3, 5, 7...)
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# TEST CASE FOR BLUE GRAY
# # Set up grid and test data
# x = 256
# y = 1024
# z = 1024
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Create coordinates
# X, Y, Z = np.meshgrid(np.arange(0, x), np.arange(0, y), np.arange(0, z))
#
# Axes3D.plot_surface(X, Y, Z, blur_gray)


# Define our parameters for Canny and run it
low_threshold = 70
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Apply HoughLines Transformation to find connected vertices

# Display the image
plt.imshow(edges, cmap='Greys_r')
plt.show()
