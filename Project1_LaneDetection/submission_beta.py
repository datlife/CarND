import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import moviepy
import numpy as np
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def gray_image(img):
    """
    Convert color image into gray scale image
    :param img: a color image
    :return: a gray image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def edge_detector(gray_img, gaussian_kernel_size=1, low_threshold=0, high_threshold=10):
    """
    Return possible edges in an image using Canny Transformation

    :param gray_img:             a grayed image
    :param gaussian_kernel_size: a kernel size for gaussian blur [default is 0]
    :param low_threshold:        default is 0
    :param high_threshold:       default is 10

    :return: edges in an image
    """
    # possible to get more contrast before blurred image
    # gray_img = equalize_image(gray_img)

    # Apply gaussian blur
    kernel_size = (gaussian_kernel_size, gaussian_kernel_size)
    blurred_image = cv2.GaussianBlur(gray_img, kernel_size, 0)

    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return canny_edges


def adaptive_equalize_image(img, level):
    """
    Equalize an image
        # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

    :param img:    an gray image
    :param level:  clipLevel
    :return: a equalized image
    """
    clahe = cv2.createCLAHE(clipLimit=level)
    result = clahe.apply(img)
    return result


def find_hough_lines(edge_img, min_votes, min_length, max_gap):
    """
    Return a list of possible Hough Lines in edge_img
    :param edge_img:    a canny edge image
    :param min_votes:   a minimum threshold [votes] to be considered a possible line
    :param min_length:  a minimum length to be considered a line
    :param max_gap:     maximum gap between points that have the same line

    :return:            a list of possible lines

    Notice : a line in openCV contains : 2 vertices (x1,y1) and (x2,y2)
    """
    rho = 1             # perpendicular distance from origin to a line
    theta = np.pi/180   # angle between line and x-axis
    lines = cv2.HoughLinesP(edge_img, rho, theta, min_votes, np.array([]), min_length, max_gap)

    return lines


def draw_points(source_image, h_lines, color=[255, 0, 0], thickness=2):

    line_img = np.copy(source_image)

    for a_line in h_lines:
        for x1, y1, x2, y2 in a_line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img


def draw_lines(source_img, lines, color=[255, 0, 0], thickness=2):
    """
    Get a list of lines from provided lines

    :param source_img : a source image
    :param lines      : a list of lines
    :param color      : color of the line
    :param thickness  : thickness of the line
    :return: an image with lines on it
     # for a_line in lines:
    #     for x1, y1, x2, y2 in a_line:
    #         cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    #
    """
    line_img = np.copy(source_img)
    x_size = line_img.shape[1]

    # Draw line segments on original image
    # line_img = draw_points(source_img, lines, [255, 0, 0], 2)

    pos_slope = [l for l in lines if calculate_slope(l) > 0]
    neg_slope = [l for l in lines if calculate_slope(l) < 0]

    # Calculate median slope

    # Get left points and right points for fitting within that slope /
    right_points = np.reshape(pos_slope, (2 * len(pos_slope), 2))  # pts = ([x1, y1], [x2, y2], [x3, y3].... [xn, yn])
    left_points = np.reshape(neg_slope, (2 * len(neg_slope), 2))

    line_img = fit_line(line_img, right_points)
    line_img = fit_line(line_img, left_points)
    # Poly-fit data using np.polyfit()/ degree 1
    # SAVE THIS ONE TO SMOOTH THE CHANGE
    # new = alp*new+(1-alp)*prev
    # alp is a number between 0 and 1 which tells how much to smooth.

    # line = polyfit_lines((left_points, right_points))
    #
    # # Draw line on that fit
    # xp = np.linspace(0, x_size, x_size)
    # right_line = np.transpose([xp, np.array(line[0](xp))]).astype('int')
    # left_line = np.transpose([xp, np.array(line[1](xp))]).astype('int')
    #
    # # Create left line and right line on image

    # #
    # plt.figure(2)
    # plt.plot(xp, line[0](xp), 'r.', xp, line[1](xp), 'b-', )

    return line_img


def fit_line(img, line):
    """
    Add line to image
    :param img:
    :param line:
    :return:
    """
    cols = img.shape[1]
    [vx, vy, x, y] = cv2.fitLine(line, cv2.DIST_L2, 0, 0.01, 0.01)
    slope = (vy-y)/(vx-x)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(img, (cols - slope, righty), (0, lefty), (255, 0, 0), 10)
    return img


def calculate_slope(line):
    """
    Calculate a slope of a line segment
    :param line:
    :return: slope value
    """
    slope = 0
    for x1, y1, x2, y2 in line:
        slope = (y1 - y2)/(x1 - x2)
    return slope


def polyfit_lines(a_list):
    """
    Polyfit data
    :param a_list:
    :return:
    """
    line = []
    for p in a_list:
        transpose_matrix = np.transpose(p)
        x_points = np.array(transpose_matrix[0])
        y_points = np.array(transpose_matrix[1])
        # Poly fit numpy
        line.append(np.poly1d(np.polyfit(y_points, x_points, 1)))
    return line


def create_region_of_interest(source_img, vertices):
    """
    Filter out not-so-important region in the image

    :param source_img:
    :param vertices:    list of vertices to create a polygon
    :return:
    """
    mask = np.zeros_like(source_img)
    ignore_mask_color = 255

    im_shape = source_img.shape
    xsize = im_shape[0]
    ysize = im_shape[1]

    # Points of polygon
    left_top = (ysize * 0.55, xsize * 0.5)
    right_top = (ysize * 0.5, xsize * 0.5)
    left_bottom = (0, xsize)
    right_bottom = (ysize, xsize)

    vertices = np.array([[left_top, right_top, left_bottom, right_bottom]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(source_img, mask)
    return masked_edges


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    # Adaptive histogram equalization - improved image quality
    equalized_image = adaptive_equalize_image(gray_image(image), 2)

    # Get Region of interest
    roi_image = create_region_of_interest(equalized_image, 0)

    # Find all possible lines in the image
    edges = edge_detector(roi_image, 3, 350, 130)
    h_lines = find_hough_lines(edges, 10, 10, 10)

    # Draw fitted line on the image
    result = draw_lines(image, h_lines, [255, 0, 0], 2)

    return result

if __name__ == "__main__":

    image = mpimg.imread('./test_images/solidYellowCurve2.jpg')

    print('This image is:', type(image), 'with dimensions:', image.shape)
    # Adaptive histogram equalization - improved image quality
    equalized_image = adaptive_equalize_image(gray_image(image), 1)

    # Get Region of interest
    roi_image = create_region_of_interest(equalized_image, 0)

    # Find all possible lines in the image
    edges = edge_detector(roi_image, 5, 50, 150)
    h_lines = find_hough_lines(edges, 12, 150, 13)

    # Draw fitted line on the image
    fit_line_image = draw_lines(image, h_lines, [255, 0, 0], 3)

    f = plt.figure(1)
    f.add_subplot(121)
    plt.imshow(edges, cmap='gray')
    f.add_subplot(122)
    plt.imshow(fit_line_image, cmap='gray')
    plt.show()

    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)   # NOTE: this function expects color images!!
    # white_clip.write_videofile(white_output, audio=False)
# Fit line using Least Square Method
# left_line = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)    # <--- not working yet
# right_line = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)

# Create a line on image
# l = (left_line, right_line)
# for l in lines:
#     [x1, y1, x2, y2] = l
#     cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
