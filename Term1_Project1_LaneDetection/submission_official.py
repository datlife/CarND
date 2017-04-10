import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

# Import everything needed to edit/save/watch video clips
import moviepy
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def draw_line_segments(source_image, h_lines, color=[255, 0, 0], thickness=2):
    """
    Draw the line segments to the source images.
    """

    line_img = np.copy(source_image)
    for a_line in h_lines:
        for x1, y1, x2, y2 in a_line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img


def edge_detector(gray_img, gaussian_kernel_size=1, low_threshold=0, high_threshold=10):
    """
    Return possible edges in an image using Canny Transformation
    :param gray_img:             a grayed image
    :param gaussian_kernel_size: a kernel size for gaussian blur [default is 0]
    :param low_threshold:        default is 0
    :param high_threshold:       default is 10
    :return: edges in an image
    """

    # Apply gaussian blur
    kernel_size = (gaussian_kernel_size, gaussian_kernel_size)
    blurred_image = cv2.GaussianBlur(gray_img, kernel_size, 0)

    # Canny Edge Detection
    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    return canny_edges


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

    rho = 1  # perpendicular distance from origin to a line
    theta = np.pi / 180  # angle between line and x-axis

    # Hough Transform Built-in function of OpenCV. Return a line segments in image
    lines = cv2.HoughLinesP(edge_img, rho, theta, min_votes, np.array([]), min_length, max_gap)

    return lines


def region_of_interest(img, vertices):
    """
    Filter out not-so-important region in the image
    :param source_img:
    :param vertices:    list of vertices to create a polygon
    :return:
    """
    mask = np.zeros_like(img)
    ignore_mask_color = 255

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_edges = cv2.bitwise_and(img, mask)
    return masked_edges


def hsv_filter(image):
    """
    Convert image to HSV color space to extract white and yellow color
    :param image:
    :return:
    """
    # http://stackoverflow.com/questions/10948589/choosing-correct-hsv-values-for-opencv-thresholding-with-inranges

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YELLOW_MIN = np.array([100, 100, 150], dtype='uint8')
    YELLOW_MAX =np.array([50, 255, 255], dtype ='uint8')

    mask = cv2.inRange(hsv,  YELLOW_MIN,  YELLOW_MAX)

    #mask = cv2.addWeighted(mask, 1.0, mask2, 1.0, 0.0)
    plt.figure(5)
    plt.imshow(hsv)
    plt.figure(4)
    plt.imshow(mask)

    return mask


def adaptive_equalize_image(img, level):
    """
    Equalize an image - Increase contrast for the image
        # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

    :param img:    an gray image
    :param level:  clipLevel
    :return: a equalized image
    """
    clahe = cv2.createCLAHE(clipLimit=level)
    result = clahe.apply(img)
    return result


def gray_image(img):
    """
    Convert color image into gray scale image
    :param img: a color image
    :return: a gray image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def calculate_slope(line):
    """
    Calculate a slope of a line segment
    :param line:
    :return: slope value
    """
    slope = 0
    for x1, y1, x2, y2 in line:
        slope = (y1 - y2) / (x1 - x2)
    return slope


def update_boundary(line_segment, curr_max, curr_min):
    """
    Update max point and min point of the line
    """
    for x1, y1, x2, y2 in line_segment:
        if y1 < curr_max[1]:
            curr_max = [x1, y1]

        elif y1 > curr_min[1]:
            curr_min = [x1, y1]

        if y2 < curr_max[1]:
            curr_max = [x2, y2]

        elif y2 > curr_min[1]:
            curr_min = [x2, y2]

    return [curr_max, curr_min]

    return (max_point, min_point)


def get_line(slope, points, y_max, yaxis):
    """
    """
    if len(points) > 0:

        # filter out points
        x = np.transpose(points)[0]
        y = np.transpose(points)[1]
        # fit = np.polyfit(x, y, 1)

        # avg_point:
        avg_x = np.mean(x)
        avg_y = np.mean(y)
        intercept = avg_y - slope*avg_x

        # bottom_point
        x_btm = int((yaxis - intercept)/slope)
        y_btm = int(yaxis)

        # top_point
        x_top = int((y_max - intercept)/slope)

        line = np.array([[x_btm, int(y_btm), x_top, y_max]], dtype='uint8')
        return line
    else:
        return np.array([[0, 0, 0, 0]])


def cal_dist(line):
    for x1,y1, x2, y2 in line:
        return (x1-x2)**2 + (y1-y2)**2


def avg_slope(slopes):
    """
    calculate avg slope based on the length of the edge (weight).
    The longer edge, the more it affects the avg slope
    """
    slope = np.transpose(slopes)[0]
    weight = np.transpose(slopes)[1]

    mul_sw = np.sum(np.multiply(slope, weight))
    sum_w = np.sum(weight)

    avg_slope = mul_sw/sum_w

    return avg_slope


def hsv_convert(image):

    # Conver BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = adaptive_equalize_image(l, 3.0)
    equalized = cv2.merge((cl, a, b))

    # Convert to HSV image
    hsv = cv2.cvtColor(equalized, cv2.COLOR_BGR2HSV)
    return hsv

def smooth_slope(new, prev, alpha):
    return alpha * new + (1 - alpha) * prev


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def pipeline(image):
    """
    Main pipeline
    :param image:
    :return:
    """
    im_shape = image.shape
    x_size = im_shape[0]
    y_size = im_shape[1]
    # Points of polygon
    left_top = (y_size * 0.46, x_size * 0.55)
    left_bottom = (0 + y_size * 0.12, x_size)
    right_top = (y_size * 0.52, x_size * 0.55)
    right_bottom = (y_size - y_size * 0.07, x_size)
    vertices = np.array([[left_top, right_top, right_bottom, left_bottom]], dtype=np.int32)

    # hsv = hsv_filter(image)

    # Apply Canny Edge Detection
    canny = edge_detector(image, 7, 50, 125)

    # Mask a Region Of Interest
    edges = region_of_interest(canny, vertices)

    # Hough-Transform
    h_lines = find_hough_lines(edges, 25, 50, 200)

    right_points, left_points, right_slopes, left_slopes = process_lines(h_lines)

    # Calculate Lines
    right_slope = np.median(right_slopes)
    left_slope = np.median(left_slopes)

    # To avoid float infinity
    y_max = x_size * 0.55
    left_line = get_line(left_slope, left_points, y_max, x_size)
    right_line = get_line(right_slope, right_points, y_max, x_size)

    line_image = draw_line_segments(np.copy(image) * 0, [left_line], [255, 0, 255], 20)
    result = weighted_img(image, line_image)

    # Display all line
    plt.figure(1)
    print("Original Image: ")
    plt.imshow(image)

    plt.figure(2)
    print("Canny Edge and Hough Transform:")
    plt.imshow(edges, cmap='gray')
    # color_edges = np.dstack((edges, edges, edges))
    return result


def process_lines(hough_lines):
    """

    :param hough_lines:
    :return:
    """
    # Initial points
    right_points = []
    left_points = []
    # Initial slopes
    right_slopes = []
    left_slopes = []

    for line in hough_lines:
        slope = calculate_slope(line)
        if slope > 0:
            right_slopes.append(slope)
            # how to eliminate bad point
            for x1, y1, x2, y2 in line:
                right_points.append((x1, y1))
                right_points.append((x2, y2))
        if slope < 0:
            left_slopes.append(slope)
            for x1, y1, x2, y2 in line:
                left_points.append((x1, y1))
                left_points.append((x2, y2))

    return right_points, left_points, right_slopes, left_slopes

if __name__ == "__main__":

    image = (mpimg.imread('challenge2.png')*255).astype('uint8')
    plt.imshow(image)

    result = pipeline(image)

    plt.figure(3)
    print("Result:")
    plt.imshow(result)
    plt.show()
