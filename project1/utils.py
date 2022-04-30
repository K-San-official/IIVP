import math

import cv2
import numpy as np
"""
Algorithms to solve image processing tasks are stored here.
"""


def rgb2hsi(img):
    """
    Converts an RGB-image to an HSI-image
    :param img:
    :return:
    """
    print("Please wait, HSI image is getting processed!")
    # Translates an RGB image to an HSI image - Color Spaces (2)
    hsi_img = np.zeros(shape=(len(img), len(img[0]), 3))
    # Iterate over every pixel
    for j in range(len(img)):
        for k in range(len(img[0])):
            hsi_img[j, k, :] = rgb2hsi_pixel(img[j, k])  # OpenCV accepts numpy arrays as decimals in range [0, 1]
    return hsi_img


def rgb2hsi_pixel(colour):
    """
    Converts the value of a pixel in RGB to HSI
    :param colour:
    :return:
    """
    # Get each component [r, g, b] in a range [0, 1]
    r = float(colour[0]/255)
    g = float(colour[1]/255)
    b = float(colour[2]/255)
    # I - intensity
    i = (r + g + b) / 3
    # S - saturation
    s = 0
    if i != 0:
        m = min([r, g, b])
        s = 1 - m / i
    # H - hue
    h = 0
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = math.sqrt((r - g) ** 2 + ((r - b) * (g - b)))
    if denominator != 0:
        theta = (180.0 / math.pi) * math.acos(numerator / denominator)
        if b <= g:
            h = theta
        else:
            h = 360 - theta
    return [h/360, s, i]


def rgb2hsv(img):
    """
    Converts an RGB-image to an HSV-image
    :param img:
    :return:
    """
    print("Please wait, HSV image is getting processed!")
    # Translates an RGB image to an HSV image - Color Spaces (2)
    hsv_img = np.zeros(shape=(len(img), len(img[0]), 3))
    # Iterate over every pixel
    for j in range(len(hsv_img)):
        for k in range(len(hsv_img[0])):
            hsv_img[j, k, :] = rgb2hsv_pixel(img[j, k])
    return hsv_img


def rgb2hsv_pixel(colour):
    """
    Converts the value of a pixel in RGB to HSV
    :param colour:
    :return:
    """
    r = float(colour[0]/255)
    g = float(colour[1]/255)
    b = float(colour[2]/255)
    # H - hue
    h = 0
    c_max = max([r, g, b])
    c_min = min([r, g, b])
    delta = c_max - c_min
    if not (r == g == b):
        if delta != 0:
            if r == c_max:
                h = 60*(g - b)/(c_max - c_min) # (60 * ((g - b) / delta) + 360) % 360
            elif g == c_max:
                h = 120 + (60*(b - r)/(c_max - c_min)) # h = (60 * ((b - r) / delta) + 120) % 360
            elif b == c_max:
                h = 240 + (60*(r - g)/(c_max - c_min)) # h = (60 * ((r - g) / delta) + 240) % 360
    if h < 0:
        h += 360
    # S - saturation
    s = 0
    if c_max != 0:
        s = delta / c_max
    # V - value
    v = c_max
    return [h/255/2, s, v]


def negative_pointwise_transform(img):
    """
    Returns a grayscale image of reversed intensity (negative image)
    :param img:
    :return:
    """
    print("Please wait, negative image is getting processed!")
    neg_img = np.zeros(shape=(len(img), len(img[0])))
    # Iterate over every pixel
    for i in range(len(neg_img)):
        for j in range(len(neg_img[0])):
            # Reverse value and normalize in range [0, 1]
            neg_img[i, j] = 1 - (img[i, j]/255)
    return np.float32(neg_img)


def power_law_pointwise_transform(img, n):
    """
    Increases or decreases the contrast of a grayscale image depending on n
    :param img:
    :param n:
    :return:
    """
    print("Please wait, image is getting processed according to the power law!")
    if n < 0:
        print("Error, n needs to be bigger than 0!")
        return None
    new_img = np.zeros(shape=(len(img), len(img[0])))
    # Iterate over every pixel
    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            # Reverse value and normalize in range [0, 1]
            new_img[i, j] = (img[i, j]/255)**n
    return np.float32(new_img)


def normal_to_polar(img):
    """
    Converts an image into polar form
    :param img:
    :return:
    """
    new_img = np.zeros(shape=(len(img[0]), len(img), 3))
    # For each pixel in the NEW image
    center_x = int(len(img[0])/2)
    center_y = int(len(img)/2)
    for i in range(len(new_img)):
        for j in range(len(new_img[0])):
            angle = j/len(img)*2*math.pi
            x_origin = int(center_x + (i*math.cos(angle)))
            y_origin = int(center_y + (i*math.sin(angle)))
            # check if it is out of the area
            if x_origin < 0 or x_origin > len(img[0]) - 1 or y_origin < 0 or y_origin > len(img) - 1:
                new_img[i, j] = [0, 0, 0]
            else:
                new_img[i, j] = img[y_origin, x_origin]/255

    return np.float32(new_img)


def cartoonify(img, threshold, color_depth):
    """
    Converts an BGR image to a cartoonified version.
    :param img:
    :param threshold: from 0 to 1. The smaller, the more areas are black.
    :param color_depth: number of colours for each channel
    :return:
    """
    print("Please wait while the cartoon image is being processed. This might take a while.")
    # Step 1: Noise reduction using Gaussian Blur (7x7 kernel)
    blurred = gaussian_blur(img)
    cv2.imwrite("output/blurred.jpg", blurred)
    # Step 2: Gradient calculation using Sobel kernels
    gradient = sobel_gradient(blurred)
    cv2.imshow("gradient", gradient)
    cv2.imwrite("output/gradient.jpg", gradient)
    # Step 3: Reduce color range and make black if it is an edge
    new_img = np.zeros(shape=(len(img), len(img[0]), 3))
    for i in range(len(img)):
        for j in range(len(img[0])):
            if gradient[i, j] > threshold:
                # Threshold for black outline
                new_img[i, j] = [0, 0, 0]
            else:
                # Reduce to 2 colours for each channel
                b = float(int((img[i, j, 0]/255) * color_depth)/color_depth)
                g = float(int((img[i, j, 1]/255) * color_depth)/color_depth)
                r = float(int((img[i, j, 2]/255) * color_depth)/color_depth)
                new_img[i, j] = [b, g, r]
    return new_img


def gaussian_blur(img):
    """
    Uses a 7x7 gaussian kernel to make the image blurry and reduce noise.
    :param img:
    :return:
    """
    kernel = np.array([
        [0, 0, 1, 2, 1, 0, 0],
        [0, 3, 13, 22, 13, 3, 0],
        [1, 13, 59, 97, 59, 13, 1],
        [2, 22, 97, 159, 97, 22, 2],
        [1, 13, 59, 97, 59, 13, 1],
        [0, 3, 13, 22, 13, 3, 0],
        [0, 0, 1, 2, 1, 0, 0]], np.float32)/1003
    new_img = cv2.filter2D(img, -1, kernel)
    cv2.imshow("blurred", new_img)
    return new_img


def sobel_gradient(img):
    """
    Applies Sobel kernel (3x3) to an image
    :param img:
    :return:
    """
    k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    k_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    new_img_x = cv2.filter2D(img, -1, k_x)
    new_img_y = cv2.filter2D(img, -1, k_y)
    new_img = cv2.cvtColor((new_img_x + new_img_y), cv2.COLOR_BGR2GRAY)
    return new_img


def fft_magnitude(img):
    """
    Returns the magnitude spectrum of 2D FFT.
    :param img:
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft_img = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shift = np.fft.fftshift(fft_img)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(fft_shift[:, :, 0], fft_shift[:, :, 1]))
    return magnitude_spectrum/255


def add_periodic_noise(img):
    """
    Adds static periodic noise to an image
    :param img:
    :return:
    """
    amplitude = 0.3
    offset = 0  #shifts the intensity of the whole image
    f_x = 1/3
    f_y = 1/4
    img = img/255
    rows, cols = img.shape
    noise_img = np.zeros(shape=(len(img), len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            noise_img[i, j] = 1 - min(1, max(0, (amplitude*(math.sin(2*np.pi*f_x*j) + math.sin(2*np.pi*f_y*i))) + offset))
    cv2.imwrite("output/noise.jpg", noise_img*255)
    new_img = cv2.multiply(img, noise_img)
    return new_img


