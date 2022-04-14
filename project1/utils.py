import math
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
