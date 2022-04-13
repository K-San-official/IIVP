import math
import numpy as np

"""
Algorithms to solve image processing tasks are stored here.
"""


def rgb2hsi(img):
    # Translates an RGB image to an HSI image - Color Spaces (2)
    hsi_img = np.zeros(shape=(len(img), len(img[0]), 3))
    # Iterate over every pixel
    for j in range(len(img)):
        for k in range(len(img[0])):
            # Get each component [r, g, b] in a range [0, 1]
            r = float(img[j, k, 0]/255)
            g = float(img[j, k, 1]/255)
            b = float(img[j, k, 2]/255)
            # I - intensity
            i = (r + g + b)/3
            # S - saturation
            s = 0
            if i != 0:
                m = min([r, g, b])
                s = 1 - m/i
            # H - hue
            numerator = 1/2*((r - g) + (r - b))
            denominator = math.sqrt((r - g)**2 + ((r - b)*(g - b)))
            if denominator == 0:  # avoid division by 0
                h = 1
            else:
                h = math.acos(numerator/denominator)
            if b < g:  # if Blue > Green
                h = 360 - h
            # Convert H from radians to degrees
            h = h*180/math.pi

            # assign new value
            hsi_img[j, k, :] = [h/255, s, i]  # OpenCV accepts numpy arrays as decimals in range [0, 1]
    return hsi_img


def rgb2hsv(img):
    # Translates an RGB image to an HSV image - Color Spaces (2)
    hsv_img = np.zeros(shape=(len(img), len(img[0]), 3))
    # Iterate over every pixel
    for j in range(len(hsv_img)):
        for k in range(len(hsv_img[0])):
            r = float(img[j, k, 0] / 255)
            g = float(img[j, k, 1] / 255)
            b = float(img[j, k, 2] / 255)

            # H - hue
            h = 0
            c_max = max([r, g, b])
            c_min = min([r, g, b])
            delta = c_max - c_min
            if delta != 0:
                if r == c_max:
                    h = 60*(((g-b)/delta) % 6)
                elif g == c_max:
                    h = 60 * (((g - b) / delta) % 6)
                elif b == c_max:
                    h = 60 * (((g - b) / delta) % 6)

            # S - saturation
            s = 0
            if c_max != 0:
                s = delta/c_max

            # V - value
            v = c_max

            # assign new value
            hsv_img[j, k, :] = [h/360, s, v]
    return hsv_img
