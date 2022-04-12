import math

"""
Algorithms to solve image processing tasks are stored here.
"""


def rgb2hsi(img):
    # Translates an RGB image to an HSI image - Color Spaces (2)
    hsi_img = img
    # Iterate over every pixel
    for j in range(len(hsi_img)):
        for k in range(len(hsi_img[0])):
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
            hsi_img[j, k, :] = [int(h), int(s*255), int(i*255)]
    return hsi_img

def rgb2hsv(img):
    pass
