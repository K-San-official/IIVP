import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
Methods: In this section, all relevant methods are stored.
"""


def save_image(name, file):
    path = "output/" + name + ".jpg"
    cv2.imwrite(path, file)


def motion_blur_filter(img, a, b):

    (c1, c2, c3) = cv2.split(img)
    c1_new = motion_blur_channel(c1, a, b).astype(np.uint8)
    c2_new = motion_blur_channel(c2, a, b).astype(np.uint8)
    c3_new = motion_blur_channel(c3, a, b).astype(np.uint8)
    return cv2.merge((c1_new, c2_new, c3_new))


def motion_blur_channel(c, a, b):
    height, width = c.shape
    c_fft = np.fft.fft2(c)
    [u, v] = np.mgrid[0:height, 0:width]
    u = 2 * u / height
    v = 2 * v / width
    h = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
    # h = (np.sin(np.pi * y) / np.pi * y) * np.exp(-1j * np.pi * y)
    # h = np.repeat(h[:, :, np.newaxis], 3, axis=2)
    return cv2.normalize(np.abs(np.fft.ifft2(c_fft * h)), None, 0, 255, cv2.NORM_MINMAX)


def blackAndWhite(img, t1, t2, t3):
    """
    Converts a BGR image to a black-and-white image
    :param img:
    :param t1: blue channel threshold
    :param t2: green channel threshold
    :param t3: red channel threshold
    :return:
    """
    # Divide into three channels
    c1 = img[:, :, 0]
    c2 = img[:, :, 1]
    c3 = img[:, :, 2]
    (thresh, bw1) = cv2.threshold(c1, t1, 255, cv2.THRESH_BINARY)
    (thresh, bw2) = cv2.threshold(c2, t2, 255, cv2.THRESH_BINARY)
    (thresh, bw3) = cv2.threshold(c3, t3, 255, cv2.THRESH_BINARY)
    bw = np.maximum(bw1, bw2)
    bw = np.maximum(bw, bw3)
    return bw

def granulometry(img, k_s_start, factor, iterations):
    """
    Calculates frequencies (sizes) of round objects in images
    :param img: input image
    :param k_s_start: starting diameter
    :param factor: increasing factor of the diameter
    :param iterations: number of iterations the diameter is increased
    :return: opening = image after applying iterative steps, result matrix = difference in surface area per radius
    """
    result_matrix = np.zeros([iterations, 2])
    # Calculate initial surface area
    sa = sum(sum(img))
    for i in range(iterations):
        diameter = k_s_start + (i * factor)
        radius = diameter * 0.5
        er_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        di_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        opening = cv2.morphologyEx(img, cv2.MORPH_ERODE, er_kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_DILATE, di_kernel)
        img = opening
        new_sa = sum(sum(img))
        result_matrix[i, 0] = radius
        result_matrix[i, 1] = abs(sa - new_sa)
        sa = new_sa
    return (opening, result_matrix)

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # --- Exercise 1 ---------------------------------------------------------------------------------------------------

    img_3_1 = cv2.imread("img/bird.jpg")
    img_3_1_blurry = motion_blur_filter(img_3_1, 0.08, 0.08)
    save_image("img_3_1_blurry", img_3_1_blurry)

    img_3_1 = cv2.imread("img/bird.jpg")
    img_3_1_blurry = motion_blur_filter(img_3_1, 0.08, 0.08)
    save_image("img_3_1_blurry", img_3_1_blurry)

    # --- Exercise 3 ---------------------------------------------------------------------------------------------------

    """
    # Exercise 3.1
    img_3_1 = cv2.imread("img/oranges.jpg")
    img_3_2 = cv2.imread("img/orangetree.jpg")

    # Pre-processing (black and white)
    img_3_1_bw = blackAndWhite(img_3_1, 100, 100, 100)
    #cv2.imshow("BW1", img_3_1_bw/255)

    img_3_2_bw = blackAndWhite(img_3_2, 255, 255, 200)
    #cv2.imshow("BW2", img_3_2_bw / 255)
    #save_image("img_3_2_bw", img_3_2_bw)
    # Since a few leaves are still marked as white, we need a close operation (Dilation followed by Erosion)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    img_3_2_bw_closing = cv2.morphologyEx(img_3_2_bw, cv2.MORPH_ERODE, erode_kernel)
    img_3_2_bw_closing = cv2.morphologyEx(img_3_2_bw_closing, cv2.MORPH_DILATE, dilate_kernel)
    #cv2.imshow("Closing", img_3_2_bw_closing)
    save_image("img_3_2_closing", img_3_2_bw_closing)

    # Count oranges

    # Image 1
    (count1, hierarchy1) = cv2.findContours(
        img_3_1_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Image 1 oranges count:", len(count1))

    # Image 2 (after closing operation)
    (count2, hierarchy2) = cv2.findContours(
        img_3_2_bw_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Image 2 oranges count:", len(count2))

    # Exercise 3.2
    img_3_3 = cv2.imread("img/lights.jpg")
    img_3_4 = cv2.imread("img/jar.jpg")

    # Convert to greyscale and scale down (otherwise the calculation takes far too long!)
    img_3_3_grey = cv2.cvtColor(img_3_3, cv2.COLOR_BGR2GRAY)
    width = int(img_3_3_grey.shape[1] / 10)
    height = int(img_3_3_grey.shape[0] / 10)
    img_3_3_grey = cv2.resize(img_3_3_grey, (width, height))
    save_image("img_3_3_grey", img_3_3_grey)

    img_3_4_grey = cv2.cvtColor(img_3_4, cv2.COLOR_BGR2GRAY)
    width = int(img_3_4_grey.shape[1] / 10)
    height = int(img_3_4_grey.shape[0] / 10)
    img_3_4_grey = cv2.resize(img_3_4_grey, (width, height))
    save_image("img_3_4_grey", img_3_4_grey)

    # Do the granulometry magic (closing operations)
    (img_3_3_high_contr, img_3_3_freq) = granulometry(img_3_3_grey, 3, 5, 20)
    save_image("img_3_3_high_contr", img_3_3_high_contr)
    print(img_3_3_freq)
    plt.plot(img_3_3_freq[:, 0], img_3_3_freq[:, 1])
    plt.title("img_3_3 light size frequencies")
    plt.show()

    (img_3_4_high_contr, img_3_4_freq) = granulometry(img_3_4_grey, 3, 2, 18)
    save_image("img_3_4_high_contr", img_3_4_high_contr)
    plt.plot(img_3_4_freq[:, 0], img_3_4_freq[:, 1])
    plt.title("img_3_4 light size frequencies")
    plt.show()
    """

    cv2.waitKey(0)
    cv2.destroyAllWindows()
