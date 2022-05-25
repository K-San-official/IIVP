import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fftpack
from skimage.util import random_noise
from numpy import pi
from numpy import r_

"""
Methods: In this section, all relevant methods are stored.
"""


def save_image(name, file):
    """
    Saves an image to the designated output folder.
    Small method but makes life a bit easier :).
    :param name:
    :param file:
    :return:
    """
    path = "output/" + name + ".jpg"
    cv2.imwrite(path, file)


def motion_blur_filter(img, a, b):
    """
    For exercise 1.1
    Adds motion blur to an image.
    :param img:
    :param a:
    :param b:
    :return:
    """
    (c1, c2, c3) = cv2.split(img)
    c1_new = motion_blur_channel(c1, a, b)
    c2_new = motion_blur_channel(c2, a, b)
    c3_new = motion_blur_channel(c3, a, b)
    result = cv2.merge((c1_new, c2_new, c3_new))
    return result


def motion_blur_channel(c, a, b):
    """
    For exercise 1.1.
    Adds motion blur to a single channel of an image.
    :param c:
    :param a:
    :param b:
    :return:
    """
    height, width = c.shape
    c_fft = np.fft.fft2(c)
    h = get_h(height, width, a, b)
    result = np.abs(np.fft.ifft2(c_fft * h))
    return result


def h_inverse(img, a, b):
    """
    For exercise 1.2.
    :param img:
    :param a:
    :param b:
    :return:
    """
    (c1, c2, c3) = cv2.split(img)
    c1_new = h_inverse_channel(c1, a, b)
    c2_new = h_inverse_channel(c2, a, b)
    c3_new = h_inverse_channel(c3, a, b)
    return cv2.merge((c1_new, c2_new, c3_new))


def h_inverse_channel(c, a, b):
    """
    For exercise 1.2
    :param c:
    :param a:
    :param b:
    :return:
    """
    height, width = c.shape
    c_fft = np.fft.fft2(c)
    h = get_h(height, width, a, b)
    return np.abs(np.fft.ifft2(c_fft / h))


def get_h(height, width, a, b):
    """
    Produces an motion blur filter mask in the frequency domain.
    :param height:
    :param width:
    :param a:
    :param b:
    :return:
    """
    [u, v] = np.mgrid[-round(height / 2):round(height / 2), -round(width / 2):round(width / 2)]
    u = 2 * u / height
    v = 2 * v / width
    h = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
    return h


def wiener_filter(original, noisy, k_ratio=False, motion_blur=False, alpha=0, beta=0):
    """
    Function for exercise 1.4
    Calculates the result of a wiener filter.
    :param original: original image
    :param noisy: noisy image
    :param k_ratio: Flag to determine if a special ratio should be used
    :param motion_blur:
    :param alpha:
    :param beta:
    :return: de-noised image (3-channel)
    """
    (c1, c2, c3) = cv2.split(original)
    (n1, n2, n3) = cv2.split(noisy)
    c1_new = wiener_filter_channel(c1, n1, k_ratio, motion_blur, alpha, beta)
    c2_new = wiener_filter_channel(c2, n2, k_ratio, motion_blur, alpha, beta)
    c3_new = wiener_filter_channel(c3, n3, k_ratio, motion_blur, alpha, beta)
    return cv2.normalize(cv2.merge((c1_new, c2_new, c3_new)), None, 0, 255, cv2.NORM_MINMAX)


def wiener_filter_channel(o_c, n_c, k_ratio, motion_blur, alpha, beta):
    """
    Calculates the result of a wiener filter (per channel).
    :param o_c: channel of the original image
    :param n_c: channel of the noisy image
    :param k_ratio: Flag to determine if a special ratio should be used
    :param motion_blur:
    :param alpha:
    :param beta:
    :return: de-noised image (1-channel)
    """
    img_fft = np.fft.fftshift(np.fft.fft2(o_c))
    img_power_spectrum = np.abs(img_fft) ** 2
    noise_fft = np.fft.fftshift(np.fft.fft2(n_c))
    noise_power_spectrum = np.abs(noise_fft) ** 2
    k = 1  # Default value for k if no other ratio is specified
    if k_ratio:
        # Calculate k based on the noise/original power spectrum ratio
        k = np.sum(noise_power_spectrum) / np.sum(img_power_spectrum)
    height, width = o_c.shape
    if motion_blur:
        h = get_h(height, width, alpha, beta)  # h(u,v) is deterministic, so we can re-create the degradation function
    else:
        h = get_h(height, width, 0, 0)
    h_squared = h * np.conj(h)
    h_w = (1 / h) * (h_squared / (h_squared + (noise_power_spectrum / img_power_spectrum)))
    img_back_fft = h_w * img_fft
    img_back = np.fft.ifft2(img_back_fft)
    return np.abs(img_back)


def dct_block(img, size=8):
    """
    Discrete Cosine Transfer for an image with a certain block-size.
    Reference: Lab 6 code
    :param img:
    :param size:
    :return:
    """
    img_size = img.shape
    dct = np.zeros(img_size)
    for i in r_[:img_size[0]:8]:
        for j in r_[:img_size[1]:8]:
            dct[i:i + 8, j:j + 8] = dct2(img[i:i + 8, j:j + 8])
    return dct


def dct2(a):
    """
    Discrete Cosine Transfer
    :param a:
    :return:
    """
    return scipy.fftpack.dct(scipy.fftpack.dct(a.T, norm='ortho').T, norm='ortho' )


def idc_block(img, size=8):
    """
    Inverse Discrete Cosine Transfer for an image with a certain block-size.
    :param img:
    :param k:
    :param size:
    :return:
    """
    img_size = img.shape
    img_dct = np.zeros(img_size)
    for i in r_[:img.shape[0]: size]:
        for j in r_[:img.shape[1]: size]:
            img_dct[i:(i + size), j:(j + size)] = idct2(img[i:(i + size), j:(j + size)])
    return img_dct


def idct2(a):
    """
    Inverse Discrete Cosine Transfer
    :param a:
    :return:
    """
    return scipy.fftpack.idct(scipy.fftpack.idct(a.T, norm='ortho').T, norm='ortho')


def k_thresh(img, k, size=8):
    """
    Sets all coefficients in a block to 0 if they are not part of the k-largest.
    :param img:
    :param k:
    :param size:
    :return:
    """
    img_size = img.shape
    img_thresh = np.zeros(img_size)
    for i in r_[:img.shape[0]: size]:
        for j in r_[:img.shape[1]: size]:
            img_thresh[i:(i + size), j:(j + size)] = keep_k_coeff(k, img[i:(i + size), j:(j + size)])
    return img_thresh


def keep_k_coeff(k, block):
    """
    Keeps the k largest coefficients of a dct block and sets the rest to 0.
    :param k:
    :param block:
    :return:
    """
    height, width = block.shape
    # Convert into 1d array
    flat = np.ravel(block)
    keep = np.argpartition(np.abs(flat), -k)[-k:]  # indices to keep
    # Set values to 0 that are not the k-largest
    for i in range(len(flat)):
        if i not in keep:
            flat[i] = 0
    # Convert back to 2d array
    result = flat.reshape(height, width)
    return result


def add_watermark(dct, w):
    """
    Adds a watermark to the dct domain.
    :param dct:
    :param w:
    :return:
    """
    #TODO add
    pass


def add_watermark_one_block(block, w):
    """
    Adds a watermark for one single block
    :param block:
    :param w:
    :return:
    """
    #TODO add
    pass


def black_and_white(img, t1, t2, t3):
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
    return opening, result_matrix

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # # --- Exercise 1 ---------------------------------------------------------------------------------------------------
    # print("Computing exercise 1")
    #
    # factor = 0.08  # alpha and beta value
    #
    # img_1_1 = cv2.imread("img/bird.jpg") / 255
    # img_1_2 = cv2.imread("img/geese.jpg") / 255
    #
    # # --- Exercise 1.1 (Adding blur) -----------------------------------------------------------------------------------
    # print("Computing exercise 1.1")
    #
    # # Just motion blur
    # img_1_1_blurry = motion_blur_filter(img_1_1, factor, factor)
    # save_image("img_1_1_blurry", img_1_1_blurry * 255)
    #
    # img_1_2_blurry = motion_blur_filter(img_1_2, factor, factor)
    # save_image("img_1_2_blurry", img_1_2_blurry * 255)
    #
    # # Motion blur and noise
    # img_1_1_blurry_noisy = random_noise(img_1_1_blurry, "gaussian", mean=0, var=0.002)
    # save_image("img_1_1_blurry_noisy", img_1_1_blurry_noisy * 255)
    #
    # img_1_2_blurry_noisy = random_noise(img_1_2_blurry, "gaussian", mean=0, var=0.002)
    # save_image("img_1_2_blurry_noisy", img_1_2_blurry_noisy * 255)
    #
    # # --- Exercise 1.2 (Removing blur) ---------------------------------------------------------------------------------
    # print("Computing exercise 1.2")
    #
    # # Inverse filter directly after motion blur (1)
    # inverse_1_directly = h_inverse(img_1_1_blurry, factor, factor)
    # save_image("img_1_1_inverse_directly", inverse_1_directly * 255)
    #
    # inverse_2_directly = h_inverse(img_1_2_blurry, factor, factor)
    # save_image("img_1_2_inverse_directly", inverse_2_directly * 255)
    #
    # # Inverse filter after motion blur and added noise (2)
    # inverse_1_after = h_inverse(img_1_1_blurry_noisy, factor, factor)
    # save_image("img_1_1_inverse_after", inverse_1_after * 255)
    #
    # inverse_2_after = h_inverse(img_1_2_blurry_noisy, factor, factor)
    # save_image("img_1_2_inverse_after", inverse_2_after * 255)
    #
    # # Only additive noise added (3)
    # img_1_1_noisy = random_noise(img_1_1, "gaussian", mean=0, var=0.002)
    # img_1_2_noisy = random_noise(img_1_2, "gaussian", mean=0, var=0.002)
    #
    # # Wiener filter with additive noise only (3)
    # img_1_1_wiener_directly = wiener_filter(img_1_1, img_1_1_noisy)
    # save_image("img_1_1_wiener_directly", img_1_1_wiener_directly)
    #
    # # Wiener filter with noise and motion blur (4)
    # img_1_1_blurry = motion_blur_filter(img_1_1, factor, factor)
    # img_1_1_blurry_noisy = random_noise(img_1_1_blurry, "gaussian", mean=0, var=0.002)  # remove later
    #
    # img_1_1_wiener_after = wiener_filter(img_1_1, img_1_1_blurry_noisy, True, True, 0.8, 0.8)
    # save_image("img_1_1_wiener_after", img_1_1_wiener_after)

    # --- Exercise 2 ---------------------------------------------------------------------------------------------------
    print("Computing exercise 2")
    img_2 = cv2.imread("img/img_2.jpg")
    img_2_gr = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    save_image("ex2/img_2_gr", img_2_gr)

    # DCT
    pos = 8 * 29
    img_2_dct = dct_block(img_2_gr)
    save_image("ex2/img_2_dct", np.abs(img_2_dct))

    # Block of the original image
    block_original = cv2.resize(img_2_gr[pos:pos + 8, pos:pos + 8], (400, 400), interpolation=cv2.INTER_NEAREST)
    save_image("ex2/block_original", block_original)

    # Block of the dct image
    block_dct = cv2.resize(img_2_dct[pos:pos + 8, pos:pos + 8], (400, 400), interpolation=cv2.INTER_NEAREST)
    save_image("ex2/block_dct", np.abs(block_dct))

    # # Keep K highest DCT coefficients
    # for k in [4, 8, 16, 32]:
    #     name = "ex2/img_after_thresh_k_" + str(k)
    #     thresh_img = k_thresh(img_2_dct, k)
    #     save_image(name, idc_block(thresh_img))
    #     block_dct_thresh = cv2.resize(thresh_img[pos:pos + 8, pos:pos + 8], (400, 400),
    #                                   interpolation=cv2.INTER_NEAREST)
    #     block_name = "ex2/block_dct_thresh_" + str(k)
    #     save_image(block_name, np.abs(block_dct_thresh))
    #
    # # Generate watermark

    k = 16
    mu = 0
    sd = 1
    a = 0.5  # watermark strength alpha
    w = np.random.normal(mu, sd, k)
    print(w)

    cv2.waitKey()

    # # --- Exercise 3 ---------------------------------------------------------------------------------------------------
    # print("Computing exercise 3")
    #
    # img_3_1 = cv2.imread("img/oranges.jpg")
    # img_3_2 = cv2.imread("img/orangetree.jpg")
    #
    # # Exercise 3.1
    # print("Computing exercise 3.1")
    #
    # # Pre-processing (black and white)
    # img_3_1_bw = black_and_white(img_3_1, 100, 100, 100)
    # img_3_2_bw = black_and_white(img_3_2, 255, 255, 200)
    #
    # save_image("img_3_1_bw", img_3_1_bw)
    # save_image("img_3_2_bw", img_3_2_bw)
    #
    # # Since a few leaves are still marked as white, we need a close operation (Dilation followed by Erosion)
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    # img_3_2_bw_closing = cv2.morphologyEx(img_3_2_bw, cv2.MORPH_ERODE, erode_kernel)
    # img_3_2_bw_closing = cv2.morphologyEx(img_3_2_bw_closing, cv2.MORPH_DILATE, dilate_kernel)
    # save_image("img_3_2_closing", img_3_2_bw_closing)
    #
    # # Count oranges
    #
    # # Image 1
    # (count1, hierarchy1) = cv2.findContours(
    #     img_3_1_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("Image 1 oranges count:", len(count1))
    #
    # # Image 2 (after closing operation)
    # (count2, hierarchy2) = cv2.findContours(
    #     img_3_2_bw_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("Image 2 oranges count:", len(count2))
    #
    # # Exercise 3.2
    # print("Computing exercise 3.2")
    # img_3_3 = cv2.imread("img/lights.jpg")
    # img_3_4 = cv2.imread("img/jar.jpg")
    #
    # # Convert to greyscale and scale down (otherwise the calculation takes far too long!)
    # img_3_3_grey = cv2.cvtColor(img_3_3, cv2.COLOR_BGR2GRAY)
    # width = int(img_3_3_grey.shape[1] / 10)
    # height = int(img_3_3_grey.shape[0] / 10)
    # img_3_3_grey = cv2.resize(img_3_3_grey, (width, height))
    # save_image("img_3_3_grey", img_3_3_grey)
    #
    # img_3_4_grey = cv2.cvtColor(img_3_4, cv2.COLOR_BGR2GRAY)
    # width = int(img_3_4_grey.shape[1] / 10)
    # height = int(img_3_4_grey.shape[0] / 10)
    # img_3_4_grey = cv2.resize(img_3_4_grey, (width, height))
    # save_image("img_3_4_grey", img_3_4_grey)
    #
    # # Do the granulometry magic (closing operations)
    # (img_3_3_high_contr, img_3_3_freq) = granulometry(img_3_3_grey, 3, 5, 20)
    # save_image("img_3_3_high_contr", img_3_3_high_contr)
    # print(img_3_3_freq)
    # plt.plot(img_3_3_freq[:, 0], img_3_3_freq[:, 1])
    # plt.title("img_3_3 light size frequencies")
    # plt.show()
    #
    # (img_3_4_high_contr, img_3_4_freq) = granulometry(img_3_4_grey, 3, 2, 18)
    # save_image("img_3_4_high_contr", img_3_4_high_contr)
    # plt.plot(img_3_4_freq[:, 0], img_3_4_freq[:, 1])
    # plt.title("img_3_4 light size frequencies")
    # plt.show()
