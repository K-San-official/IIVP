import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt


# --- Prepare images ---
bgr_1_1 = cv2.imread("img/img_1_1.jpg")
bgr_1_2 = cv2.imread("img/img_1_2.jpg")

rgb_1_1 = cv2.cvtColor(bgr_1_1, cv2.COLOR_BGR2RGB)
rgb_1_2 = cv2.cvtColor(bgr_1_2, cv2.COLOR_BGR2RGB)

hsv_1_1 = cv2.cvtColor(rgb_1_1, cv2.COLOR_RGB2HSV)
hsv_1_2 = cv2.cvtColor(rgb_1_2, cv2.COLOR_RGB2HSV)

bgr_2_1 = cv2.imread("img/img_2_1.jpg")
bgr_2_2 = cv2.imread("img/img_2_2.jpg")

gr_2_1 = cv2.cvtColor(bgr_2_1, cv2.COLOR_BGR2GRAY)
gr_2_2 = cv2.cvtColor(bgr_2_2, cv2.COLOR_BGR2GRAY)

bgr_3_1 = cv2.imread("img/img_3_1.jpg")
bgr_3_2 = cv2.imread("img/img_3_2.jpg")

bgr_4 = cv2.imread("img/img_4.jpg")
bgr_4_t = cv2.imread("img/img_4_t.jpg")
#
"""
--- Collection of functions to show images from the GUI ----------------------------------------------------------------
Functions follow the following naming convention:
show_x_y_z_type_channel, where
x = task number
y = task number suffix
z = image number
type = colour coding
channel = channel name
example: show_1_1_1_rgb_r() means to show the R-channel in RGB space of the first image of task 1.1
"""


# --- Exercise 1.1 - Image 1 -------------------------------------------------------------------------------------------
def show_1_1_1_original():
    cv2.imshow("Task 1.1 - Image 1 - Original", bgr_1_1)


def show_1_1_1_rgb():
    cv2.imshow("Task 1.1 - Image 1 - RGB", rgb_1_1)


def show_1_1_1_rgb_r():
    cv2.imshow("Task 1.1 - Image 1 - R-Channel", rgb_1_1[:, :, 0])


def show_1_1_1_rgb_g():
    cv2.imshow("Task 1.1 - Image 1 - G-Channel", rgb_1_1[:, :, 1])


def show_1_1_1_rgb_b():
    cv2.imshow("Task 1.1 - Image 1 - B-Channel", rgb_1_1[:, :, 2])


def show_1_1_1_hsv():
    cv2.imshow("Task 1.1 - Image 1 - HSV with inbuilt function", hsv_1_1)


def show_1_1_1_hsv_h():
    cv2.imshow("Task 1.1 - Image 1 - HSV H-Channel", hsv_1_1[:, :, 0])


def show_1_1_1_hsv_s():
    cv2.imshow("Task 1.1 - Image 1 - HSV S-Channel", hsv_1_1[:, :, 1])


def show_1_1_1_hsv_v():
    cv2.imshow("Task 1.1 - Image 1 - HSV V-Channel", hsv_1_1[:, :, 2])


# --- Exercise 1.1 - Image 2 -------------------------------------------------------------------------------------------
def show_1_1_2_original():
    cv2.imshow("Task 1.1 - Image 2 - Original", bgr_1_2)


def show_1_1_2_rgb():
    cv2.imshow("Task 1.1 - Image 2 - RGB", rgb_1_2)


def show_1_1_2_rgb_r():
    cv2.imshow("Task 1.1 - Image 2 - R-Channel", rgb_1_2[:, :, 0])


def show_1_1_2_rgb_g():
    cv2.imshow("Task 1.1 - Image 2 - G-Channel", rgb_1_2[:, :, 1])


def show_1_1_2_rgb_b():
    cv2.imshow("Task 1.1 - Image 2 - B-Channel", rgb_1_2[:, :, 2])


def show_1_1_2_hsv():
    cv2.imshow("Task 1.1 - Image 2 - HSV with inbuilt function", hsv_1_2)


def show_1_1_2_hsv_h():
    cv2.imshow("Task 1.1 - Image 2 - HSV H-Channel", hsv_1_2[:, :, 0])


def show_1_1_2_hsv_s():
    cv2.imshow("Task 1.1 - Image 2 - HSV S-Channel", hsv_1_2[:, :, 1])


def show_1_1_2_hsv_v():
    cv2.imshow("Task 1.1 - Image 2 - HSV V-Channel", hsv_1_2[:, :, 2])


# --- Exercise 1.2 - Image 1 -------------------------------------------------------------------------------------------
def show_1_2_1_hsi():
    cv2.imshow("Task 1.2 - Image 1 - HSI", utils.rgb2hsi(rgb_1_1))


def show_1_2_1_hsi_h():
    cv2.imshow("Task 1.2 - Image 1 - HSI H-Channel", utils.rgb2hsi(rgb_1_1)[:, :, 0])


def show_1_2_1_hsi_s():
    cv2.imshow("Task 1.2 - Image 1 - HSI S-Channel", utils.rgb2hsi(rgb_1_1)[:, :, 1])


def show_1_2_1_hsi_i():
    cv2.imshow("Task 1.2 - Image 1 - HSI I-Channel", utils.rgb2hsi(rgb_1_1)[:, :, 2])


def show_1_2_1_hsv():
    cv2.imshow("Task 1.2 - Image 1 - HSV", utils.rgb2hsv(rgb_1_1))


def show_1_2_1_hsv_h():
    cv2.imshow("Task 1.2 - Image 1 - HSV H-Channel", utils.rgb2hsv(rgb_1_1)[:, :, 0])


def show_1_2_1_hsv_s():
    cv2.imshow("Task 1.2 - Image 1 - HSV S-Channel", utils.rgb2hsv(rgb_1_1)[:, :, 1])


def show_1_2_1_hsv_v():
    cv2.imshow("Task 1.2 - Image 1 - HSV V-Channel", utils.rgb2hsv(rgb_1_1)[:, :, 2])


# --- Exercise 1.2 - Image 2 -------------------------------------------------------------------------------------------
def show_1_2_2_hsi():
    cv2.imshow("Task 1.2 - Image 2 - HSI", utils.rgb2hsi(rgb_1_2))


def show_1_2_2_hsi_h():
    cv2.imshow("Task 1.2 - Image 2 - HSI H-Channel", utils.rgb2hsi(rgb_1_2)[:, :, 0])


def show_1_2_2_hsi_s():
    cv2.imshow("Task 1.2 - Image 2 - HSI S-Channel", utils.rgb2hsi(rgb_1_2)[:, :, 1])


def show_1_2_2_hsi_i():
    cv2.imshow("Task 1.2 - Image 2 - HSI I-Channel", utils.rgb2hsi(rgb_1_2)[:, :, 2])


def show_1_2_2_hsv():
    cv2.imshow("Task 1.2 - Image 2 - HSV", utils.rgb2hsv(rgb_1_2))


def show_1_2_2_hsv_h():
    cv2.imshow("Task 1.2 - Image 2 - HSV H-Channel", utils.rgb2hsv(rgb_1_2)[:, :, 0])


def show_1_2_2_hsv_s():
    cv2.imshow("Task 1.2 - Image 2 - HSV S-Channel", utils.rgb2hsv(rgb_1_2)[:, :, 1])


def show_1_2_2_hsv_v():
    cv2.imshow("Task 1.2 - Image 2 - HSV V-Channel", utils.rgb2hsv(rgb_1_2)[:, :, 2])


# --- Exercise 2.1 - Image 1 -------------------------------------------------------------------------------------------
def show_2_1_1_original():
    cv2.imshow("Task 2.1 - Image 1 - Original", gr_2_1)


def show_2_1_1_hist():
    hist = cv2.calcHist(gr_2_1, [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Histogram - Image 1 - low contrast")
    plt.show()


# --- Exercise 2.1 - Image 2 -------------------------------------------------------------------------------------------
def show_2_1_2_original():
    cv2.imshow("Task 2.1 - Image 2 - Original", gr_2_2)


def show_2_1_2_hist():
    hist = cv2.calcHist(gr_2_2, [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Histogram - Image 2 - high contrast")
    plt.show()


# --- Exercise 2.2 - Image 1 -------------------------------------------------------------------------------------------
def show_2_2_1_neg():
    cv2.imshow("Task 2.2 - Image 1 - Negative pointwise transform", utils.negative_pointwise_transform(gr_2_1))


# --- Exercise 2.2 - Image 2 -------------------------------------------------------------------------------------------
def show_2_2_2_neg():
    cv2.imshow("Task 2.2 - Image 2 - Negative pointwise transform", utils.negative_pointwise_transform(gr_2_2))


# --- Exercise 2.3 - Image 1 -------------------------------------------------------------------------------------------
def show_2_3_1_hist():
    neg = utils.negative_pointwise_transform(gr_2_1)
    hist = cv2.calcHist(neg, [0], None, [256], [0, 1])
    plt.plot(hist)
    plt.title("Histogram NPT - Image 1 - low contrast")
    plt.show()


# --- Exercise 2.3 - Image 2 -------------------------------------------------------------------------------------------
def show_2_3_2_hist():
    neg = utils.negative_pointwise_transform(gr_2_2)
    hist = cv2.calcHist(neg, [0], None, [256], [0, 1])
    plt.plot(hist)
    plt.title("Histogram NPT - Image 2 - high contrast")
    plt.show()


# --- Exercise 2.4 - Image 1 -------------------------------------------------------------------------------------------
def show_2_4_1_plpt():
    cv2.imshow("Task 2.4 - Image 1 - Higher contrast with Power Law", utils.power_law_pointwise_transform(gr_2_1, 2))


# --- Exercise 2.4 - Image 2 -------------------------------------------------------------------------------------------
def show_2_4_2_plpt():
    cv2.imshow("Task 2.4 - Image 2 - Lower contrast with Power Law", utils.power_law_pointwise_transform(gr_2_2, 0.5))


# --- Exercise 3.1 - Image 1 -------------------------------------------------------------------------------------------
def show_3_1_1_original():
    cv2.imshow("Task 3.1 - Image 1 - Original", bgr_3_1)


def show_3_1_1_polar():
    cv2.imshow("Task 3.1 - Image 1 - Polar", utils.normal_to_polar(bgr_3_1))


# --- Exercise 3.1 - Image 2 -------------------------------------------------------------------------------------------
def show_3_1_2_original():
    cv2.imshow("Task 3.1 - Image 2 - Original", bgr_3_2)


def show_3_1_2_polar():
    cv2.imshow("Task 3.1 - Image 2 - Polar", utils.normal_to_polar(bgr_3_2))


def show_3_2_1_cartoon():
    cv2.imshow("Task 3.2 - Image 1 - Cartoon", utils.cartoonify(bgr_3_1, 0.5, 4))


def show_3_2_2_cartoon():
    cv2.imshow("Task 3.2 - Image 2 - Cartoon", utils.cartoonify(bgr_3_2, 0.5, 4))


def show_4_original():
    cv2.imshow("Task 4 - Original", bgr_4)
    cv2.imshow("Task 4 - Translated", bgr_4_t)

def show_4_fdp():
    cv2.imshow("Task 4 - Original Magnitude spectrum", utils.fft_magnitude(bgr_4))
    cv2.imshow("Task 4 - Translated Magnitude spectrum", utils.fft_magnitude(bgr_4_t))


def show_5_1_pn():
    pass


def show_5_2_fft():
    pass


def show_5_3_pnr():
    pass

