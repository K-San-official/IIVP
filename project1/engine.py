import cv2
import utils


"""
Introduction to Image and Video Processing
------------------------------------------
Project 1:  spatial filtering, processing
Date:       04/2022
Author:     Konstantin Sandfort
"""

bgr_1_1 = cv2.imread("img/img_1_1.jpg")
bgr_1_2 = cv2.imread("img/img_1_2.jpg")


def show_1_1_original():
    cv2.imshow("Task 1.1 - Image 1 - Original", bgr_1_1)


def make_later(self):
    # Transform to RGB images
    rgb_1_1 = cv2.cvtColor(bgr_1_1, cv2.COLOR_BGR2RGB)
    rgb_1_2 = cv2.cvtColor(bgr_1_2, cv2.COLOR_BGR2RGB)

    # (1) --- Transformation from RGB to HSV ---------------------------------------------------------------------------
    hsv_1_1 = cv2.cvtColor(rgb_1_1, cv2.COLOR_RGB2HSV)
    hsv_1_2 = cv2.cvtColor(rgb_1_2, cv2.COLOR_RGB2HSV)

    # Show original image, HSV image and HSV channels
    cv2.imshow("1_1 - Original", bgr_1_1)
    cv2.imshow("1_1 - RGB image", rgb_1_1)
    cv2.imshow("1_1 - HSV image", hsv_1_1)
    cv2.imshow("1_1 - H-Channel HSV", hsv_1_1[:, :, 0])
    cv2.imshow("1_1 - S-Channel HSV", hsv_1_1[:, :, 1])
    cv2.imshow("1_1 - V-Channel HSV", hsv_1_1[:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("1_2 Original", bgr_1_2)
    cv2.imshow("1_2 RGB image", rgb_1_2)
    cv2.imshow("1_2 HSV image", hsv_1_2)
    cv2.imshow("2 - H-Channel Bright Colours", hsv_1_2[:, :, 0])
    cv2.imshow("2 - S-Channel Bright Colours", hsv_1_2[:, :, 1])
    cv2.imshow("2 - V-Channel Bright Colours", hsv_1_2[:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # (2) --- Transformation from RGB to HSI/HSV manually --------------------------------------------------------------

    # RGB to HSI
    hsi_1_1 = utils.rgb2hsi(rgb_1_1)
    cv2.imshow("1_1 HSI image", hsi_1_1)

    # RGB to HSV
    hsv_1_1 = utils.rgb2hsv(rgb_1_1)
    cv2.imshow("1_1 HSV image", hsv_1_1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # RGB to HSI
    hsi_1_2 = utils.rgb2hsi(rgb_1_2)
    cv2.imshow("1_2 HSI image", hsi_1_2)

    # RGB to HSV
    hsv_1_2 = utils.rgb2hsv(rgb_1_2)
    cv2.imshow("1_2 HSV image", hsv_1_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
