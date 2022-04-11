import cv2
import utils

"""
Introduction to Image and Video Processing
------------------------------------------
Project 1:  spatial filtering, processing
Date:       04/2022
Author:     Konstantin Sandfort
------------------------------------------
This class gets executed at the start to give results for all exercises in the project
Algorithms for certain calculations are stored in utils.py to make the code more readable.
"""

if __name__ == "__main__":
    # 1 Color Spaces
    rgb_1_1 = cv2.imread("img/img_1_1.jpg", cv2.IMREAD_COLOR)
    rgb_1_2 = cv2.imread("img/img_1_2.jpg", cv2.IMREAD_COLOR)

    # (1) Transformation from RGB to HSV
    hsv_1_1 = cv2.cvtColor(rgb_1_1, cv2.COLOR_RGB2HSV)
    hsv_1_2 = cv2.cvtColor(rgb_1_2, cv2.COLOR_RGB2HSV)

    # Show original image, HSV image and HSV channels
    cv2.imshow("1 - RGB Bright Colours", rgb_1_1)
    cv2.imshow("1 - HSV Bright Colours", hsv_1_1)
    cv2.imshow("1 - H-Channel Bright Colours", hsv_1_1[:, :, 0])
    cv2.imshow("1 - S-Channel Bright Colours", hsv_1_1[:, :, 1])
    cv2.imshow("1 - V-Channel Bright Colours", hsv_1_1[:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("1 - RGB Pale Colours", rgb_1_2)
    cv2.imshow("1 - HSV Pale Colours", hsv_1_2)
    cv2.imshow("2 - H-Channel Bright Colours", hsv_1_2[:, :, 0])
    cv2.imshow("2 - S-Channel Bright Colours", hsv_1_2[:, :, 1])
    cv2.imshow("2 - V-Channel Bright Colours", hsv_1_2[:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # (2)
