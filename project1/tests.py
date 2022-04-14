import utils as u

"""
This class tests the algorithms on correctness.
"""


def rgb_hsv_test(colour):
    result = u.rgb2hsv_pixel(colour)
    # H value needs to be multiplied with 360 to get the value in degrees
    print("RGB -> HSV :", colour, "->", result)


def rgb_hsi_test(colour):
    result = u.rgb2hsi_pixel(colour)
    # H value needs to be multiplied with 360 to get the value in degrees
    print("RGB -> HSI :", colour, "->", result)


if __name__ == "__main__":
    # RGB to HSV tests:
    hsv_test1 = [255, 255, 255]
    rgb_hsv_test(hsv_test1)

    hsv_test2 = [10, 100, 20]
    rgb_hsv_test(hsv_test2)

    # RGB to HSV tests:
    hsi_test1 = [255, 255, 255]
    rgb_hsi_test(hsi_test1)

    hsi_test2 = [10, 100, 20]
    rgb_hsi_test(hsi_test2)


