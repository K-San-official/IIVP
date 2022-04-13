import cv2
import utils


# --- Prepare images ---
bgr_1_1 = cv2.imread("img/img_1_1.jpg")
bgr_1_2 = cv2.imread("img/img_1_2.jpg")

rgb_1_1 = cv2.cvtColor(bgr_1_1, cv2.COLOR_BGR2RGB)
rgb_1_2 = cv2.cvtColor(bgr_1_2, cv2.COLOR_BGR2RGB)

hsv_1_1 = cv2.cvtColor(rgb_1_1, cv2.COLOR_RGB2HSV)
hsv_1_2 = cv2.cvtColor(rgb_1_2, cv2.COLOR_RGB2HSV)

# --- Collection of functions to show images from the GUI ---

# Exercise 1 - Image 1
def show_1_1_original():
    cv2.imshow("Task 1.1 - Image 1 - Original", bgr_1_1)


def show_1_1_rgb():
    cv2.imshow("Task 1.1 - Image 1 - RGB", rgb_1_1)


def show_1_1_rgb_r():
    cv2.imshow("Task 1.1 - Image 1 - R-Channel", rgb_1_1[:, :, 0])


def show_1_1_rgb_g():
    cv2.imshow("Task 1.1 - Image 1 - G-Channel", rgb_1_1[:, :, 1])


def show_1_1_rgb_b():
    cv2.imshow("Task 1.1 - Image 1 - B-Channel", rgb_1_1[:, :, 2])


# Exercise 1 - Image 2
def show_1_2_original():
    cv2.imshow("Task 1.1 - Image 2 - Original", bgr_1_2)


def show_1_2_rgb():
    cv2.imshow("Task 1.1 - Image 2 - RGB", rgb_1_2)


def show_1_2_rgb_r():
    cv2.imshow("Task 1.1 - Image 2 - R-Channel", rgb_1_2[:, :, 0])


def show_1_2_rgb_g():
    cv2.imshow("Task 1.1 - Image 2 - G-Channel", rgb_1_2[:, :, 1])


def show_1_2_rgb_b():
    cv2.imshow("Task 1.1 - Image 2 - B-Channel", rgb_1_2[:, :, 2])


def make_later(self):
    # (1) --- Transformation from RGB to HSV ---------------------------------------------------------------------------


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
