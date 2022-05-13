import math

import cv2
import numpy as np

"""
Methods: In this section, all relevant methods are stored.
"""

def save_image(name, file):
    path = "output/" + name + ".jpg"
    cv2.imwrite(path, file)

def motion_blur_noise(fft):
    new_img = fft.copy()
    height, width, channels = fft.shape
    print(height, width, channels)
    #height = int(width/10)
    #width = int(width/10)
    a = 0.0002
    b = 0.0002
    for u in range(0, height):
        print(u)
        for v in range(0, width):
            for c in range(0, channels):
                new_value = (np.sinc((a*u) + (b*v)))*np.exp(-1j*np.pi*((a*u) + (b*v)))
                new_img[u, v, c] = new_value
                #print(new_value)
    magnitude_spectrum1 = (20 * np.log(np.abs(new_img)))*255*255
    cv2.imshow("MS", magnitude_spectrum1)
    return new_img

    #new_img[0:height, 0:width, 0:channels] = np.sinc()*ex

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Read images for the exercises

    img_1_1 = cv2.imread("img/bird.jpg")
    img_1_2 = cv2.imread("img/geese.jpg")


    # --- Exercise 1 ---------------------------------------------------------------------------------------------------

    fft_1_1 = np.fft.fft2(img_1_1)
    fft_1_1_shifted = np.fft.fftshift(fft_1_1)
    blur_filter = motion_blur_noise(fft_1_1_shifted)
    magnitude_spectrum = (20 * np.log(np.abs(fft_1_1_shifted)))/255
    cv2.imshow("Mag", magnitude_spectrum)
    filter_back = np.fft.ifftshift(fft_1_1_shifted)
    img_back = np.fft.ifft2(filter_back)
    img_back = np.real(img_back)/255
    cv2.imshow("Back", img_back)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

