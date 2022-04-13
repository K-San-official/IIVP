from tkinter import *
import engine as e

"""
Introduction to Image and Video Processing
------------------------------------------
Project 1:  spatial filtering, processing
Date:       04/2022
Author:     Konstantin Sandfort
"""


class Gui:
    # --- Create GUI instance ---

    root = Tk()
    root.title("IIVP - Project 1 - Konstantin Sandfort")

    # --- Create all elements ---

    # Exercise 1.1
    label_ex1 = Label(root, text="1. Color Spaces")
    label_ex1_i1 = Label(root, text="Image 1 (bright colours)")
    label_ex1_i2 = Label(root, text="Image 2 (pale colours)")
    label_ex1_1 = Label(root, text="(1) - From BGR to RGB and HSV WITH inbuilt functions")

    btn_1_1_i1_original = Button(root, text="Original", command=e.show_1_1_original)
    btn_1_1_i1_rgb = Button(root, text="RGB", command=e.show_1_1_rgb)
    btn_1_1_i1_rgb_r = Button(root, text="R-Channel", command=e.show_1_1_rgb_r)
    btn_1_1_i1_rgb_g = Button(root, text="G-Channel", command=e.show_1_1_rgb_g)
    btn_1_1_i1_rgb_b = Button(root, text="B-Channel", command=e.show_1_1_rgb_b)
    btn_1_1_i1_hsv = Button(root, text="HSV")
    btn_1_1_i1_hsv_h = Button(root, text="H-Channel")
    btn_1_1_i1_hsv_s = Button(root, text="S-Channel")
    btn_1_1_i1_hsv_v = Button(root, text="V-Channel")

    btn_1_1_i2_original = Button(root, text="Original")
    btn_1_1_i2_rgb = Button(root, text="RGB")
    btn_1_1_i2_rgb_r = Button(root, text="R-Channel")
    btn_1_1_i2_rgb_g = Button(root, text="G-Channel")
    btn_1_1_i2_rgb_b = Button(root, text="B-Channel")
    btn_1_1_i2_hsv = Button(root, text="HSV")
    btn_1_1_i2_hsv_h = Button(root, text="H-Channel")
    btn_1_1_i2_hsv_s = Button(root, text="S-Channel")
    btn_1_1_i2_hsv_v = Button(root, text="V-Channel")

    # Exercise 1.2
    label_ex1_2 = Label(root, text="(2) - From RGB to HSI and HSV WITHOUT inbuilt functions")
    btn_1_2_i1_hsi = Button(root, text="HSI")
    btn_1_2_i1_hsi_h = Button(root, text="H-Channel")
    btn_1_2_i1_hsi_s = Button(root, text="S-Channel")
    btn_1_2_i1_hsi_i = Button(root, text="I-Channel")
    btn_1_2_i1_hsv = Button(root, text="HSV")
    btn_1_2_i1_hsv_h = Button(root, text="H-Channel")
    btn_1_2_i1_hsv_s = Button(root, text="S-Channel")
    btn_1_2_i1_hsv_v = Button(root, text="V-Channel")

    btn_1_2_i2_hsi = Button(root, text="HSI")
    btn_1_2_i2_hsi_h = Button(root, text="H-Channel")
    btn_1_2_i2_hsi_s = Button(root, text="S-Channel")
    btn_1_2_i2_hsi_i = Button(root, text="I-Channel")
    btn_1_2_i2_hsv = Button(root, text="HSV")
    btn_1_2_i2_hsv_h = Button(root, text="H-Channel")
    btn_1_2_i2_hsv_s = Button(root, text="S-Channel")
    btn_1_2_i2_hsv_v = Button(root, text="V-Channel")


    def create_GUI(self):
        # --- Assign elements to grid layout

        self.label_ex1.grid(row=0, columnspan=6)
        self.label_ex1_i1.grid(row=1, column=0)
        self.label_ex1_i2.grid(row=1, column=3)
        self.label_ex1_1.grid(row=2, columnspan=2)

        self.btn_1_1_i1_original.grid(row=3, column=0)
        self.btn_1_1_i1_rgb.grid(row=3, column=1)
        self.btn_1_1_i1_rgb_r.grid(row=4, column=1)
        self.btn_1_1_i1_rgb_g.grid(row=5, column=1)
        self.btn_1_1_i1_rgb_b.grid(row=6, column=1)
        self.btn_1_1_i1_hsv.grid(row=3, column=2)
        self.btn_1_1_i1_hsv_h.grid(row=4, column=2)
        self.btn_1_1_i1_hsv_s.grid(row=5, column=2)
        self.btn_1_1_i1_hsv_v.grid(row=6, column=2)

        self.btn_1_1_i2_original.grid(row=3, column=3)
        self.btn_1_1_i2_rgb.grid(row=3, column=4)
        self.btn_1_1_i2_rgb_r.grid(row=4, column=4)
        self.btn_1_1_i2_rgb_g.grid(row=5, column=4)
        self.btn_1_1_i2_rgb_b.grid(row=6, column=4)
        self.btn_1_1_i2_hsv.grid(row=3, column=5)
        self.btn_1_1_i2_hsv_h.grid(row=4, column=5)
        self.btn_1_1_i2_hsv_s.grid(row=5, column=5)
        self.btn_1_1_i2_hsv_v.grid(row=6, column=5)

        self.root.mainloop()


if __name__ == "__main__":
    application = Gui()
    application.create_GUI()
