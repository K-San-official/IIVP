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

    # --- Exercise 1.1 -------------------------------------------------------------------------------------------------
    label_ex1 = Label(root, text="1. Color Spaces", font="Helvetica 18 bold", background="cyan")
    label_ex1_i1 = Label(root, text="Image 1 (bright colours)")
    label_ex1_i2 = Label(root, text="Image 2 (pale colours)")
    label_ex1_1 = Label(root, text="(1) - From BGR to RGB and HSV WITH inbuilt functions",
                        font="Helvetica 12 bold", background="lightgray")

    btn_1_1_i1_original = Button(root, text="Original", command=e.show_1_1_1_original)
    btn_1_1_i1_rgb = Button(root, text="RGB", command=e.show_1_1_1_rgb)
    btn_1_1_i1_rgb_r = Button(root, text="R-Channel", command=e.show_1_1_1_rgb_r)
    btn_1_1_i1_rgb_g = Button(root, text="G-Channel", command=e.show_1_1_1_rgb_g)
    btn_1_1_i1_rgb_b = Button(root, text="B-Channel", command=e.show_1_1_1_rgb_b)
    btn_1_1_i1_hsv = Button(root, text="HSV", command=e.show_1_1_1_hsv)
    btn_1_1_i1_hsv_h = Button(root, text="H-Channel", command=e.show_1_1_1_hsv_h)
    btn_1_1_i1_hsv_s = Button(root, text="S-Channel", command=e.show_1_1_1_hsv_s)
    btn_1_1_i1_hsv_v = Button(root, text="V-Channel", command=e.show_1_1_1_hsv_v)

    btn_1_1_i2_original = Button(root, text="Original", command=e.show_1_1_2_original)
    btn_1_1_i2_rgb = Button(root, text="RGB", command=e.show_1_1_2_rgb)
    btn_1_1_i2_rgb_r = Button(root, text="R-Channel", command=e.show_1_1_2_rgb_r)
    btn_1_1_i2_rgb_g = Button(root, text="G-Channel", command=e.show_1_1_2_rgb_g)
    btn_1_1_i2_rgb_b = Button(root, text="B-Channel", command=e.show_1_1_2_rgb_b)
    btn_1_1_i2_hsv = Button(root, text="HSV", command=e.show_1_1_2_hsv)
    btn_1_1_i2_hsv_h = Button(root, text="H-Channel", command=e.show_1_1_2_hsv_h)
    btn_1_1_i2_hsv_s = Button(root, text="S-Channel", command=e.show_1_1_2_hsv_s)
    btn_1_1_i2_hsv_v = Button(root, text="V-Channel", command=e.show_1_1_2_hsv_v)

    # --- Exercise 1.2 -------------------------------------------------------------------------------------------------
    label_ex1_2 = Label(root, text="(2) - From RGB to HSI and HSV WITHOUT inbuilt functions", font="Helvetica 12 bold",
                        background="lightgray")

    btn_1_2_i1_hsi = Button(root, text="HSI", command=e.show_1_2_1_hsi)
    btn_1_2_i1_hsi_h = Button(root, text="H-Channel", command=e.show_1_2_1_hsi_h)
    btn_1_2_i1_hsi_s = Button(root, text="S-Channel", command=e.show_1_2_1_hsi_s)
    btn_1_2_i1_hsi_i = Button(root, text="I-Channel", command=e.show_1_2_1_hsi_i)
    btn_1_2_i1_hsv = Button(root, text="HSV", command=e.show_1_2_1_hsv)
    btn_1_2_i1_hsv_h = Button(root, text="H-Channel", command=e.show_1_2_1_hsv_h)
    btn_1_2_i1_hsv_s = Button(root, text="S-Channel", command=e.show_1_2_1_hsv_s)
    btn_1_2_i1_hsv_v = Button(root, text="V-Channel", command=e.show_1_2_1_hsv_v)

    btn_1_2_i2_hsi = Button(root, text="HSI", command=e.show_1_2_2_hsi)
    btn_1_2_i2_hsi_h = Button(root, text="H-Channel", command=e.show_1_2_2_hsi_h)
    btn_1_2_i2_hsi_s = Button(root, text="S-Channel", command=e.show_1_2_2_hsi_s)
    btn_1_2_i2_hsi_i = Button(root, text="I-Channel", command=e.show_1_2_2_hsi_i)
    btn_1_2_i2_hsv = Button(root, text="HSV", command=e.show_1_2_2_hsv)
    btn_1_2_i2_hsv_h = Button(root, text="H-Channel", command=e.show_1_2_2_hsv_h)
    btn_1_2_i2_hsv_s = Button(root, text="S-Channel", command=e.show_1_2_2_hsv_s)
    btn_1_2_i2_hsv_v = Button(root, text="V-Channel", command=e.show_1_2_2_hsv_v)

    # --- Exercise 2.1 -------------------------------------------------------------------------------------------------
    label_ex2 = Label(root, text="2. Pointwise transforms, Histogram Equalization", font="Helvetica 18 bold",
                      background="cyan")
    label_ex2_i1 = Label(root, text="Image 1 (low contrast)")
    label_ex2_i2 = Label(root, text="Image 2 (high contrast)")
    label_ex2_1 = Label(root, text="(1) - Histograms", font="Helvetica 12 bold", background="lightgray")

    btn_2_1_i1_original = Button(root, text="Original", command=e.show_2_1_1_original)
    btn_2_1_i1_hist = Button(root, text="Histogram 1", command=e.show_2_1_1_hist)

    btn_2_1_i2_original = Button(root, text="Original", command=e.show_2_1_2_original)
    btn_2_1_i2_hist = Button(root, text="Histogram 2", command=e.show_2_1_2_hist)

    # --- Exercise 2.2 -------------------------------------------------------------------------------------------------
    label_ex2_2 = Label(root, text="(2) - Negative Pointwise Transform", font="Helvetica 12 bold",
                        background="lightgray")

    btn_2_2_i1_npt = Button(root, text="NPT", command=e.show_2_2_1_neg)

    btn_2_2_i2_npt = Button(root, text="NPT", command=e.show_2_2_2_neg)

    # --- Exercise 2.3 -------------------------------------------------------------------------------------------------
    label_ex2_3 = Label(root, text="(3) - Negative Pointwise Transform Histograms", font="Helvetica 12 bold",
                        background="lightgray")

    btn_2_3_i1_hist = Button(root, text="NPT Histogram 1", command=e.show_2_3_1_hist)

    btn_2_3_i2_hist = Button(root, text="NPT Histogram 2", command=e.show_2_3_2_hist)

    # --- Exercise 2.3 -------------------------------------------------------------------------------------------------
    label_ex2_4 = Label(root, text="(4) - Power Law Pointwise Transform", font="Helvetica 12 bold",
                        background="lightgray")

    btn_2_4_i1_plpt = Button(root, text="Power Law PT 1", command=e.show_2_4_1_plpt)

    btn_2_4_i2_plpt = Button(root, text="Power Law PT 2", command=e.show_2_4_2_plpt)

    # --- Exercise 3.1 -------------------------------------------------------------------------------------------------
    label_ex3 = Label(root, text="3. Special Effects", font="Helvetica 18 bold",
                      background="cyan")

    label_ex3_i1 = Label(root, text="Image 1")
    label_ex3_i2 = Label(root, text="Image 2")
    label_ex3_1 = Label(root, text="(1) - Polar coordinates", font="Helvetica 12 bold", background="lightgray")

    btn_3_1_i1_original = Button(text="Original", command=e.show_3_1_1_original)
    btn_3_1_i1_polar = Button(root, text="Polar Image 1", command=e.show_3_1_1_polar)

    btn_3_1_i2_original = Button(text="Original", command=e.show_3_1_2_original)
    btn_3_1_i2_polar = Button(root, text="Polar Image 2", command=e.show_3_1_2_polar)

    # --- Exercise 3.2 -------------------------------------------------------------------------------------------------
    label_ex3_2 = Label(root, text="(2) - Cartoon", font="Helvetica 12 bold", background="lightgray")

    btn_3_2_i1 = Button(root, text="Cartoon Image 1", command=e.show_3_2_1_cartoon)

    btn_3_2_i2 = Button(root, text="Cartoon Image 2", command=e.show_3_2_2_cartoon)

    # --- Exercise 4 ---------------------------------------------------------------------------------------------------
    label_ex4 = Label(root, text="4. Frequency Domain Properties", font="Helvetica 18 bold",
                      background="cyan")

    btn_4_original = Button(root, text="Original", command=e.show_4_original)

    btn_4_fdp = Button(root, text="FDP", command=e.show_4_fdp)

    # --- Exercise 5.1 -------------------------------------------------------------------------------------------------
    label_ex5 = Label(root, text="5. Periodic noise removal", font="Helvetica 18 bold",
                      background="cyan")

    btn_5_original = Button(root, text="Original")

    label_ex5_1 = Label(root, text="(1) - Periodic noise", font="Helvetica 12 bold", background="lightgray")

    btn_5_1_pn = Button(root, text="Noisy Image", command=e.show_5_1_pn)

    # --- Exercise 5.2 -------------------------------------------------------------------------------------------------
    label_ex5_2 = Label(root, text="(2) - 2D FFT noisy Image", font="Helvetica 12 bold", background="lightgray")

    btn_5_2_fft = Button(root, text="Noisy FFT Image", command=e.show_5_2_fft)

    # --- Exercise 5.3 -------------------------------------------------------------------------------------------------
    label_ex5_3 = Label(root, text="(3) - Periodic noise removal", font="Helvetica 12 bold", background="lightgray")

    btn_5_3_pnr = Button(root, text="De-Noised Image", command=e.show_5_3_pnr)

    def create_GUI(self):
        # --- Assign elements to grid layout ---

        # --- Exercise 1.1 ---
        self.label_ex1.grid(row=0, columnspan=6)
        self.label_ex1_i1.grid(row=1, column=0)
        self.label_ex1_i2.grid(row=1, column=3)
        self.label_ex1_1.grid(row=2, columnspan=6)

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

        # --- Exercise 1.2 ---
        self.label_ex1_2.grid(row=7, columnspan=6)

        self.btn_1_2_i1_hsi.grid(row=8, column=0)
        self.btn_1_2_i1_hsi_h.grid(row=9, column=0)
        self.btn_1_2_i1_hsi_s.grid(row=10, column=0)
        self.btn_1_2_i1_hsi_i.grid(row=11, column=0)
        self.btn_1_2_i1_hsv.grid(row=8, column=1)
        self.btn_1_2_i1_hsv_h.grid(row=9, column=1)
        self.btn_1_2_i1_hsv_s.grid(row=10, column=1)
        self.btn_1_2_i1_hsv_v.grid(row=11, column=1)

        self.btn_1_2_i2_hsi.grid(row=8, column=3)
        self.btn_1_2_i2_hsi_h.grid(row=9, column=3)
        self.btn_1_2_i2_hsi_s.grid(row=10, column=3)
        self.btn_1_2_i2_hsi_i.grid(row=11, column=3)
        self.btn_1_2_i2_hsv.grid(row=8, column=4)
        self.btn_1_2_i2_hsv_h.grid(row=9, column=4)
        self.btn_1_2_i2_hsv_s.grid(row=10, column=4)
        self.btn_1_2_i2_hsv_v.grid(row=11, column=4)

        # --- Exercise 2.1 ---
        self.label_ex2.grid(row=12, columnspan=6)
        self.label_ex2_i1.grid(row=13, column=0)
        self.label_ex2_i2.grid(row=13, column=3)
        self.label_ex2_1.grid(row=14, columnspan=6)

        self.btn_2_1_i1_original.grid(row=15, column=0)
        self.btn_2_1_i1_hist.grid(row=15, column=1)

        self.btn_2_1_i2_original.grid(row=15, column=3)
        self.btn_2_1_i2_hist.grid(row=15, column=4)

        # --- Exercise 2.2 ---
        self.label_ex2_2.grid(row=16, columnspan=6)

        self.btn_2_2_i1_npt.grid(row=17, column=0)

        self.btn_2_2_i2_npt.grid(row=17, column=3)

        # --- Exercise 2.3 ---
        self.label_ex2_3.grid(row=18, columnspan=6)

        self.btn_2_3_i1_hist.grid(row=19, column=0)

        self.btn_2_3_i2_hist.grid(row=19, column=3)

        # --- Exercise 2.4 ---
        self.label_ex2_4.grid(row=20, columnspan=6)

        self.btn_2_4_i1_plpt.grid(row=21, column=0)

        self.btn_2_4_i2_plpt.grid(row=21, column=3)

        # --- Exercise 3.1 ---
        self.label_ex3.grid(row=22, columnspan=6)
        self.label_ex3_i1.grid(row=23, column=0)
        self.label_ex3_i2.grid(row=23, column=3)
        self.label_ex3_1.grid(row=24, columnspan=6)

        self.btn_3_1_i1_original.grid(row=25, column=0)
        self.btn_3_1_i1_polar.grid(row=25, column=1)
        self.btn_3_1_i2_original.grid(row=25, column=3)
        self.btn_3_1_i2_polar.grid(row=25, column=4)

        # --- Exercise 3.2 ---
        self.label_ex3_2.grid(row=26, columnspan=6)

        self.btn_3_2_i1.grid(row=27, column=0)
        self.btn_3_2_i2.grid(row=27, column=3)

        # --- Exercise 4 ---
        self.label_ex4.grid(row=28, columnspan=6)

        self.btn_4_original.grid(row=29, column=0)
        self.btn_4_fdp.grid(row=29, column=1)

        # --- Exercise 5.1 ---
        self.label_ex5.grid(row=30, columnspan=6)

        self.btn_5_original.grid(row=31, column=0)

        self.label_ex5_1.grid(row=32, columnspan=6)

        self.btn_5_1_pn.grid(row=33, column=0)

        # --- Exercise 5.2 ---
        self.label_ex5_2.grid(row=34, columnspan=6)

        self.btn_5_2_fft.grid(row=35, column=0)

        # --- Exercise 5.3 ---
        self.label_ex5_3.grid(row=36, columnspan=6)

        self.btn_5_3_pnr.grid(row=37, column=0)

        # Start GUI
        self.root.mainloop()


if __name__ == "__main__":
    application = Gui()
    application.create_GUI()
