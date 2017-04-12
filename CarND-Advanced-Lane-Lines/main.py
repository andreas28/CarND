from calibration import load_calib_data
import cv2
import numpy as np
from thresholding import thresholding
from warp import prepare_coordinates, warp
from lanefinding import initial_sliding_window
import matplotlib.pyplot as plt


def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")

    for img_file in images:
        img = cv2.imread(img_file)
        img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
        thres = thresholding(img_undistorted, weight=(0.5,0.5), thres=60)
        src, dst = prepare_coordinates(thres)
        img_warped = warp (thres, src, dst )

        hist = initial_sliding_window(img_warped)
        #plt.plot(hist)
        plt.show()


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
