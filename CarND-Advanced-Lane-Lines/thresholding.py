from calibration import load_calib_data
import glob
import cv2
import numpy as np


def thresholding( image ):

    sobel_kernel = 3

    img_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    img_HLS_S = img_HLS[:,:,2]
    ret, img_HLS_S = cv2.threshold(img_HLS_S, 170, 255, cv2.THRESH_BINARY)



    sobelx = cv2.Sobel(img_HLS[:,:,1], cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    sobely = cv2.Sobel(img_HLS[:,:,1], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    sxbinary = np.uint8(255*sxbinary/np.max(sxbinary))



    return img_HLS_S, scaled_sobel


def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")

    # Make a list of test images
    images = glob.glob('test_images/*.jpg')

    for img_file in images:
        img = cv2.imread(img_file)
        img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
        img_HLS_S, sxbinary = thresholding(img_undistorted)

        cv2.imshow('img_undist', img_undistorted)
        cv2.imshow('img_HLS_S', img_HLS_S)
        cv2.imshow('sxbinary', sxbinary)

        cv2.waitKey(-1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
