from calibration import load_calib_data
import glob
import cv2
import numpy as np


def region_of_interest(img, vertices):

    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image\n",
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def thresholding( image ):

    sobel_kernel = 3

    imshape = image.shape
    lb = (10, imshape[0])                   # left bottom
    lt = (imshape[1]/2.1, imshape[0]/1.70)   # left top
    rt = (imshape[1]/1.9, imshape[0]/1.70)   # right top
    rb = (imshape[1]-10,imshape[0])         # right bottom
    vertices = np.array([[lb, lt, rt, rb]], dtype=np.int32)
    img_roi = region_of_interest(image, vertices)

    img_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = img_HLS[:,:,2]

    # Threshold color channel
    #s_thresh_min = 100
    #s_thresh_max = 255
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    #sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    #sob_thresh_min = 20
    #sob_thresh_max = 100
    #sxbinary = np.zeros_like(scaled_sobel)
    #sxbinary[(sobelx >= sob_thresh_min) & (sobelx <= sob_thresh_max)] = 1
    #sxbinary = np.uint8(255*sxbinary/np.max(sxbinary))
    #cv2.imshow("sobelx", scaled_sobel)


    #combined_binary = np.zeros_like(sxbinary)
    #combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255
    weight = cv2.addWeighted(s_channel, 0.5, scaled_sobel, 0.5, 0)
    ret, th = cv2.threshold(weight, 100, 255, cv2.THRESH_BINARY)

    return th #s_binary * 255


def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")

    # Make a list of test images
    images = glob.glob('test_images/*.jpg')

    for img_file in images:
        img = cv2.imread(img_file)
        img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
        combined = thresholding(img_undistorted)

        cv2.imshow('img_undist', img_undistorted)
        cv2.imshow('combined', combined)

        cv2.waitKey(-1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
