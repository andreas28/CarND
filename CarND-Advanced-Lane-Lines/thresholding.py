from theano.tensor.basic import vertical_stack

from calibration import load_calib_data
import glob
import cv2
import numpy as np


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gray = cv2.resize(gray, (int(gray.shape[1]/3),int(gray.shape[0]/3)))
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output


def thresholding( image, weight=(0.4, 0.6), thres=40 ):

    sobel_kernel = 7



    img_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = img_HLS[:,:,2]


    # Threshold color channel
    #s_thresh_min = 100
    #s_thresh_max = 255
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
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

    #s_channel = cv2.equalizeHist(s_channel)


    weight = cv2.addWeighted(s_channel, weight[0], scaled_sobel, weight[1], 0)
    ret, th = cv2.threshold(weight, thres, 255, cv2.THRESH_BINARY)

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
