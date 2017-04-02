from calibration import load_calib_data
import glob
import cv2


def thresholding( image ):

    img_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #img_Lab = img_Lab[:,:,2]



    return img_Lab


def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")

    # Make a list of test images
    images = glob.glob('test_images/*.jpg')

    for img_file in images:
        img = cv2.imread(img_file)
        img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
        img = thresholding(img_undistorted)

        cv2.imshow('img_undist', img_undistorted)
        cv2.imshow('img0', img[:,:,0])
        cv2.imshow('img1', img[:,:,1])
        cv2.imshow('img2', img[:,:,2])
        cv2.waitKey(-1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
