import numpy as np
import cv2
import glob
import pickle
#import matplotlib.pyplot as plt
#matplotlib qt


# Load camera matrix and distortion parameters from file
def load_calib_data(filename):
    calib_data_file = open(filename, "rb")
    calib_data = pickle.load(calib_data_file)
    calib_mtx = calib_data["mtx"]
    calib_dist = calib_data["dist"]
    calib_data_file.close()
    return calib_mtx, calib_dist


# Calibrate camera and save camera matrix and distortion parameters to file
def main():

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(10)

    cv2.destroyAllWindows()

    # Test undistortion on an image
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    # Undistort image
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Visualize before after
    show = np.hstack((img, dst))
    cv2.imshow('Before After', show)
    cv2.waitKey(10)
    cv2.imwrite('output_images/distorted_undistorted.jpg',show)

    # Undistort example image
    img = cv2.imread("test_images/straight_lines1.jpg")
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    show = np.hstack((img, dst))
    cv2.imshow('Before After', show)
    cv2.waitKey(10)
    cv2.imwrite('output_images/distorted_undistorted2.jpg',show)

    cv2.destroyAllWindows()

    print (mtx)
    print (dist)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "calib_data.p", "wb" ) )

if __name__ == '__main__':
    main()
