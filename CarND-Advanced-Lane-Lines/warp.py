from calibration import load_calib_data
import glob
import cv2
import numpy as np
from thresholding import thresholding

def region_of_interest(img):

    imshape = img.shape
    lb = (10, imshape[0])                   # left bottom
    lt = (imshape[1]/2.1, imshape[0]/1.70)   # left top
    rt = (imshape[1]/1.9, imshape[0]/1.70)   # right top
    rb = (imshape[1]-10,imshape[0])         # right bottom
    vertices = np.array([[lb, lt, rt, rb]], dtype=np.int32)

    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image\n",
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def warp( image, M ):

    img_size = ( image.shape[1], image.shape[0]) #1280,720
    return cv2.warpPerspective(image, M, img_size , flags=cv2.INTER_LINEAR)


def prepare_perspective_transform ():

    offset_xt = 550
    offset_xb = 0

    src = np.float32(
        [[0+offset_xb, 650],
         [offset_xt, 450],
         [1280-offset_xt, 450],
         [1280-offset_xb, 650]])

    dst = np.float32(
        [[0,720],
         [0,0],
         [1280,0],
         [1280,720]])

    # offset_bX = 200
    # offset_bY = 60  #720-60=660
    # offset_tM = 80 #720 - 640 = 80
    #
    # lt = [640-offset_tM, 440]
    # rt = [640+offset_tM, 440]
    # lb = [offset_bX, 720-offset_bY]  #x,y
    # rb = [1280-offset_bX, 720-offset_bY]
    #
    # src = np.float32(
    #     [lt,
    #      rt,
    #      lb,
    #      rb])
    #
    # dst = np.float32(
    #     [[offset_bX,0],
    #      [1280-offset_bX,0],
    #      [offset_bX,720],
    #      [1280-offset_bX,720]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return src, dst, M, Minv

def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")

    # Make a list of test images
    images = glob.glob('test_images/*.jpg')

    for img_file in images:
        img = cv2.imread(img_file)
        img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
       # img_roi = region_of_interest(img_undistorted)
        thres = thresholding(img_undistorted)
        src, dst, M, Minv = prepare_perspective_transform()
        img_warped = warp (thres, M)


        cv2.imshow("org", img_undistorted)
        cv2.imshow("warp", img_warped)
        cv2.imshow("thres", thres)
        cv2.waitKey(-1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
