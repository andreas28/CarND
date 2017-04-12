from calibration import load_calib_data
import cv2
import numpy as np
from thresholding import thresholding
from warp import prepare_coordinates, warp
from lanefinding import initial_sliding_window
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")


    # Load Video
    clip1 = VideoFileClip("project_video.mp4")
    #clip1 = VideoFileClip("harder_challenge_video.mp4")

    g_thres = 40

    for img in clip1.iter_frames():
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
        thres = thresholding(img_undistorted, weight=(0.3,0.7), thres=g_thres)#thres=20)#20)
        src, dst = prepare_coordinates(thres)
        img_warped = warp (thres, src, dst )
        nonzeroes = (np.sum(img_warped) / (255*img_warped.shape[0]*img_warped.shape[1]))

        if (nonzeroes <= 0.07):
            g_thres = g_thres - 2
        if (nonzeroes >= 0.075):
            g_thres = g_thres + 2

        print (nonzeroes)
        print (g_thres)

        hist = initial_sliding_window(img_warped)
        #plt.plot(hist)
        plt.show()

        cv2.imshow("org", img_undistorted)
        cv2.imshow("warp", img_warped)
        cv2.imshow("thres", thres)
        cv2.waitKey(10)


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
