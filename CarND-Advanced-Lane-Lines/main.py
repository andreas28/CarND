from calibration import load_calib_data
import cv2
import numpy as np
from thresholding import thresholding
from warp import prepare_perspective_transform, warp
from lanefinding import initial_sliding_window, search_in_margin
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


def draw_polynom( img, left_fit, right_fit ):

    height = img.shape[0]
    steps = 30
    pixels_per_step = height // steps

    for i in range(steps):
        start = i * pixels_per_step
        stop = start + pixels_per_step

        start_point_l = (int(left_fit[0]*start**2 + left_fit[1]*start + left_fit[2]), start)
        start_point_r = (int(right_fit[0]*start**2 + right_fit[1]*start + right_fit[2]), start)
        end_point_l = (int(left_fit[0]*stop**2 + left_fit[1]*stop + left_fit[2]), stop)
        end_point_r = (int(right_fit[0]*stop**2 + right_fit[1]*stop + right_fit[2]), stop)

        img = cv2.line(img, end_point_l, start_point_l, [255, 0, 0], 10)
        img = cv2.line(img, end_point_r, start_point_r, [255, 0, 0], 10)

    return img

def plausible_values(left_coeffs, prev_left_coeffs, right_coeffs, prev_right_coeffs,
                     left_curvature, prev_left_curvature, right_curvature, prev_right_curvature):
    if prev_left_coeffs == None or prev_right_coeffs == None:
        return True

    diff_left = np.absolute(prev_left_coeffs[1] - left_coeffs[1])
    diff_right = np.absolute(prev_right_coeffs[1] - right_coeffs[1])
    diff_curvature_left = np.absolute(prev_left_curvature - left_curvature)
    diff_curvature_right = np.absolute(prev_right_curvature - right_curvature)
    if diff_left > 0.5 or diff_right > 0.5: #or diff_curvature_left > 200 or diff_curvature_right > 200:
        #print ("Diff Left %d Diff Right %d Diff Left %d Diff Right %d" % (diff_left, diff_right, diff_curvature_left, diff_curvature_right))
        #print (prev_left_coeffs )
        #print (left_coeffs )
        #print (prev_right_coeffs )
        #print (right_coeffs )
        return False
    else:
        return True

def plausible_pixel_pos(coeffs, prev_coeffs, y_positions, diffs_thresholds):

    if prev_coeffs == None:
        return True

    for i in range(0, len(y_positions)):
        pos_now = (int(coeffs[0]*y_positions[i]**2 + coeffs[1]*y_positions[i] + coeffs[2]))
        pos_prev = (int(prev_coeffs[0]*y_positions[i]**2 + prev_coeffs[1]*y_positions[i] + prev_coeffs[2]))
        #print (np.absolute(pos_now - pos_prev))
        if np.absolute(pos_now - pos_prev) > diffs_thresholds[i]:
            return False

    return True

def calculate_center(left_coeffs, right_coeffs):

    # Define conversions in x from pixels space to meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    car_y = 719

    right = int(right_coeffs[0]*car_y**2 + right_coeffs[1]*car_y + right_coeffs[2])
    left = int(left_coeffs[0]*car_y**2 + left_coeffs[1]*car_y + left_coeffs[2])

    diff_right = (right - 640) * xm_per_pix
    diff_left = (640 - left) * xm_per_pix

    return diff_left, diff_right


def write_on_image(img, left_center, right_center, curvature_left, curvature_right):

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255,255,255)
    diff_center = right_center - left_center
    medium_curvature = (curvature_left + curvature_right ) / 2


    cv2.putText(img, 'Car is %.2fm off the center' % (diff_center), (50,50), font, 1, color, 2)
    cv2.putText(img, 'Medium curvature is %.2fm' % (medium_curvature), (50,100), font, 1, color, 2)



def main():

    # Load camera matrix and distortion parameters from pickle file
    calib_mtx, calib_dist = load_calib_data("calib_data.p")


    # Load Video
    #clip1 = VideoFileClip("project_video.mp4")
    #clip1 = clip1.subclip(35)
    clip1 = VideoFileClip("challenge_video.mp4")

    #initial threshold for color/sobel
    g_thres = 40
    left_not_plausible_frames = 100
    right_not_plausible_frames = 100

    #prepare forward and inverse matrix
    src, dst, M, Minv = prepare_perspective_transform()

    #store previous coefficients
    prev_left_coeffs = None
    prev_right_coeffs = None
    prev_left_curvature = None
    prev_right_curvature = None

    for img in clip1.iter_frames():
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
        thres = thresholding(img_undistorted, weight=(0.3,0.7), thres=g_thres)#thres=20)#20)
        img_warped = warp (thres, M)
        nonzeroes = (np.sum(img_warped) / (255*img_warped.shape[0]*img_warped.shape[1]))

        if (nonzeroes <= 0.065):#0.07):
            g_thres = g_thres - 2
        if (nonzeroes >= 0.08):#0.075):
            g_thres = g_thres + 2

        #print (nonzeroes)
        #print (g_thres)

        if prev_right_coeffs == None or prev_left_coeffs == None:
            left_coeffs, right_coeffs, left_curverad, right_curverad = initial_sliding_window(img_warped)
        else:
            left_coeffs, right_coeffs, left_curverad, right_curverad = search_in_margin(img_warped, prev_left_coeffs, prev_right_coeffs)

        #print (left_curverad)
        #print (right_curverad)
        #print("-------")
        #print ("1: %f 2: %f 3: %f" % (right_coeffs[0], right_coeffs[1], right_curverad))
        #print ("1: %f 2: %f 3: %f" % (left_coeffs[0], left_coeffs[1], left_curverad))
        left_center, right_center = calculate_center(left_coeffs, right_coeffs)
        #print ("left_center %f right_center %f" % (left_center, right_center))

        # Check left lane for plausibility
        if (left_not_plausible_frames > 0) and not plausible_pixel_pos(left_coeffs, prev_left_coeffs, y_positions=[0, 700], diffs_thresholds=[100, 100] ):
            left_coeffs = prev_left_coeffs
            left_curverad = prev_left_curvature
            left_not_plausible_frames -= 1
            #print ("not plausible left")
            #cv2.waitKey(-1)
        else:
            left_not_plausible_frames = 100
            prev_left_coeffs = left_coeffs
            prev_left_curvature = left_curverad

        # Check right lane for plausibility
        if (right_not_plausible_frames > 0) and not plausible_pixel_pos(right_coeffs, prev_right_coeffs, y_positions=[0, 700], diffs_thresholds=[100, 100] ):
            right_coeffs = prev_right_coeffs
            right_curverad = prev_right_curvature
            right_not_plausible_frames -= 1
            #print ("not plausible right")
            #cv2.waitKey(-1)
        else:
            left_not_plausible_frames = 100
            prev_right_coeffs = right_coeffs
            prev_right_curvature = right_curverad


        img_black = np.zeros_like(img_warped)
        img_poly = cv2.cvtColor(img_black.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        draw_polynom(img_black, left_coeffs, right_coeffs)

        #img_poly[img_black > 1] = [255,0,0]
        #unwarped_poly = cv2.warpPerspective(img_poly, Minv, (img_undistorted.shape[1], img_undistorted.shape[0]))
        unwarped_poly = cv2.warpPerspective(img_black, Minv, (img_undistorted.shape[1], img_undistorted.shape[0]))

        img_undistorted[unwarped_poly > 1] = [0,0,255]
        #final = cv2.add(unwarped_poly, img_undistorted)

        write_on_image(img_undistorted, left_center, right_center, left_curverad, right_curverad)

        #cv2.imshow("poly", final)


        #plt.plot(hist)
        #plt.show()



        cv2.imshow("org", img_undistorted)
        cv2.imshow("warp", img_warped)
        cv2.imshow("thres", thres)
        cv2.waitKey(10)


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
