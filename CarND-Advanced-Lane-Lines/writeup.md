
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/distorted_undistorted.jpg "Distorted -> Undistorted"
[image2]: ./output_images/threshold.png "Threshold"
[image3]: ./output_images/warp.png "Warp"
[image4]: ./output_images/histogram.jpg "Histogram"
[image5]: ./output_images/initial_search.jpg "Initial Search"
[image_undis]: ./output_images/distorted_undistorted2.jpg "Undistorted"



## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / R EADME that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is it :)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step can be found in `calibration.py. First object and image points are created, then for all calibration board images the same following steps are taken:

```sh
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

# If found, add object points, image points
if ret == True:
	objpoints.append(objp)
	imgpoints.append(corners)

```

After collecting all objects and image points, the calibration step is run using

```sh
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

```

The camera matrix mtx and distortion parameters dist are saved to a pickle file, so that the calibration step only has to be run once. 


The image shows an undistorted chessboard image on the left and after undistortion using the calculated parameters on the right:

![alt text][image1]




### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

First the calibration parameters need to be loaded from the pickle file:

```sh
# Load camera matrix and distortion parameters from pickle file
calib_mtx, calib_dist = load_calib_data("calib_data.p")
```

Then for each new image the camera matrix and distortion parameters are used to undistort the image:

```sh
img_undistorted = cv2.undistort(img, calib_mtx, calib_dist, None, calib_mtx)
```

On the left side an undistorted image from the test images is shown, on the right side its undistorted result. There is hardly a difference visible, except in the lower left and right edges.
![alt text][image_undis]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Thresholding is provided in `thresholding.py` in the function `thresholding(image, weight=(0.4, 0.6), thres=40)`. The function uses two methods for thresholding. A filtering of the grayscale image with a horizontal sobel filter of size 7 and the saturation channel of the image in HLS color space. 
The two results are then combind to one image using a weight of 0.2 for the s-channel and 0.8 for the sobel filtered image.

```sh
weight = cv2.addWeighted(s_channel, weight[0], scaled_sobel, weight[1], 0)
```

The resulting image is then thresholded using an pixel-adaptive threshold value. For each frame in the warped image, the number of white pixels is counted. If the number of pixels is above a certain value (and therefore it's likely that too much non-lane pixels are shown and it will be harder to find the correct lane pixels) the threshold is increased for the next frame. If it is below a certain value, the threshold is decreased, because it's likely that useful lane pixels were cut out of the warped image. 

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Warping functionality is provided in `warp.py` in the function `prepare_perspective_transform`, which provides the transformation matrix and it's inverse. The source and destination points in the code are:

```
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

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 650        | 0, 720        | 
| 550, 450      | 0, 0          |
| 730, 720      | 1280, 0       |
| 1280, 650     | 1280, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

As proposed in the course, I first calculated the histogram of the lower quater of the image, which gave a good indication of the position of the left and right lane. The histogram is split in the center and the highest value on each side is then considered to be the position of each lane. 

![alt text][image4]

From these positions a windows search like the one proposed in the course is conducted. Within each window the non zero pixels are considered to be lane pixels. The horizontal mean of those pixels is then used as the center of the next window above the current one until the top of the image.
Each of those lane pixels are collected and then used to fit a 2nd degree polynom on it - the lane lines.

![alt text][image5]

Once run, the following frames don't need to perform a new histogram and window search. Now the lane pixels are only considered to be within a certain margin around the previously calculated polynoms. Once found, new polynoms are calculated.

I run a plausability check after each polynom calculation. If the new found polynom differs too much from the one found in the previous frame, it is likely that the new one is wrong e.g. due to changed image conditions. In this case the previous calculated polynom is taken as lane polynom. This can only be done for a certain number of frames, if after 50 frames the new calculated polynom still differs too much, a new initial histogram and window search is performed.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

