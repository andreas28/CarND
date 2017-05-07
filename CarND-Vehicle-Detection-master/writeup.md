
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog.png
[image2]: ./output_images/hog4.png
[image3]: ./output_images/hog5.png
[image4]: ./output_images/hog3.png
[image5]: ./output_images/search_windows.jpg
[image6]: ./output_images/old_new.png
[image7]: ./output_images/old_new2.png
[image8]: ./output_images/old_new3.png
[image9]: ./output_images/final_boxes.png
[image10]: ./output_images/final_heatmap.png
[image11]: ./output_images/final_labels.png
[image12]: ./output_images/final_merged.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the file `classifier.py` in the method `own_main()`. First the method `get_images` randomly chooses the specified number of images from the car and non-car directory. The random picking is important to avoid time-series effects from the images.  

I started by reading in all the `vehicle` and `non-vehicle` images and I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters during the HOG part of the lessons but setteled with the ones suggested but used `YCrCb` and all colors instead of `RGB` because it gave a better performance.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM applying `GridSearchCV` to find the best parameters in my `own_main()` function:

```
	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	print ("Training classifier...")
	parameters = {'kernel': ['linear'], 'C':[0.01, 0.1, 1, 10]}
	svr = svm.SVC()
	clf = GridSearchCV(svr, parameters)
	clf.fit(scaled_X, y)
	print ("Classifier trained...")
	print (clf.best_params_)
	print (clf.best_score_)
	print (clf.cv_results_)

	# Check the score of the SVC
	save_to_pickle(file_classifier, classifer_object, clf.best_estimator_)
```

I also experimented with a radial basis function kernel, but despite having a much longer training time, the results were the same. The grid search converged on a `C` parameter of 0.01, and I verified the accuracy of about 0.98 using a separate test set. I didn't want to use an even smaller `C` value to avoid overfitting. 
I saved the classifier to a pickle file to avoid retraining.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window function is taken from the lessons and is slightly modified implemented in `lesson_functions.py` in the function `find_cars_boxes`. I experimented with the slower version and got good results with a 75% window overlap, which is already implemented in the function. I chose three different scales at three different overlapping areas in the image as displayed here:

![alt text][image5]

The yellow area has the smalles window scale, the orange one the biggest and the green one is in between. This is supposed to match the sizes of the cars in different distances. 
The slower version of the sliding window approach takes about 6.6 seconds per image, the faster one with three overlapping layers takes only 2.6 seconds.

```
#1
cars_boxes1 = find_cars_boxes(draw_image, 370, 500, scale=1, svc=clf, X_scaler=X_scaler, orient=orient,
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 	
			spatial_size=spatial_size, hist_bins=hist_bins)

#2
cars_boxes2 = find_cars_boxes(draw_image, 400, 550, scale=1.5, svc=clf, X_scaler=X_scaler, orient=orient,
                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 	
			spatial_size=spatial_size, hist_bins=hist_bins)

#3
cars_boxes3 = find_cars_boxes(draw_image, 400, 600, scale=2, svc=clf, X_scaler=X_scaler, orient=orient,
                      pix_per_cell=pix_per_cell, cell_per_block =cell_per_block, 
			spatial_size=spatial_size, hist_bins=hist_bins)


# Add heat to each box in box list
heat2 = np.zeros_like(draw_image[:,:,0]).astype(np.float)
heat2 = add_heat(heat2, cars_boxes1)
heat2 = add_heat(heat2, cars_boxes2)
heat2 = add_heat(heat2, cars_boxes3)
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
The left column shows the result of the slower one size version, the right column the faster three window sizes version. The different sizes are colored in blue (smallest), green (middle) and red (biggest) size.

![alt text][image6]

![alt text][image7]

![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and added it to the heatmap of the previous frame in order to a.) get more stable and less wobbly and shaky bounding boxes and b) further reduce the false positives. Then I thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here is the result of the detected bounding boxes:

![alt text][image9]

Here the generated heatmap:

![alt text][image10]

Here the individual blobs returned by the `label()`function:

![alt text][image11]

And finally the resultig overall bounding boxes:

![alt text][image12]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some ideas for improvement:

1. Cars that are further away are not detected, this could maybe improved by choosing smaller windows. 

2. The processing is extremely slow on nowhere near real-timeness. Some approaches for a speed up could be to track already detected cars instead of reclassifying them.

3. The stability of bounding boxes could be improved by taking into account that a car can't just "dissapear", so the box is kept and tracked. 

4. Using an average over two frames for a heat map has the dissadvantage, that the current bounding box is influenced by the "old" frame. With two frames this won't matter much, but with more frame, the box will be drawn behind the car. This could be avoided by predicting the position of the bounding box taking into account its position and movement.

5. Also using more frames will result in bigger, less tight bounding boxes. A different weight or different thresholding for old frames could reduce that. 
 





