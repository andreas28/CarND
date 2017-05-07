
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
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
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

I also experimented with an radial basis function kernel, but despite having a much longer training time, the results were the same. The grid search setteled on a `C` parameter of 0.01, and I verified the accuracy of about 0.98 using a separate test set. I didn't want to use an even smaller `C` value to avoid overfitting. 
I saved the classifier to a pickle file to avoid retraining.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window function is taken from the lessons and is slightly modified implemented in `lesson_functions.py` in the function `find_cars_boxes`. I experimented with the slower version and got good results with a 75% window overlap, which is already implemented in the function. I chose three different scales at three different areas in the image as displayed here:

![alt text][image5]

The yellow area has the smalles window scale, the orange one the biggest and the green one is in between. This is supposed to match the sizes of the cars in different distances. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

