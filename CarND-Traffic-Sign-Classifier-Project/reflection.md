# Traffic Sign Recognition


---

## Overview

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./traffic-signs-data/1.jpg "Traffic Sign 1"
[image2]: ./traffic-signs-data/11.jpg "Traffic Sign 2"
[image3]: ./traffic-signs-data/12.jpg "Traffic Sign 3"
[image4]: ./traffic-signs-data/14.jpg "Traffic Sign 4"
[image5]: ./traffic-signs-data/25.jpg "Traffic Sign 5"
[image6]: ./traffic-signs-data/2.jpg "Traffic Sign 6"
[image7]: ./traffic-signs-data/13.jpg "Traffic Sign 7"
[image8]: ./traffic-signs-data/13_2.jpg "Traffic Sign 8"
[image9]: ./writeup/dataPerClassBar.jpg "Training Data Distribution"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Link to the [IPython notebook file](https://github.com/andreas28/CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

---

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

The code for this step is contained in the **second code cell** of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 by 32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Visualization and training data analysis

The code for this step is contained in the **third code cell** of the IPython notebook.  

Here is an exploratory visualization of the data set. The data in the bar chart shows the percental distribution of training data according to their class. The training data seems to be highly unbalanced which could lead to bad classification performance. 

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Preprocessing the data

The code for this step is contained in the **first code cell** of the IPython notebook.

The only preprocessing step I finally used was normalizing the data (reasons are given in the next section) by divinding it by the maximum intensity value of 255 (uint8) which converts to image data to float values between 0.0 and 1.0.
I tested a normalization with zero mean which lead to values between -1.0 and 1.0 but didn't experience any improvement in the validation accuarcy. 
The normalization step leads to an invariance in exposure differences which helped to improve the validation accuracy from 0.854 to 0.938 for an earlier model I used. 

I sticked to RGB values over greyscale, because in my opinion colors are very important in classifying traffic signs. The color differences between traffic signs are quite small (red, white, black), but considering color, the model might be able to distinguish between the traffic sign and the background. The background often consists of sky (blue color), plants (green color) or maybe buildings (grey, brown) which differs from traffic sign colors. I considered converting the RGB channels to YUV or Lab color space, but due to the promising classification performance I sticked to RGB values.


#### 2. Training, validation and testing data. 

The code for splitting the data into training and validation sets is contained in the ***first code cell*** of the IPython notebook.

The given data was already separated into training (train.p), verification (valid.p) and testing sets (test.p). If not I first would have chosen a test set of about 15%, then used cross validation to separate training and validation sets.

I modified the model before I considered augmenting or balancing the training data. As the chosen model showed very promising result of ~97% on the validation set, no further preprocessing on the training data was done. 

However I considered augmenting the training data by using affine transformations in order to incorparate differences in the viewing angle of traffig signs. 
Also rotating the data by a small degree would probably help to a more robust classifier. 
Additionaly I considered balancing the training data.

#### 3. Final model architecture.

The code for my final model is located in the ***cell 47*** of the ipython notebook. 

I started with the classic LeNet model and modified the number of inputs/outputs between the different layers. Pure LeNet gave me a validation accuracy of 0.874, which seemed to be a good starting point. 
I experimented with dropout layers in the LeNet model and also in my modified model but it gave me equal or slightly worse performance than without a dropout layer (with dropout 0.953, without dropout 0.96). 

I modified the classic LeNet number of outputs per layer to account for the higher number of classification classes. The LeNet model was intended to classify 10 different classes, whereas the Traffic Sign Classifier needs to distinguish between 48 different traffic signs, which is a much more challenging problem. In order to cope with the higher dimensionality I raised the number of output layers/filters from 6 to 24 in the first layer. I chose 24 by simply using a thumb rule (10 classes ~ 6 filters, 48 classes ~ 4*6 filters), this of course is a simplification, but proofed to be a good value to boost the performance. I chose this thumb rule for the following layers (eg. max pooling after conv2 with 64 instead of 16 filters). For the fully connected layers I wanted to reduce the high dimensionality and used the same output sizes as LeNet. In my experience it is better to choose a higher dimensionality in the beginning layers than in the later ones. 
As the size of input images was the same as LeNet (32 by 32) I sticked to the filter sizes for convolution and max pooling. 


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x24 	|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x64    | 
RELU 				|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x64 	|							|
| Fully connected		| input 1600, output 120
RELU					|	
| Fully connected		| input 120, output 84
RELU					|	
| Fully connected		| input 84, output 43 
| Softmax                   | 


#### 4. Model Training

The code for training the model is located in ***cell 48*** of the ipython notebook. 

I raised the size of epochs from 10 to 30 but the results showed that there is no benefit in more epochs than about 20. The batch size in the final model is 32. A smaller batch size seemed to be slightly better and the GPU on the AWS could handle the load easily. I experimented with different learning rates. A higher learning rate of 0.01 seemed to oszillate at a worse performance at higher epochs (as expected). A lower learning rate of 0.0001 didn't show any advantages even with a very high number of epochs (~60). The final learning rate I chose was 0.001, which seems to be a good trade off between fast convergence and accuracy. The optimizer I chose is AdamOptimizer, because it is slightly more advanced than Stochastic Gradient Descent as it adjust the learning rate on the fly, depending on the data.

#### 5. The approach to finding the final model

The code for calculating the accuracy of the model is located in ***cell 48*** of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.974 
* test set accuracy of 0.947

First I experimented on the standard LeNet model, I tried different epoch sizes, batch sizes and learning rates. This however didn't seem to make much difference in the validation accuracy, so I added dropout layers after the first then after the second convolutional layer. But this approach also didn't give better results which indicated that the model itself had to be changed. 

The explanation how I modified the model and what was the reason behind choosing the final hyperparamters are already described in the previous sections. Basically I adapted the LeNet model to account for the higher number of classes to be classified. I didn't add any additional layer as the validation accuracy was already quite good and an additional layer probably would only have made the model computationally more expensive without gaining much more accuracy.

### Test a Model on New Images

#### 1. German Traffic Signs from the Web.

The code for importing the images and resizing them can be found in ***cell 49***

Here are eight German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] 



#### 2. The model's predictions on the web images


The code for making predictions on my final model is located in ***cell 50*** of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   | 
| Right-of-way at the next intersection    | Right-of-way at the next intersection |
| Priority road					| Priority road|
| Stop	      		| Stop					 				|
| Road work			| Road work   							|
| Speed limit (50km/h) | Priority road |
| Yield | No passing for vehicles over 3.5 metric tons |
| Yield | Yield |


The first five images I chose from the web didn't seem to be a problem to the classifier, all signs were classified correctly with an almost 100% confidence for each sign regarding the softmax probabilities.
So I collected an additional two more challenging images. 

The "50 speed sign" image was taken from a very different angle and the bar is visible. It is falsely classified as a yield sign. The different angle of viewing point "destroys" the circular shape of the traffic sign, which is a crucial feature for the classification. The training data has no such image and the layers seem to have learned circular shapes and edges. So this sign apparently looks more like a rectangular shape to the classifier. 

I chose a yield image of non-quadratic size to see how well the classifier generalizes to wrong aspect ratios. After reading in the images I resize them to the expected network input size of 32x32, which shrinks the yield sign horizontaly. Maybe the higher slope of the yield sign got the classifier to  predict the "No passing for vehicles over 3.5 metric tons" sign, because of the shape of the truck. 
A correct aspect ratio on the yield image leads to a correct classification.

The overall classification on the initial 5 images was 100%, for all 8 images it's 75%.

#### 3. Softmax probabilities

The code for making predictions on my final model is located in ***cell 50*** of the Ipython notebook.

For all the images the classifier is very certain about it's class prediction. Six images respond with 1.000e+00 certainty on the first predicted class, two with 9.99e-01. 
The softmax top5 response of the two falsely classified images don't show the correct class.

This result leads to the assumption that the model is overfitted on the traffic sign database which contains nicely cut-out images of roughly the same viewing angle. Although the model perfectly predicts the easy web images, it fails to predict the challenging ones and even fails in returning one of the correct classes in the softmax probabilities.

The next step I would consider is augmenting the data to gain a more robust classifier. This probably would lead in a lower validation accuary, as the data is more challenging, but the final model would be more applicable in real-world scenarios.



