# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.jpg "Center Driving"
[image3]: ./examples/recover1.png "Recovering. Drifting to left lane"
[image4]: ./examples/recover2.png "Recovering. Maximum left."
[image5]: ./examples/recover3.png "Recovering. Steering to the right"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA paper on end-to-end learning for self-driving cars.
The convolutional layers are unchanged in filter sizes (3x3 and 5x5) and depths (24,36,48,64). 
Each convolutional layer includes a RELU layer to introduce nonlinearity. 
The second layer uses normalizes the data by applying a Keras lambda function.


#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer after the second convolution with a probability of 25% for setting a unit to 0 in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting.

Line 32:
```sh
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

```

Line 90 and 91:
```sh
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

Line 114:
```sh
model.compile(loss='mse', optimizer='adam')
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Multiple driving strategies were recorded. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a small known model which already proved to be working for similar tasks.

My first step was to use a convolution neural network model similar to the model presented in the NVIDIA paper on end-to-end learning for self-driving cars. I thought this model might be appropriate because the model proved to be working on the exact same problem we were facing in this project. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. At first I wasn't using any augmentation and randomly picked images from the data set. During the training process the training set showed decreasing a mean squared error, whereas the validation set didn't decrease at a same rate (if at all). The difference between training and validation set loss grew bigger with more training steps. This indicated that the model might be overfitting, but without any intuition of how to interpret the MSE loss, I chose to try the trained model in the simulator. The steering angle didn't change at all, not even slightly, and the car just drove straigt off the road. A strong indication for overfitting. 

To combat the overfitting, I tried modifying the model by changing the sampling of the convutional layers and/or adding a max pooling layer. However, this didn't achieve any change in the steering angle prediction, the predicted angle was still a fix value around zero and the car went off the road, again.

Augmenting the data and increasing the offset for left and right images resulted in changing steering angles at least, but the car still went off the road almost immediatley. 

I followed a hint in the forum, which suggested neglecting the center images, because the highly unbalanced amount of near to zero steering angles will result in an overfitted model. So I reduced the amount of images taken from the center camera, which finally got the car steering correctly for some time. The problem seemed to be that it didn't recover very well in curves. I recorded another lap focussing on recovering, which helped to get the car driving one lap almost without driving off track. The situation where the model seemed to have problems seemed to indicate that the model is still overfitting, because sometimes the steering angle seemed to be stuck around zero when the car was at the border of the road. I raised the number of units in the last dense layers and added a dropout layer to the model and neglected any center images which got the car driving around the lap without going off the road.


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

```sh
model = Sequential()
model.add(Cropping2D(cropping=((crop_top,crop_bottom),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(p=0.25))
model.add(Convolution2D(48,5,5, subsample=(1,1), activation="relu"))
model.add(Convolution2D(64,3,3, subsample=(1,1), activation="relu"))
model.add(Convolution2D(64,3,3, subsample=(1,1), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(50))
model.add(Dense(1))
```

Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center after drifting too far to one side of the road. These images show what a recovery looks like starting from driving into a sharp curve, drifting to the left and then steering back into the middle of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I drove the track in the opposite direction in order to generalize the model and avoid fitting it to only one direction.

To augment the data sat, I also flipped images and angles thinking that this would create a larger set of training images to train from but mainly to avoid overfitting the model to left steering angles as suggested during the course. 

One step which prooved to boost the ability of the model the most was uniformly choosing from center, left or right image. The final model discards the center image at all, so the minimal steering angle is the offset for the left and right image.

```sh
camera = np.random.choice(['center', 'left', 'right'], p=[0.0,0.5,0.5])
```

After the collection process, I had about 18000 numbers of data points. I then preprocessed this data by cropping the top of the image by 75 pixels and the bottom by 20. Then I normalized the image by dividing it by 255 and substracting it by 1.0 for a zero mean of the data. 

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 3 to 4 as the validation set error wasn't decreasing anymore. I used an adam optimizer so that manually training the learning rate wasn't necessary.
