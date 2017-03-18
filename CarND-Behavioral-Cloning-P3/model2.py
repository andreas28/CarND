

from keras.models import Sequential, Merge
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

# Read in lines from files
lines = []
with open('./data/data/test_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)

# from sklearn.model_selection import train_test_split
# train_samples, validation_samples = train_test_split(lines, test_size=0.2)
#
# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1:
#         shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]
#             images = []
#             angles = []
#
#             for batch_sample in batch_samples:



images = []
measurements = []
for line in lines:
    for i in range(3):              #all 3 images
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#Center Image
model_center = Sequential()
model_center.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model_center.add(Lambda(lambda x: x / 255.0 - 0.5))
model_center.add(Convolution2D(24,5,5, subsample=(1,1), activation="relu"))

#Left Image
model_left = Sequential()
model_left.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model_left.add(Lambda(lambda x: x / 255.0 - 0.5))
model_left.add(Convolution2D(24,5,5, subsample=(1,1), activation="relu"))

#Right Image
model_right = Sequential()
model_right.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model_right.add(Lambda(lambda x: x / 255.0 - 0.5))
model_right.add(Convolution2D(24,5,5, subsample=(1,1), activation="relu"))


#Nvidia
model = Sequential()
model.add(Merge([model_center, model_left, model_right], mode='concat'))
#model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,5,5, subsample=(2,2), activation="relu"))
#model.add(Convolution2D(64,5,5, subsample=(2,2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
print("Fitting")
model.fit([X_train, X_train, X_train], y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

model.save('model.h5')
exit()

