

from keras.models import Sequential, Merge
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import csv
import cv2
import numpy as np
from sklearn.utils import shuffle


batch_size = 32#32
nb_epoch = 3#5#10
crop_bottom = 25
crop_top = 75#75 #Crop less to see further into curves
steering_offset = 0.35#0.3#0.2
steering_rate = 1.0#1.0
samples_factor = 1#1#2

# Read in lines from files
lines = []
with open('./data/data_combined/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:

                camera = np.random.choice(['center', 'left', 'right'], p=[0.0,0.5,0.5])#p=[0.1,0.45,0.45])
                flip = np.random.choice(['flip', 'noflip'])
                filename = ""
                angle = float(batch_sample[3])

                if camera == "center":
                    filename = '.'+batch_sample[0].split('CarND-Behavioral-Cloning-P3')[1]
                elif camera == "left":
                    filename = '.'+batch_sample[1].split('CarND-Behavioral-Cloning-P3')[1]
                    angle = (angle * steering_rate) + steering_offset
                elif camera == "right":
                    filename = '.'+batch_sample[2].split('CarND-Behavioral-Cloning-P3')[1]
                    angle = (angle * steering_rate) - steering_offset

                image = cv2.imread(filename)

                # Flip
                if flip == "flip":
                    image = cv2.flip(image,1)
                    angle = -1.0 * angle
                #elif flip == "noflip":

                images.append(image)
                angles.append(angle)

                X_train = np.array(images)
                y_train = np.array(angles)

                if False:
                    print ("\n\n")
                    print (filename)
                    print (flip)
                    print (camera)
                    print(float(batch_sample[3]))
                    print (angle)
                    cv2.imshow("test", image)
                    cv2.waitKey(-1)

                shuffle(X_train, y_train)
                yield (X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



#Nvidia
model = Sequential()
model.add(Cropping2D(cropping=((crop_top,crop_bottom),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(p=0.25))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, subsample=(1,1), activation="relu"))
model.add(Convolution2D(64,3,3, subsample=(1,1), activation="relu"))
model.add(Convolution2D(64,3,3, subsample=(1,1), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(50))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*samples_factor,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*samples_factor,
                    nb_epoch=nb_epoch,
                    verbose=1)

model.save('model.h5')
exit()

