import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
from classifier import get_images

# Read in our vehicles and non-vehicles
images = glob.glob('/home/andreas/work/CarND/CarND-Vehicle-Detection-master/test_images/*.jpg')
cars = []
notcars = []

cars = get_images("cars", 5)
notcars = get_images("non_cars", 5)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

    if vis == True:
        features, hog_image1 = hog(img[:,:,0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        features, hog_image2 = hog(img[:,:,1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        features, hog_image3 = hog(img[:,:,2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image1, hog_image2, hog_image3
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features

# Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image1, hog_image2, hog_image3 = get_hog_features(gray, orient,
                        pix_per_cell, cell_per_block,
                        vis=True, feature_vec=False)


# Plot the examples
fig = plt.figure()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(222)
plt.imshow(hog_image1, cmap='gray')
plt.title('HOG Y')
plt.subplot(223)
plt.imshow(hog_image2, cmap='gray')
plt.title('HOG Cr')
plt.subplot(224)
plt.imshow(hog_image3, cmap='gray')
plt.title('HOG Cb')
plt.show()
