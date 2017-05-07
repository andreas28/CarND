import pickle
from lesson_functions import *
from sklearn.model_selection import train_test_split, GridSearchCV
import glob
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from scipy.ndimage.measurements import label
import time






#load trained classifier file
def load_from_pickle(filename, object_name):
    file = open(filename, "rb")
    data = pickle.load(file)
    object_data = data[object_name]
    file.close()
    return object_data

def save_to_pickle(filename, object_name, object):
    try:
        pickle_data = {}
        pickle_data[object_name] = object
        pickle.dump( pickle_data, open(filename, "wb"))
    except:
        print("Pickle error writing classifier file")


#Return filenames of randomly sampled images in folder
def get_images(object='cars', number=None):
    if object=='cars':
        images = glob.glob("/home/andreas/work/CarND/CarND-Vehicle-Detection-master/data/vehicles/**/*.png", recursive=True)
    elif object=='non_cars':
        images = glob.glob("/home/andreas/work/CarND/CarND-Vehicle-Detection-master/data/non-vehicles/**/*.png", recursive=True)
    else:
        raise ValueError('get_images parameter must be cars or non_cars')

    if number==-1 or number==None or number>len(images):
        return images
    else:
        images = random.sample(images, number)
    return images


def main():
    # Read in cars and notcars
    images = glob.glob('*.jpeg')
    cars = []
    notcars = []
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()

    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    image = mpimg.imread('bbox-example-image.jpg')
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)


def clf_test():
    file_classifier = "/home/andreas/work/CarND/CarND-Vehicle-Detection-master/data/pickle/classifier.p"
    classifer_object = "classifier"
    clf = load_from_pickle(file_classifier, classifer_object)

# Same as find_cars, but returns boxes instead of image
def find_cars_boxes(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)


    # Empty list for boxes
    boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return boxes


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return draw_img


def own_main():

    ###DATA FOR PICKLE
    #Features
    generate_features = False
    file_pickle_directory = "/home/andreas/work/CarND/CarND-Vehicle-Detection-master/data/pickle/"
    file_car_features = file_pickle_directory + "8500_features_car.p"
    file_notcar_features = file_pickle_directory + "8500_features_notcar.p"
    car_object = "cars"
    notcar_object = "notcars"
    #Classifier
    generate_classifier = False
    file_classifier = file_pickle_directory + "8500_classifier.p"
    classifer_object = "classifier"
    #Scaler
    generate_scaler = False
    file_scaler = file_pickle_directory + "8500_scaler.p"
    scaler_object = "scaler"

    cars = get_images("cars", 8500)
    notcars = get_images("non_cars", 8500)

    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [370, 650] # Min and max in y to search in slide_window()

    if generate_features:
        print ("Generating features...")
        car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        print ("Car features generated...")
        notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        print ("Not car features generated...")


        save_to_pickle(file_car_features, car_object, car_features)
        save_to_pickle(file_notcar_features, notcar_object, notcar_features)
    elif generate_classifier or generate_scaler:
        car_features = load_from_pickle(file_car_features, car_object)
        notcar_features = load_from_pickle(file_notcar_features, notcar_object)
        print("Feature files loaded...")

    if generate_scaler or generate_classifier:
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        save_to_pickle(file_scaler, scaler_object, X_scaler)
    else:
        print ("Loading X_scaler...")
        X_scaler = load_from_pickle(file_scaler, scaler_object)



    if generate_classifier:

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        #rand_state = np.random.randint(0, 100)
        #X_train, X_test, y_train, y_test = train_test_split(
        #scaled_X, y, test_size=0.2, random_state=rand_state)

        print ("Training classifier...")
        parameters = {'kernel': ['linear'], 'C':[0.01, 0.1, 1, 10]}
        svr = svm.SVC()
        clf = GridSearchCV(svr, parameters)
        #clf.fit(X_train, y_train)
        clf.fit(scaled_X, y)
        print ("Classifier trained...")
        print (clf.best_params_)
        print (clf.best_score_)
        print (clf.cv_results_)

        # Check the score of the SVC
        #print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
        save_to_pickle(file_classifier, classifer_object, clf.best_estimator_)

    else:
        print ("Loading classifier...")
        clf = load_from_pickle(file_classifier, classifer_object)



    # Make a list of test images
    images = glob.glob('/home/andreas/work/CarND/CarND-Vehicle-Detection-master/test_images/*.jpg')

    for imag in images:
        print ("inside")
        image = mpimg.imread(imag)
        draw_image = np.copy(image)
        draw2_image = np.copy(image)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        if np.max(image) > 1.0:
            image = image.astype(np.float32)/255

        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(96, 96), xy_overlap=(0.75, 0.75))

        t=time.time()
        hot_windows = search_windows(image, windows, clf, X_scaler, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds old method...')


        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)



        # Add heat to each box in box list
        heat = np.zeros_like(window_img[:,:,0]).astype(np.float)
        heat = add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        # Test find cars
        t=time.time()
        #370 650
        #find_cars_img = find_cars(draw2_image, y_start_stop[0], y_start_stop[1], scale=1.5, svc=clf, X_scaler=X_scaler, orient=orient,
        #                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)

        all_boxes = []
        # 1
        cars_boxes1 = find_cars_boxes(draw2_image, 420, 500, scale=1, svc=clf, X_scaler=X_scaler, orient=orient,
                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)
        #find_cars_img = draw_boxes(draw2_image, cars_boxes, color=(0,255,0), thick=6)


        #2
        cars_boxes2 = find_cars_boxes(draw2_image, 400, 550, scale=1.5, svc=clf, X_scaler=X_scaler, orient=orient,
                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)

        #3
        cars_boxes3 = find_cars_boxes(draw2_image, 400, 600, scale=2, svc=clf, X_scaler=X_scaler, orient=orient,
                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)

        window_img2 = draw_boxes(draw2_image, cars_boxes1, color=(0, 0, 255), thick=6)
        window_img2 = draw_boxes(window_img2, cars_boxes2, color=(0, 255, 0), thick=6)
        window_img2 = draw_boxes(window_img2, cars_boxes3, color=(255, 0, 0), thick=6)

        # Add heat to each box in box list
        heat2 = np.zeros_like(window_img[:,:,0]).astype(np.float)
        heat2 = add_heat(heat2, cars_boxes1)
        heat2 = add_heat(heat2, cars_boxes2)
        heat2 = add_heat(heat2, cars_boxes3)

        # Apply threshold to help remove false positives
        heat2 = apply_threshold(heat2,1)

        # Visualize the heatmap when displaying
        heatmap2 = np.clip(heat2, 0, 255)

        # Find final boxes from heatmap using label function
        labels2 = label(heatmap2)
        find_cars_img = draw_labeled_bboxes(np.copy(image), labels2)

        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to find cars...')


        fig = plt.figure()
        plt.subplot(321)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(322)
        plt.imshow(window_img2)
        plt.title('Finddcars')
        plt.subplot(323)
        plt.imshow(window_img)
        plt.title('Org')
        plt.subplot(324)
        plt.imshow(find_cars_img)
        plt.title('FindCars')
        plt.subplot(325)
        plt.imshow(heatmap2, cmap='hot')
        plt.title('Heatmap')
        plt.subplot(326)
        plt.imshow(labels2[0], cmap='gray')
        plt.title('Labels')
        fig.tight_layout()

        plt.show()


if __name__ == '__main__':
    own_main()

