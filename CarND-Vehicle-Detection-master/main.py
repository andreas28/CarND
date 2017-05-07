import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from classifier import load_from_pickle
from classifier import find_cars_boxes
from lesson_functions import *
from scipy.ndimage.measurements import label



def main():
     # Load Video
     clip1 = VideoFileClip("project_video.mp4")
     #clip1 = VideoFileClip("test_video.mp4")
     out_clip = clip1.fl_image(process_image)
     out_clip.write_videofile("out.mp4", audio=False)

file_pickle_directory = "/home/andreas/work/CarND/CarND-Vehicle-Detection-master/data/pickle/"
file_classifier = file_pickle_directory + "8500_classifier.p"
classifer_object = "classifier"
file_scaler = file_pickle_directory + "8500_scaler.p"
scaler_object = "scaler"
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
y_start_stop = [350, 680] # Min and max in y to search in slide_window()
print ("Loading X_scaler...")
X_scaler = load_from_pickle(file_scaler, scaler_object)
print ("Loading classifier...")
clf = load_from_pickle(file_classifier, classifer_object)

counter = 2
last_heat_map = None

def process_image(image):

    global counter
    global last_heat_map

    draw_image = np.copy(image)


    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #if np.max(image) > 1.0:
    image = image.astype(np.float32)/255

    #1
    cars_boxes1 = find_cars_boxes(draw_image, 370, 500, scale=1, svc=clf, X_scaler=X_scaler, orient=orient,
                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)

    #2
    cars_boxes2 = find_cars_boxes(draw_image, 400, 550, scale=1.5, svc=clf, X_scaler=X_scaler, orient=orient,
                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)

    #3
    cars_boxes3 = find_cars_boxes(draw_image, 400, 600, scale=2, svc=clf, X_scaler=X_scaler, orient=orient,
                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, hist_bins=hist_bins)


    # Add heat to each box in box list
    heat2 = np.zeros_like(draw_image[:,:,0]).astype(np.float)
    heat2 = add_heat(heat2, cars_boxes1)
    heat2 = add_heat(heat2, cars_boxes2)
    heat2 = add_heat(heat2, cars_boxes3)

    if last_heat_map is None:
        last_heat_map = np.zeros_like(heat2).astype(np.float)

    combined_heat = last_heat_map + heat2
    last_heat_map = heat2

    # Apply threshold to help remove false positives
    #heat2 = apply_threshold(heat2,1)
    heat2 = apply_threshold(combined_heat,2)

    # Visualize the heatmap when displaying
    heatmap2 = np.clip(heat2, 0, 255)

    # Find final boxes from heatmap using label function
    labels2 = label(heatmap2)
    find_cars_img = draw_labeled_bboxes(np.copy(draw_image), labels2)

    return find_cars_img


if __name__ == '__main__':
    main()
