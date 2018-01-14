
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup/car_not_car.png
[image2]: ./writeup/HOG_example.png
[image3]: ./writeup/sliding_windows.jpg
[image4]: ./writeup/sliding_window.png
[image5]: ./writeup/bboxes_and_heat.png
[image7]: ./writeup/output_bboxes.png
[video1]: ./out_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

There are 6 main functions listed below that I used to build the project and generate content for the writeup. they start on line: 311 in the `image_generation.py` file:
    `display_random_images()`
    `train_classifier()`
    `classifier_examples()`
    `calculate_hog_for_entire_region()`
    `process_sample_images()`
    `generate_video()`

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 74 through 86 of the file called `image_generation.py`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
I wrapped this a function called `display_random_images` on line: 309

Here is an examples out the output I generated:

color_space is 'RGB'
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 4932
4.32 Seconds to train SVC...
Test Accuracy of SVC =  0.9575

color_space is 'YCrCb'
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
1.54 Seconds to train SVC...
Test Accuracy of SVC =  0.9775

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and there was a good improvement in accuracy to 0.99 when I ran it over the entire dataset.

color_space = 'YCrCb'
hog_channel = 'ALL'
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a function I produced called: `train_classifier` (on line 348) to do this and I used pickle to serialize the svc and X_Scaler.
for use in later functions.

### Sliding Window Search

I implemented a slide_window function on line 158, to take an image, start stop positions and a window size.  And a search_windows function to find positive detection windows.
I used these functions in the `classifier_examples` function on line 408 to produce content and get a sense of how well the detection pipeline was functioning. I aslo reduced the section of the image
to be searched along the y axis, as it was giving false positives in the top half of the screen. By setting `y_start_stop = [400, 656]`

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Utimately I took the hog features once, and sub-sampled the array to improve performance. This is performed in a function called `calculate_hog_for_entire_region` on line 458. this works on
the test data. Here are some example images:

![alt text][image4]

I then used the `calculate_hog_for_entire_region` to construct a function called find_cars on line 559, it is Similar function to calculate_hog_for_entire_region, but works on one image.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


I combined the bounding boxes in the operation draw_labeled_bboxes on line 293 of the `image_generation.py` file. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are several frames and their corresponding heatmaps:

![alt text][image5]

### Here the resulting bounding boxes are drawn onto the sample images:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I did attempt to build the project as a series of steps that i could build further features on top of. I would like to improve it by abstracting and hiding away
some of the image presenting and processing operations to focus on the SVM training.
