# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[camera_calibration_dist]: ./output_images/camera_cal_dist.png "Distorted images"
[camera_calibration_undist]: ./output_images/camera_cal_undist.png "Undistorted images"
[dist_corrected]: ./output_images/dist_corrected.png "Distortion corrected images"
[direc_grad]: ./output_images/direc_grad.png "Directional gradient threshold"
[mag_grad]: ./output_images/mag_grad.png "Magnitude of gradient"
[direction_grad]: ./output_images/direction_grad.png "Direction of gradient"
[hls_s]: ./output_images/hls_s.png "HLS S channel"
[hls_h]: ./output_images/hls_h.png "HLS H channel"
[roi]: ./output_images/roi.png "Region of interest"
[combined]: ./output_images/combined.png "Combined binary image"
[be]: ./output_images/be.png "Birds-eye transform"
[hist]: ./output_images/hist.png "Histogram"
[curve]: ./output_images/curve.png "Curve"
[roi_curve]: ./output_images/roi_curve.png "ROI around the curve"
[results]: ./output_images/result.png "Results"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Introduction

Advanced lane finding project is the second project in the Self-Driving Car Engineer nanodegree. The project is focused on image calibration and application of various thresholding techniques to find road lanes. The objective is to create a processing pipeline which takes series of steps required to find the lane lines in each image or  video frame. All referenced code is stored within Jupyter notebook "./advanced_lane_lines.ipynb" located in project root. 

### Camera Calibration

The code for this step is contained within the section *Camera calibration*.

The code int this section is based on an example provided within this project. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world, assuming the chessboard is fixed on the (x, y) plane at z=0.  `objp` is always the same as the coordinates don't change and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. The implementation is wrapped inside a helper function:

```python
def img_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```

Distorted images:
![Distorted images][camera_calibration_dist]

Undistorted images:
![Undistorted images][camera_calibration_undist]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained within the notebook section *Distortion correction*.

To undistort each testing image/video frame, I use my **img_undistort()** function described in chapter *Camera calibration*. The results I obtained can be seen in the following images:

![Distortion corrected images][dist_corrected]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained within the notebook section *Thresholded binary image*.

To create binary image used for perspective transform, I implemented various techniques to increase the reliability of the detection. To implement these, I followed the instructions from curicullum:

1. Directional gradient threshold - using the **cv2.Sobel()** operator, I implmented universal function to find either x or y gradient. The threshold values I use are **(30, 120)** and kernel size is **9**. I combine the results of both x and y gradient in the resulting combined threshold image. 
![Directional gradient threshold][direc_grad]

2. Magnitude of gradient - using the **cv2.Sobel()** operator, I implmented the calculation of absolute value of gradient. The threshold values I use are **(40, 150)** and kernel size is **9**.
![Magnitude of gradient][mag_grad]

3. Direction of gradient - using the **cv2.Sobel()** operator, I implmented threshold based on direction of gradient. This way I am able to emphasize the direction with combination of other gradients. The threshold values I use are **(0.7, 1.2)** and kernel size is **15**.
![Direction of gradient][direction_grad]

4. HLS threshold - the **S channel in HLS** colorspace performs well for emphasizing the lane lines in various lighting conditions. The threshold I use is **(120, 255)**.
![HLS S channel][hls_s]

5. HLS threshold - the **H channel in HLS** colorspace seems to emphasize on shadows. Since the gradient is high there, these are causing a lot of trouble (e.g. in the challenge video). The inverse of H channel can be actually used to filter out these false positives. The threshold I use is **(100, 255)**.
![HLS H channel][hls_h]

6. Region of interest - finally, I cut out only a portion of the image which might contain interesting data. The vertices for ROI polygon are defined as:

```python
x_center = testImages[0].shape[1]/2
x_lo_offset = 560
x_hi_offset = 100
y_lo = testImages[0].shape[0]
y_hi = int(testImages[0].shape[1]/3) + 40

vertices = np.array([[
	(x_center - x_lo_offset, y_lo), 
	(x_center - x_hi_offset, y_hi), 
	(x_center + x_hi_offset, y_hi), 
	(x_center + x_lo_offset, y_lo)]], dtype=np.int32)
```

![Region of interest][roi]

The resulting binary images display mostly the lane lines with only a small amount of artefacts:

![Combined binary image][combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained within the notebook section *Birds-eye perspective transform*.

To warp the binary image, I implemented a single fucntion called **warp()** which internally calls **cv2.warpPerspective** function using the parameters I predefined in this step:

```python
warp_offset = 300
warp_src = np.float32(vertices)
warp_dst = np.float32([[warp_offset, img.shape[0]], 
                       [warp_offset - 70, 0], 
                       [img.shape[1] - warp_offset + 70, 0], 
                       [img.shape[1] - warp_offset, img.shape[0]]])

M = cv2.getPerspectiveTransform(warp_src, warp_dst)
Minv = cv2.getPerspectiveTransform(warp_dst, warp_src)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 80,  720      | 300, 720      | 
| 535, 466      | 230, 0        |
| 745, 466      | 1050, 0       |
| 1200, 720     | 980, 720      |

![Birds-eye transform][be]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code related to this chapter can be find in the python notebook chapter *Finding the lane lines*.

The first step to identify the lane lines is to create the histogram of white pixels on x axis. The highest peaks will (probably) correspond to the lane lines. To create the histogram, I implemented a function named *hist()*:

```python
def hist(image):
    img = np.int32(np.where(image > 0, 1, 0))
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)    
    return histogram
```
![Histogram][hist]

Once I found the histogram, I used the sliding windows technique to identify the x and y coordinates of lane-line pixels. The code for this technique is located in the method **find_lane_pixels()** and is based on the code available in curriculum lessons. The code for this method can be found in the *Sliding windows* chapter in the Jupyter notebook. The hyperparameters that I found to work the best were defined like this:

```python
# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 10
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 80
```

Once I found the x and y coordinates for the pixels, I can use them to fit second-order polynomial to approximate the lane-line using curve. To do this, I've implemented a function named *fit_polynomial()*, which internally calls *np.polyfit()* function. With the polynomial fitted, I use the new fit to calculate x values for the y values passed as an argument. The y values represent all possible y values for an image. 

```python
def fit_polynomial(img, ploty, x, y):
    ### Fit a second order polynomial to each with np.polyfit() ###
    fit = np.polyfit(y, x, 2)
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    
    return fitx, fit
```

For the sake of validation, I vizualized the curve, obtaining following results:
![Curve][curve]

To make the search more efficient when processing video frames, I implemented also area-restricted search for the lane lines around the previous lane position. To do this, I implemented the function named **search_around_poly()**. The margin I used to search for the frames has the width of **90 px**. 
![ROI around the curve][roi_curve]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code related to this chapter can be find in the python notebook chapter *Calculating the radius of curvature*.

Once I obtained fits for both polynomials, I was able to calculate the actual curvature of the road. The function simply implements the formula for the calculation of curvature radius described in the curriculum. 

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension

def measure_curvature_real(x, y):
    fit = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    y_eval = np.max(y)
    curverad = (1 + (2 * fit[0] * y_eval * ym_per_pix + fit[1]) ** 2) ** 1.5 / np.abs(2 * fit[0]) 
   
    return curverad
```

Offset of the car is being calulated using the formula:

```python
def measure_car_offset_real(image, lx, rx):
    return round(((image.shape[1] / 2) - ((rx - lx) / 2 + lx)) * xm_per_pix, 4)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code related to this chapter can be find in the python notebook chapter *Implementing the pipeline*.

I implemented the actual processing pipeline in the method **processing_pipeline()**. It can be split into 4 logical parts:

1. Extracting the thresholded binary image.
2. Detecting the lane lines.
3. Calculating the measurements from best average fits (curvature and car offset).
4. Extra sanity checks if the algorithm is tracking the lines well or if they didn't diverge too much.

The lane lines are being detected using the **detect()** method, which internally calls either **find_lane_pixels()** (for full search) or **search_around_poly()** for faster detection after good previous detections. Once the lines are detected and polynomial is fit (if lines are found), I do some sanity checks for left and right line - I check if the curvature is similar to previous detection (with maximum difference of **10%**) and I check if the curve is in the correct half of the image. If these conditions are met, the current detection is being stored with previous detections with the total of up to **15** entries, which are than later used to calculate an average and draw the lane area using the **draw_image()** method.

At the end of each frame processing, the pipeline does extra sanity checks to see if we are still tracking the lanes correctly, namely:

1. Checking for how many frames we missed the lane detection (up to **35**). Once this counter is full, we reset the lane data history and begin a clean search.
2. We check if the curvature difference between the lanes is not bigger than **50%**. In theory, this number should be much smaller, but the algorithm is not well tuned in this regard. 

Note: if we run the pipeline on static images (e.g. the testing images), the averages will contain only one detection - from the actual image we're processing.

![Results][results]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

The result seem to be still a little wobbly.

Here's a [link to my challenge video result](./output_images/challenge_video.mp4)

This video is much harder but the detection is still relatively good.

Here's a [link to my harder challenge video result](./output_images/harder_challenge_video.mp4)

The results for the hard challenge video are quite bad. 


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As described in previous chapters, I implemented various techniques to increase the reliability of detection:

1. Directional gradient threshold
2. Magnitude of gradient
3. Direction of gradient
4. HLS threshold for S and H channels
5. Region of interest - finall

The overall idea was to emphasize the lane lines while getting rid of as much noise as possible. This generally works well on the project video, however, seems to be more difficult in the challenge videos. 

The challenge video adds an extra complexity to the detection - a strong shadow on the left and the road color changing in the middle od the lane. I managed to at least partially mitigate the problem with shadow (which creates strong edge in the same direction as lane lines) by using the inverse of H channel in the HLS colorspace. To deal with the strong edge in the middle of the road, I would probably need to either add more sanity checks or to make line acceptance criteria even more strict. 

The harder challenge video is even more difficult to work with, especially because of the sharp turns. I suspect the main reason for my algorithm to fail here is the ROI settings I use to clip the detection region. In this case it would probably be worth to either widen the ROI area, or not to rely on it at all.