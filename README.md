# Optimize-image-duplicates
This is a python repository that detects and remove all similar-looking images from the datasets.


## Basic Usage of the scripts
Run the following script to find duplicates in specified images/dataset folder path

```python
python.exe main.py -i "C:\Users\dataset_1"

```

## Task
```
Find and remove all similar-looking images in a folder.
```

## Questions
```
• What did you learn after looking on our dataset?


-- There are images of different dimension, sizes and from different cameras. the data is collected from the cameras that are fized in particular location inside a parking lot at differnt timestamps with 4 different cameras (c10, c20, c21, c23).

```
```
• How does you program work?

1. At first, I took the input dataset path from the arguments, validate the path/directory.

2. Initialize a list and a dictionary to store the file names and images for processing.

3. Read the directory, iterated over all the files and checked the image file format.

4. Processed with preprocess_image_change_detection (existing function).

5. Converted the image files into numberic data (Numpy array) and append to a list.

6. Calculated the desired size for computation and resize the image (to speed up the comparision).

7. Then looped over the directory for finding the combinations of similar images for comparing two images.

8. Passed the search and compare images to compare_frames_change_detection(existing function) for score calculation.

9. Checked the Images with score less than 1000 (gives the images with similar/ Duplicate) for image quality

10.Added the low quality images to the duplicate list

11. Once all the files in the directory are finished, delete all the files in the duplicate list created.

```
```
• What values did you decide to use for input parameters and how did you find these
values?

There three input parameters that are considered min_contour_area, size of the image and the score value.

Images are resized to lower size image considering to speed up the comparision algorithm and not to lose essential data  from  the dataset.

The value of min_contour_area is given as 1000 considering the images contours that I found when comparing two images and leaving small lines from the images

The value of score is also given the input as 1000 for the images when the total area of the contours to be less than 1000.
```
```
• What you would suggest to implement to improve data collection of unique cases
in future?

There are many parameters to consider for improving the data collection of unique cases: some of them are

- Create a pipeline for video frames like motion detection and then store the images in collection.

- Ensuring the camera positions that are not effected directly by light sources.

- Find the location which covers entire path when there is a movement of pedestrians/vehicles and is not blocked by the existing objects  or parked cars in case of parking lots.

- Creating datasets with 5-10 cameras recording the data all the time (24/7) with good resolution (1920 x 1080) and fps rate of 30 and adding standard compression techniques even though the data have 1 tb of data per day
```
```
• Any other comments about your solution?

There are different ways of solving the problem like using Machine learning algorithms or keras models to provide the efficiency of the algorithm. but the solution is provided considering, the dataset doesn't contain any subfolders.
```