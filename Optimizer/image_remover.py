#
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#

# Created on Fri Apr 22 2022
#
# The MIT License (MIT)
# Copyright (c) ${2022} Prudhvi raj Panisetti
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import os
import sys
import json
import time
import collections
import imghdr

from typing import List, Tuple, Dict
from argparse import ArgumentParser

import cv2
import numpy as np

from Optimizer.imaging_interview import compare_frames_change_detection, preprocess_image_change_detection


class ImageRemover(object):
    """
    Class to detect similar/duplicate images and remove them from the input path provided
    """

    def  __init__(self):
        """
        Initialize the class and parser private varibale for getting the input from  the command line
        parising the dataset path to read the images from the folder
        """

        self._parser =  ArgumentParser(
            description = "Get input dataset form Command line."
        )

        self._parser.add_argument(
            "-i", "--input",
            dest = "inputpath",
            required = True,
            help = "Path for input folder with Images"
        )

        self.__dataset_directory  = None


    def __str__(self):
        return "This is the object of class {} with dataset folder {}".format(self.__class__.__name__, self.__dataset_directory)


    def __get_dataset_directory(self):
        """
        get method returns the dataset_directory
        """

        return self.__dataset_directory


    def __set_dataset_directory(self):
        """
        parses the arguement and sets the dataset directory path
        """

        args = self._parser.parse_args()
        directory = args.inputpath
        if(directory != None):
            self.__dataset_directory = ImageRemover.__validate_directory(directory)


    def __validate_directory(directory: str = "") -> str:
        """
        Check if the provided path for the directory is valid or not
        """

        directory = directory + os.path.sep
        try:
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Dataset directory: " + directory + " does not exist. Please provide correct input directory path.")
        except FileNotFoundError as error:
            print(error)
            sys.exit(1)

        return directory


    def __desired_size_calc(search_img: np.ndarray, compare_img: np.ndarray) -> Tuple:
        """
        Compares the size of two numpy arrays and returns the shape of smallest size numpy array
        """

        d_size = compare_img.shape[1::-1] if (search_img.size > compare_img.size)  else search_img.shape[1::-1]

        return d_size


    def __img_resize(img,d_size):
        """
        Resize the image to the given dimension size and returns the image
        """

        img = cv2.resize(img, dsize = d_size, interpolation = cv2.INTER_CUBIC)

        return img


    def __result_to_json(result: Dict = {}):
        """
        Writes the duplicate image file names from dictionary to json file
        """

        with open("duplicates_list.json", "w") as outfile:
            outfile.write(json.dumps(result))


    def __image_process(image_path: str = "") -> np.ndarray:
        """
        Read the image and convert the image to gray scale

        returns an image
        """

        # read the image from path
        img = cv2.imread(image_path)

        # pre-process the image like gray conversion and drawing the borders
        img = preprocess_image_change_detection(img)

        return img


    def __list_of_images(directory: str = "") -> List:
        """
        create list of all files in directory
        """

        folder_files = [(filename, os.path.join(directory, filename)) for filename in os.listdir(directory)]

        return folder_files


    def __search_directory(directory: str = "") -> Tuple[Dict , List]:
        """
        Search for the duplicate images and returns the file names in a dictionary and duplicates in list
        """

        start_time = time.time()
        images_list = ImageRemover.__list_of_images(directory)
        img_arrays, filenames = ImageRemover.__create_img_array(images_list)

        result = {}
        duplicate_list = []
        sort_output = True

        # find duplicates/similar images within one folder
        for img_search_index, img_search in enumerate(img_arrays):
            for img_compare_index, img_compare in enumerate(img_arrays):

                # check for the combinations of image names
                if (img_compare_index != 0 and img_compare_index > img_search_index and img_search_index != len(img_arrays)):

                    # check if the search image is in duplicate list
                    if((directory + os.path.sep + filenames[img_compare_index]) not in duplicate_list):

                        desired_size = ImageRemover.__desired_size_calc(img_search, img_compare)

                        image_search_array = ImageRemover.__img_resize(img_search,desired_size)
                        image_compare_array = ImageRemover.__img_resize(img_compare,desired_size)

                        similarity_score, residual_counts, threshold = compare_frames_change_detection(image_search_array, image_compare_array, 1000)

                        # Image is non-essential/Duplicate if similarity score is greater than 1000 (Considering the contour sizes)
                        if(similarity_score < 1000):

                            # Add duplicates and respective Originals to a dictionary for debugging and verification
                            if (filenames[img_search_index] in result.keys()):
                                result[filenames[img_search_index]]["duplicates"] = result[filenames[img_search_index]]["duplicates"] + [directory + os.path.sep + filenames[img_compare_index]+" "+str(similarity_score)]
                            else:
                                result[filenames[img_search_index]] = {"location" : directory + os.path.sep + filenames[img_search_index],
                                                                            "duplicates" : [directory + os.path.sep + filenames[img_compare_index]+" "+str(similarity_score)]
                                                                    }

                            # Checks for low quality image and add to duplicate List
                            low_size_image =  ImageRemover._check_low_img_size(directory, filenames[img_search_index], filenames[img_compare_index] )
                            duplicate_list.append(low_size_image)
        # Sort results for debugging
        if(sort_output == True):
            result = collections.OrderedDict(sorted(result.items()))

        time_elapsed = np.round(time.time() - start_time, 4)

        print("Found", len(result), "image/images with one or more duplicate/similar images in", time_elapsed, "seconds.")

        return result, set(duplicate_list)

    def _check_low_img_size(directoryA, image_search, image_compare):
        """
        Checks the quality/size of the image and returns the lower quality image for deleting the lower quality image
        """
        dir = ImageRemover.__validate_directory(directoryA)
        size_img_search = os.stat(dir + image_search).st_size
        size_img_compare = os.stat(dir + image_compare).st_size
        if size_img_search >= size_img_compare:
            return directoryA + "/" + image_compare
        else:
            return directoryA + "/" + image_search


    def __create_img_array(folder_files: List = []) -> Tuple[List, List]:
        """
        Merge all the images into a single list and image file names into another list
        """

        start_time = time.time()

        img_filenames = []
        # create images matrix
        imgs_array = []

        for filename, path in folder_files:
            try:
            # check if the file is not a folder
                if not os.path.isdir(path):

                    # check if the file is image  or not
                    if imghdr.what(path):

                        img_search = ImageRemover.__image_process(path)
                        imgs_array.append(img_search)
                        img_filenames.append(filename)

            except OSError as error:
                    print(error)
                    sys.exit(1)

        time_elapsed = np.round(time.time() - start_time, 4)
        print("Time taken to create img matrix  ", time_elapsed, "seconds.")
        return imgs_array, img_filenames


    def __delete_imgs(duplicate_images: list = []):
        """
        Delete the duplicate images from the dataset directory
        """

        count_deleted_images = 0

        for image in duplicate_images:
            try:
                os.remove(image)        # Remove the image from the folder

                count_deleted_images = count_deleted_images + 1
            except OSError as error :
                print(error, " Cannot not delete file:", image, end = "\r")
                sys.exit(1)
        print("\n****************\nDeleted", count_deleted_images, "images.")


    def dataset_optimizer(self):
        """
        Remove Duplicate/Similar images from the dataset and
        store the original image and its respective dulicate image filenames in json file
        """

        self.__set_dataset_directory()
        result, duplicate_list = ImageRemover.__search_directory(self.__get_dataset_directory())
        ImageRemover.__result_to_json(result)
        ImageRemover.__delete_imgs(duplicate_list)

