import math
import os
from pathlib import Path
import numpy as np
import torch
from math import pi as pi
import cv2 as cv
from icecream import ic


def fitEllipse_directly(img, contours):
    ellipse = cv.fitEllipse(contours[0])

    COLOR = (255, 255, 0)
    THICKNESS = 10
    img = cv.ellipse(img, ellipse, COLOR, THICKNESS)
    return img, ellipse

def fitEllipse_after_filling(img, contours):
    # Concatenate contours into one
    contours = np.concatenate(contours)

    # fit ellipse on the bigger contours
    ellipse = cv.fitEllipse(contours)

    # fill above ellipse to get a new img
    color = (255, 255, 255)
    img = cv.ellipse(img, ellipse, color, -1)

    # fit ellipse on the new img
    contours = extract_contours_from_img(img)

    COLOR = (0, 255, 0)
    THICKNESS = 10
    ellipse = cv.fitEllipse(contours[0])
    img = cv.ellipse(img, ellipse, COLOR, THICKNESS)

    return img, ellipse


def obtain_paths_of_files_in_string(parent_dir, extension=None):
    parent_dir = Path(parent_dir)
    json_paths = parent_dir.glob(f'*{extension}')
    json_paths = [str(path) for path in json_paths]
    return json_paths


def convert_to_greyscale_for_findContours(path):
    img = cv.imread(path)
    img[img != 0] = 1
    img = img * 255 # vì ảnh đang chỉ gồm [0, 1]
    return img


def extract_contours_from_img(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

    # return only the first contour
    # if len(contours) != 0:
    #     return contours[0]
    # else:
    #     return []


def compose_path_of_png_for_imwrite(save_dir, json_path):
    # in case PosixPath is passed
    save_dir, json_path = str(save_dir), str(json_path)
    name = os.path.basename(json_path)
    name = os.path.splitext(name)[0]
    png_path = save_dir + '/' + name + '.png'
    return png_path

def compute_ellipse_foci(ellipse):
    (center_x, center_y), (minor_axis, major_axis), theta = ellipse
    c = math.sqrt(abs(major_axis**2 - minor_axis**2))

    foci_1_x = center_x + c/2 * math.sin(theta)
    foci_1_y = center_y + c/2 * math.cos(theta)

    theta = theta + 90

    # foci_1_x = (center_x + c/2)*math.cos(theta) - center_y*math.sin(theta)
    # foci_1_y = (center_x + c/2)*math.sin(theta) + center_y*math.cos(theta)

    foci_2_x = center_x - c/2 * math.sin(theta)
    foci_2_y = center_y - c/2 * math.cos(theta)

    foci_1_x, foci_1_y = np.rint(foci_1_x), np.rint(foci_1_y)
    foci_2_x, foci_2_y = np.rint(foci_2_x), np.rint(foci_2_y)

    foci_1_x, foci_1_y = np.rint(foci_1_x).astype(int), np.rint(foci_1_y).astype(int)
    foci_2_x, foci_2_y = np.rint(foci_2_x).astype(int), np.rint(foci_2_y).astype(int)
    return (foci_1_x, foci_1_y), (foci_2_x, foci_2_y)

def compute_ellipse_parameters(ellipse):
    foci_1, foci_2 = compute_ellipse_foci(ellipse)
    (center_x, center_y), (minor_axis, major_axis), theta = ellipse

    center_x, center_y     = np.rint(center_x).astype(int), np.rint(center_y).astype(int)
    minor_axis, major_axis = np.round(minor_axis, 2), np.round(major_axis, 2)
    theta                  = np.rint(theta)

    params = [(center_x, center_y), foci_1, foci_2, minor_axis, major_axis, theta]
    return params