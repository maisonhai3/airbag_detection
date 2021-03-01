from pathlib import Path
from tqdm import tqdm
import cv2 as cv
from utils.haims_utils import *
import matplotlib.pyplot as plt
from icecream import ic
import pandas as pd
import argparse


# pass arguments
parser = argparse.ArgumentParser()
parser.add_argument('Image_path', metavar='image_path', type=str, help='Absolute string path to images')
args = parser.parse_args()


SHOW_IMG = True
SAVE_IMG = False

IMG_DIR = args.Image_path
IMG_FORMAT = 'png'

SAVE_DIR = '/home/maihai/0_PROJECT_heads_airbags_cubebox/1_gui_Tung/s2_ellipse_0_AND_1'
fitEllipse_0_AND_1_DIR = Path(SAVE_DIR) / 'fitEllipse_0_and_1'
fitEllipse_0_AND_1_DIR.mkdir(parents=True, exist_ok=True)

fitEllipse_DOUBLE_CHECK_DIR = Path(SAVE_DIR) / 'fitEllipse_double_check'
fitEllipse_DOUBLE_CHECK_DIR.mkdir(parents=True, exist_ok=True)
SAVE_csv = SAVE_DIR + '/' + 'ellipse_params.csv'

# fit an ellipse and get ellipse params for each image in IMG_DIR
img_paths = obtain_paths_of_files_in_string(IMG_DIR, IMG_FORMAT)
df = []

for path in tqdm(img_paths):
    ic(path + '\n')
    org_img = cv.imread(path)
    grey_img = convert_to_greyscale_for_findContours(path)
    contours = extract_contours_from_img(grey_img)

    # fit ellipse
    if len(contours) == 0:
        print('This image has 0 airbag \n')
        continue
    elif len(contours) == 1:
        print('This image has 1 piece of airbag \n')
        img_with_ellipse, ellipse = fitEllipse_directly(grey_img, contours)
    elif len(contours) > 1:
        print('This image has >= 2 pieces of airbag \n')
        img_with_ellipse, ellipse = fitEllipse_after_filling(grey_img, contours)

    # ellipse parameters
    ellipse_params = compute_ellipse_parameters(ellipse)
    image_name = [os.path.basename(path)]
    df.append(image_name + ellipse_params)

    # draw center
    (center_x, center_y) = ellipse_params[0]
    img_with_center = cv.circle(img_with_ellipse, (center_x, center_y), 80, (200,100,255), -1)

    double_check_img = np.hstack((org_img*255, img_with_center))
    if SHOW_IMG:
        plt.imshow(double_check_img)
        plt.show()

    if SAVE_IMG:
        path_0_and_1 = compose_path_of_png_for_imwrite(fitEllipse_0_AND_1_DIR, path)
        cv.imwrite(path_0_and_1, img_with_ellipse)

        path_double_check = compose_path_of_png_for_imwrite(fitEllipse_DOUBLE_CHECK_DIR, path)
        cv.imwrite(path_double_check, double_check_img)

df = pd.DataFrame(df, columns=['Image name', 'Center', 'Foci 1', 'Foci 2', 'Minor axis', 'Major axis', 'Theta'])
df.to_csv(SAVE_csv)
print('\n Ellipse parameters')
print(df.head())