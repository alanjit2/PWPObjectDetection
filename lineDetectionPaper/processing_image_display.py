import cv2 as cv  # Computer vision and image processing library
import numpy as np  # Numerical and math library


def process_img_for_lines(img):
    # Makes image grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Finds the edges in an image
    gray_edge = cv.Canny(gray, 125, 130)

    return gray_edge


def display_img(img, winname):
    # Displays image with window name, or the name of the tab
    cv.imshow(winname, img)
    cv.waitKey(1)


def mask_img(img):  # create a mask
    # Code from https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python Example 2
    # Region to mask
    height_1, height_2 = 200, 950
    width_1, width_2 = 450, 1350

    # Set Whole image as black
    mask = np.zeros(img.shape[:2], np.uint8)

    # Set region of interest as white
    mask[height_1:height_2, width_1:width_2] = 255

    # Make the mask
    masked_img = cv.bitwise_and(img, img, mask=mask)
    # Draw mask
    masked_img = cv.rectangle(masked_img, (width_1, height_1), (width_2, height_2), (0, 0, 0), 5)

    return masked_img
