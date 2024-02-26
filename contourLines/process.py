import cv2 as cv  # Computer vision and image processing library
import numpy as np  # Numerical and math library


def process_img_for_contours(img):
    # Makes image grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Finds the edges in an image
    gray_edge = cv.Canny(gray, 125, 50)

    return gray_edge


def mask_img(img):  # create a mask
    # Code from https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python Example 2
    # Region to mask
    # SCHOOL
    height_1, height_2 = 200, 850
    width_1, width_2 = 450, 1250

    # Set Whole image as black
    mask = np.zeros(img.shape[:2], np.uint8)

    # Set region of interest as white
    mask[height_1:height_2, width_1:width_2] = 255

    # Make the mask
    masked_img = cv.bitwise_and(img, img, mask=mask)

    return (width_1, height_1), (width_2, height_2), masked_img


def draw_mask(point1, point2, img):
    # Draw mask
    masked_img = cv.rectangle(img, point1, point2, (0, 0, 0), 5)
    return masked_img


def display_img(img, winname):
    # Displays image with window name, or the name of the tab
    cv.imshow(winname, img)
