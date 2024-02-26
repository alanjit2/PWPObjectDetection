import cv2 as cv  # Computer vision and image processing library
import numpy as np  # Numerical and math library


def find_contours(processed_img):
    # Code from https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
    contours, hierarchy = cv.findContours(processed_img,
                                          cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)
    return contours


def draw_contours(img, contours):
    # Code from https://www.tutorialspoint.com/how-to-draw-polylines-on-an-image-in-opencv-using-python
    # Draws actual contours
    img_with_contours = cv.polylines(img,
                                     contours,
                                     isClosed=False,
                                     color=(0, 255, 0),
                                     thickness=5)

    if len(contours) >= 2:  # Checks if 2 or more contours exist
        dimensions = min(len(contours[0]), len(contours[-1]))

        midline = []
        for x in range(dimensions):
            # Finds average
            average_of_x = (contours[0][x][0][0] + contours[-1][x][0][0]) // 2
            average_of_y = (contours[0][x][0][1] + contours[-1][x][0][1]) // 2
            midline.append((average_of_x, average_of_y))

        midline_np = np.array([midline])  # Transforms to Numpy Array

        # Draws midline
        img_with_contours = cv.polylines(img_with_contours,
                                         midline_np,
                                         isClosed=False,
                                         color=(255, 0, 0),
                                         thickness=10)

    return img_with_contours
