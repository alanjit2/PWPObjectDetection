import cv2 as cv  # Computer vision and image processing library
import numpy as np  # Numerical and math library


def detect_lines(img):  # This detects line using Hough Lines P function in opencv
    # code from https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/ Example 2
    lines = cv.HoughLinesP(
        img,  # edge image
        rho=1,
        theta=np.pi / 180,
        threshold=30,  # Sets the num of intersections on polar plane to consider a line
        minLineLength=50,  # Line must be over set pixels
        maxLineGap=1  # max allowed gap to consider a separate line
    )
    return lines  # In the form [x1 y1 x2 y2]


def draw_lines(lines, img):
    # code from https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/ example 2
    if lines is not None:
        for line in lines:
            # Extracted points from the list "line"
            x1, y1, x2, y2 = line[0]
            # Draw the lines joining the points
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


def draw_center(edge_img, img):  # Calculate midpoint in canny edge directly
    # Checks row by row
    for idx in range(edge_img.shape[0]):  # For each row
        mid_point = np.mean(np.where(edge_img[idx, :] > 0))  # sets midpoint based on edge detection
        if not (np.isnan(mid_point)):
            img[idx, int(mid_point)] = 50

    # Column by Column
    for idx in range(edge_img.shape[1]):  # For Each column
        mid_point = np.mean(np.where(edge_img[:, idx] > 0))  # sets midpoint based on sharp pixel change
        if not (np.isnan(mid_point)):  # If mean point exists
            img[int(mid_point), idx] = 50  # Draw it out on main

    return edge_img
