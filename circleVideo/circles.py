import cv2 as cv  # Computer vision and image processing library
import numpy as np  # Library for arrays, better for memory and commonly used for larger arrays


def detect_circles(processed_img):
    # NumPy array method which indicates how many elements the first array has
    rows = processed_img.shape[0]

    # param 1 & 2 correlate to cv.HOUGH_GRADIENT, which detects the edges
    # dp is ratio of accumulator resolution to image resolution
    # Stores center line and radius

    circles = cv.HoughCircles(processed_img, cv.HOUGH_GRADIENT,
                              1, rows / 8,
                              param1=19, param2=55,
                              minRadius=300, maxRadius=550)
    return circles


def draw_circle(img, circles):
    # If statement to prevent an error if no circles are detected
    if circles is not None:
        # Round up/down numbers in array
        circles = np.uint16(np.around(circles))
        # Element 0, 1 is center of circle
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            # circle center, draws a dot (very tiny circle)
            cv.circle(img, center, 1, (0, 0, 255), 20)
            # circle outline, element 2 is radius
            radius = circle[2]
            cv.circle(img, center, radius, (0, 255, 0), 20)

        return img
    else:
        return img