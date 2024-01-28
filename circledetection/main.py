from processing_display_image import *  # My library for displaying and processing images
from circles import *  # My library for circle detection/drawing
import cv2 as cv  # Computer vision and image processing library, to detect and draw circles, and load images


if __name__ == "__main__":
    # File name of the image
    file = 'Circle_Image.jpg'
    img = cv.imread(file)

    # Makes image grayscale and uses blurring methods
    gray_img = process_img(img)

    # Detects and draws circles with OpenCV
    detected_circle = detect_circles(gray_img)
    img = draw_circle(img, detected_circle)

    # Displays image with a delay until user presses "0" on the keyboard
    display_img(img, "Detected Circles")
