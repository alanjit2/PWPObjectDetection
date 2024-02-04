import cv2 as cv  # Computer vision and image processing library

def process_img(img):
    # Makes image grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Blurs image as a whole
    gray = cv.blur(gray, (20, 20))

    # Takes the median color of a specific area and used to limit noise (number of pixels)
    gray = cv.medianBlur(gray, 1)

    return gray


def display_img(img, winname):
    # Displays image with window name, or the name of the tab
    cv.imshow(winname, img)
    cv.waitKey(1)


def display_video_frame(frame, winname):
    cv.imshow(winname, frame)
    cv.waitKey(1)