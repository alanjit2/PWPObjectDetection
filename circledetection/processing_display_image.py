import cv2 as cv # Computer vision and image processing library


def process_img(img):
    # Makes image grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Blurs image generally with 10 intensity
    gray = cv.blur(gray, (10, 10))

    # Takes the median color of a specific area and used to blur
    # This mainly is used to prevent detection of the inner circles in the can
    gray = cv.medianBlur(gray, 31)

    return gray


def display_img(img, winname):
    # Displays image with window name, or the name of the tab
    cv.imshow(winname, img)

    # Keeps image until user presses 0 on keyboard, then destroys it
    cv.waitKey(0)
    cv.destroyAllWindows()
