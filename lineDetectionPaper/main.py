from processing_image_display import *  # Adds blurs to image, past library
from lines import *  # Library for everything lines (draw and detect)


if __name__ == "__main__":
    # Take Video
    capture = cv.VideoCapture(0)
    while True:
        # Process it frame by frame
        frame_exist, img = capture.read()

        if frame_exist:  # Prevents code from crashing

            # Processing img, draws circles, displays image

            masked_img = mask_img(img)

            edge_img = process_img_for_lines(masked_img)
            edge_img = draw_center(edge_img, img)

            lines = detect_lines(edge_img)
            new_img = draw_lines(lines, img)

            # Shows video
            display_img(new_img, "Video with Lines")
        else:
            break
    # Close the window
    capture.release()

    # De-allocate any associated memory usage
    cv.destroyAllWindows()
