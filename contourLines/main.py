from process import *
from contours import *

if __name__ == "__main__":
    # Take Video
    capture = cv.VideoCapture(0)
    while True:
        # Process it frame by frame
        frame_exist, img = capture.read()

        if frame_exist:  # Prevents code from crashing

            # Processing img, draws lines, displays image
            edge_img = process_img_for_contours(img)
            mask_point_1, mask_point_2, masked_img = mask_img(edge_img)

            contours = find_contours(masked_img)

            new_img = draw_mask(mask_point_1, mask_point_2, img)

            new_img = draw_contours(new_img, contours)

            display_img(new_img, "Video")
            # Stream ends when user types 0
            if cv.waitKey(1) & 0xFF == ord('0'):
                break

    # Lines 28-end: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
    # After the loop release the video
    capture.release()
    # Destroy all the windows
    cv.destroyAllWindows()
