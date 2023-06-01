import cv2
import os


def main():
    files = [f"assets/{file}" for file in os.listdir("assets")]
    files.sort()
    images = [cv2.imread(file) for file in files]

    # for index, image in enumerate(images[2:22]):
    for index, image in enumerate(images[2:3]):
        index = index+2
        # crop image
        image = image[30:542, 82:594]
        # find contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # draw contours
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        # show image

        cv2.imshow(f"image_{index}", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()