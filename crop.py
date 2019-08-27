import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
#               y1:y2 ,  x1:x2
cropped = image[60:220, 500:675]
cv2.imshow("T-Rex Face", cropped)
cv2.waitKey(0)
