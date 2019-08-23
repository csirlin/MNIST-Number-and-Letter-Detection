import numpy as np
import cv2
import time


#setup
canvas = np.zeros((960, 1680, 3), dtype = "uint8")
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)

#shapes
"""cv2.line(canvas, (0, 0), (300, 300), green)

cv2.line(canvas, (300, 0), (0, 300), red, 3)
cv2.rectangle(canvas, (10, 10), (60, 60), green)
cv2.rectangle(canvas, (50, 200), (200, 225), red, 5)
cv2.rectangle(canvas, (200, 50), (225, 125), blue, -1)"""

#window
"""cv2.rectangle(canvas, (0,0), (300,300), green, 10)
cv2.line(canvas, (150,0), (150,300), green, 5)
cv2.line(canvas, (0,150), (300,150), green, 5)"""

#circles
"""cv2.circle(canvas, (150, 150), 150, blue, -1)
cv2.circle(canvas, (150, 150), 125, white, -1)
cv2.circle(canvas, (150, 150), 100, red, -1)
cv2.circle(canvas, (150, 150), 75, blue, -1)
cv2.circle(canvas, (150, 150), 50, white, -1)
cv2.circle(canvas, (150, 150), 25, red, -1)"""

#randomcircles-automatic
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
for i in range(0, 1000):

    radius = np.random.randint(5, high = 69)
    color = np.random.randint(0, high = 256, size = (3,)).tolist()
    pt = np.random.randint(0, high = 960)
    pt2 = np.random.randint(0, high = 1680)
    cv2.circle(canvas, (pt2, pt), radius, color, -1)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(50)

#drawing
cv2.waitKey(0)
