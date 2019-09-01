#STEP 1

#Import Statements
import numpy as np
import argparse
import cv2

#Create command line arguments to import photos.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the first image")
ap.add_argument("-I", "--Image", required = True,
    help = "Path to the second image")
args = vars(ap.parse_args())

#Assign photos to variables with cv2.
image = cv2.imread(args["image"])
image2 = cv2.imread(args["Image"])

#STEP 2

#Print length and width of pikachu.
print("Width:" + str(image.shape[1]) + " Height:" + str(image.shape[0]))

#STEP 3

#Display pikachu.
cv2.imshow("Pikachu", image)
cv2.waitKey(0)

#STEP 4

#Darken pikachu by 50 and display the new image.
subtract = np.ones(image.shape, dtype = "uint8") * 50
darkened = cv2.subtract(image, subtract)
cv2.imshow("Darkened", darkened)
cv2.waitKey(0)

#STEP 5

#Create Pikachu Noir with the dark image: any light gray pixel (the background) is rebrightened to full white.
pikachuNoir = darkened
for i in range(0, darkened.shape[0]):
    for j in range(0, darkened.shape[1]):
        (b, g, r) = darkened[i, j]
        if b >= 170 and g >= 170 and r >= 170: #Changed cutoff from 190 to 170. The original cutoff left some background pixels unchanged, making the mask in step 7 look messy or fuzzy.
            pikachuNoir[i, j] = (255, 255, 255)

#Display Pikachu Noir.
cv2.imshow("Pikachu Noir", pikachuNoir)
cv2.waitKey(0)

#STEP 6

#Create a mask that white circles will be added to
revealMask = np.zeros((pikachuNoir.shape[:2]), dtype = "uint8")
cv2.imshow("Pikachu Noir Reveal", revealMask)

#For loop runs 100 times, once for each circle.
for i in range(0, 100):
    cv2.waitKey(25)

    #Adds a circle of random size and location to the mask.
    radius = np.random.randint(5, high = 60)
    pt = np.random.randint(0, high = image.shape[0])
    pt2 = np.random.randint(0, high = image.shape[1])
    cv2.circle(revealMask, (pt2, pt), radius, (255, 255, 255), -1)

    #The mask is then applied to Pikachu Noir to reveal the next circle. Display is updated.
    pikachuNoirReveal = cv2.bitwise_and(pikachuNoir, pikachuNoir, mask = revealMask)
    cv2.imshow("Pikachu Noir Reveal", pikachuNoirReveal)

cv2.waitKey(0)

#STEP 7

#Black image is created for mask.
pikachuMask = np.zeros(image.shape, dtype = "uint8")

#For loop runs for every pixel in Pikachu Noir.
for i in range(0, pikachuNoir.shape[0]):
    for j in range(0, pikachuNoir.shape[1]):
        (b, g, r) = pikachuNoir[i, j]

        #Every pixel of Pikachu Noir is made white.
        if b < 255 and g < 255 and r < 255:
            pikachuMask[i, j] = (255, 255, 255)

        #Every background pixel is made black. Unfortunately part of Pikachu's eyes are included in the mask.
        else:
            pikachuMask[i, j] = (0, 0, 0)

#Display Pikachu mask.
cv2.imshow("Pikachu Mask", pikachuMask)
print("Pikachu Mask Width:" + str(pikachuMask.shape[1]) + " Height:" + str(pikachuMask.shape[0]))
cv2.waitKey(0)

#STEP 8

#Display second image, and print length and width in the console.
cv2.imshow("Second Image", image2)
print("Second image Width:" + str(image2.shape[1]) + " Height:" + str(image2.shape[0]))
cv2.waitKey(0)

#STEP 9

#Create ratio for resizing the second image.
r = image.shape[1] / image2.shape[1]
dim = (image.shape[1], int(image2.shape[0] * r))

#Resizes second image and displays it.
resized = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Second Image Resized", resized)
print("Resized Width:" + str(resized.shape[1]) + " Height:" + str(resized.shape[0]))
cv2.waitKey(0)

#Crops second image to a square and displays it. Now it has the same dimensions as Pikachu Noir.
cropped = resized[200:200+resized.shape[1], 0:resized.shape[1]]
print("Cropped Width:" + str(cropped.shape[1]) + " Height:" + str(cropped.shape[0]))
cv2.imshow("Second Image Cropped", cropped)
cv2.waitKey(0)

#STEP 10

#Applies pikachu mask to cropped image and displays result.
maskedImage2 = cv2.bitwise_and(cropped, pikachuMask)
cv2.imshow("Second Image with Pikachu Mask", maskedImage2)
cv2.waitKey(0)
