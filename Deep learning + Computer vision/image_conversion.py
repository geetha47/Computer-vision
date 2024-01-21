import cv2 as cv 
import os


""" LOAD IMAGE FROM THE DIRECTORY. """
colored_images = [ ]
for filename in os.listdir("NON_DEFECT/"):
    colored = cv.imread("NON_DEFECT/" + filename)
    colored_images.append(colored)
   

print(len(colored_images))

""" CONVERT IMAGE TO JPG FORMAT."""
image_name = "image"
for j in range(len(colored_images)):
    cv.imwrite(f"{image_name}{j}.jpg", colored_images[j])
    