
import cv2
import sys
import numpy as np

class MyDetectionMethods:
    def __init__(self):
        pass

    def filter(self, img, thres1, thres2, kernel1, kernel2, iterations,filter_name):
        self.img = img
        self.thres1 = thres1
        self.thres2 = thres2
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.iterations = iterations
        self.filter_name=filter_name
        img_copy = img.copy()

        # Convert it to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blurred img
        blurred = cv2.GaussianBlur(gray, (kernel1, kernel2), 0)

        # mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,19,5 )

        # Canny to find the edges
        if filter_name=='canny':
            edged = cv2.Canny(blurred, thres1, thres2)
        elif filter_name=='binary':
            _, edged = cv2.threshold(blurred, thres1, thres2, cv2.THRESH_BINARY)

        # Dilate the edges
        dilalted = cv2.dilate(edged,None, iterations=iterations)

        # find the countors in the diluted picture
        countours, _ = cv2.findContours(dilalted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return countours, dilalted

cv2.waitKey(0)
cv2.destroyAllWindows()