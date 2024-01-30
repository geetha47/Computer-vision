**Title: Paperclip defect detection**

Developed a CNN model using TensorFlow and Keras for efficient paperclip defect detection. Utilizes OpenCV for image classification, contour detection, and offers object recognition on defective paperclips.
***Note: The dataset for this project is generated using Basler camera in a lab environment.***

**Building a cnn image classifer**
Steps performed:
1. Data creation 
3. Image preprocessing
4. Data augmentation
5. build cnn architecture
6. Model evaluation

**Python scipts uses OpenCV library to perform image classification and drawing on a test image.**
Steps involved:
1. Loads the pre-trained cnn model
2. Find contours by applying techniques like:
      - Smoothing using Gaussian blur
      - Thresholding using Canny filter
      - Dilation method
3. Object classification and drawing
    when the predicted class is "defective"

 **Image conversion- Converts the original 'bmp' format to 'jpg' format to smooth the model performace**
        
