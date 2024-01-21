*****Title: Paperclip defect detection****
Developed a CNN model using TensorFlow and Keras for efficient paperclip defect detection. Utilizes OpenCV for image classification, contour detection, and offers object recognition on defective paperclips.
**Building a cnn image classifer**
Steps performed:
1. Data creation
2. Image preprocessing
3. Data augmentation
4. build cnn architecture
5. Model evaluation

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
        
