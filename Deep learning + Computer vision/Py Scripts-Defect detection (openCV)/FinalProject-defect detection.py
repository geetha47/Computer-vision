import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    r'/Users/fourier/Library/CloudStorage/OneDrive-Sotai/HV/REC-BOOKS/COURSE-BOOKS/STB600/FINAL_PROJECT/computer_vision/resources/models/imageclassifier01v1.h5')


def preprocess_image(image):
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = resized_image / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image


test_image = cv2.imread(r"/Users/fourier/Library/CloudStorage/OneDrive-Sotai/HV/REC-BOOKS/COURSE-BOOKS/STB600/GROUP-4_FINAL-PROJECT/test_data/test/imagex01.jpg")


def find_Contours(test_image):
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    GaussianBlur1 = 5
    GaussianBlur2 = 5
    thresh1 = 100
    thresh2 = 255
    SetIterations = 2

    if GaussianBlur1 % 2 == 0:
        GaussianBlur1 += 1

    if GaussianBlur2 % 2 == 0:
        GaussianBlur2 += 1

    blurred = cv2.GaussianBlur(gray, (GaussianBlur1, GaussianBlur2), 0)
    # edged = cv2.Canny(blurred, thresh1, thresh2)
    # edged=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    a, edged = cv2.threshold(blurred, thresh1, thresh2, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(edged, None, iterations=SetIterations)
    countours, heirarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return countours


if test_image is not None:
    frame = test_image.copy()
    preprocessed_image = preprocess_image(frame)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    print(predictions)

    contours = find_Contours(frame)
    print(len(contours))
    Objects = 0

    for i in contours:
        M = cv2.moments(i)
        if (M['m00'] != 0):
            Objects = Objects + 1
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            contour_area = cv2.contourArea(i)
            if contour_area < 500:
                continue
            # Draw contours

            if predictions[0][0] > predictions[0][1]:
                #cv2.drawContours(frame, [i], -1, (0, 255, 0), 2)

                # Display "Defective" text
                cv2.putText(frame, 'Defective', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                # Draw rectangle using calculated coordinates
                rect_size = 500
                cv2.rectangle(frame, (cx - rect_size, cy - rect_size), (cx + rect_size, cy + rect_size), (0, 0, 255), 2)
                print('Paperclip is defective')
                break

            else:
                print("Paperclip is in good condition:Approved")
                # cv2.drawContours(frame, [i], -1, (0, 255, 0), 2)
                cv2.putText(frame, 'Non Defective:Approved', (cx - 50, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 3,
                            (0, 255, 0), 6)
                break

    frame = cv2.resize(frame, (900, 600))
    cv2.imshow('Paperclip classification', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
