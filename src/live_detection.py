import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the "Face Finder" and the "Emotion Brain"
face_haar_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
model = load_model('models/emotion_model.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

cap = cv2.VideoCapture(0) # Open Webcam

while True:
    ret, frame = cap.read() # Capture image from webcam
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
        roi_gray = gray_img[y:y+w, x:x+h] # Cropping the face area
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        img_pixels = np.array(roi_gray).astype('float32') / 255.0
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Facial Emotion Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to stop
        break

cap.release()
cv2.destroyAllWindows()