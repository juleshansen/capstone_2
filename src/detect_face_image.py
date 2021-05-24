import cv2
import sys
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model/densenet_0.4')
em_labels = np.array([
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread(sys.argv[1])
faces = face_cascade.detectMultiScale(img, 1.1, 4)
for (x, y, w, h) in faces:
    face = img[y:y + h, x:x + w].copy()
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    face = np.dstack((face, face, face))
    face = face.reshape(1, 48, 48, 3)
    emotion = em_labels[model.predict(face).argmax()]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        img, emotion, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
        )
cv2.imshow('img', img)
cv2.waitKey(0) & 0xff
