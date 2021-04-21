import cv2
import sys
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread(sys.argv[1])
faces = face_cascade.detectMultiScale(img, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0) & 0xff
