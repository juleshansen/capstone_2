import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('../data/ExtendedYaleB/yaleB13/yaleB13_P00A-005E-10.pgm')
faces = face_cascade.detectMultiScale(img, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey()