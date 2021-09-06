import cv2
detector = cv2.CascadeClassifier("/Users/shreyashrivastava/opt/anaconda3/pkgs/libopencv-4.5.2-py39h852ad08_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("/Users/shreyashrivastava/Desktop/sample.mp4")
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 7)
    img = cv2.putText(img, 'The no of faces:{}'.format(len(faces)), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in faces:
        if len(faces)==1:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display
        cv2.imshow('img', img)
        # Stop if q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()











































































'''import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier("/Users/shreyashrivastava/opt/anaconda3/pkgs/libopencv-4.5.2-py39h852ad08_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/shreyashrivastava/opt/anaconda3/pkgs/libopencv-4.5.2-py39h852ad08_1/share/opencv4/haarcascades/haarcascade_eye.xml")
img = cv2.imread("/Users/shreyashrivastava/Desktop/XYZ.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''



'''
# Load the cascade
face_cascade = cv2.CascadeClassifier("/Users/shreyashrivastava/opt/anaconda3/pkgs/libopencv-4.5.2-py39h852ad08_1/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()'''