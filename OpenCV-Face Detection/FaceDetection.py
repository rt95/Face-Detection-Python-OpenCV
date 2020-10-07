import cv2
import sys

cascPath = "C:/Users/Laptop/Desktop/projekt cv/frontalFace.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:

    # Frame by frame capture
    ret, frame = video_capture.read()

    # Grayscale it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Face
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw the Rectangles
    for (x, y, w, h) in faces:
        rect = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        topLeft = (x, y)
        bottomRight = (x+w, y+h)

        # Blur ROI - Region of intrest
        x2, y2 = topLeft[0], topLeft[1]
        w2, h2 = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

        # Grab ROI with slicing and blur it
        ROI = frame[y2:y2+h2, x2:x2+w2]
        blur = cv2.GaussianBlur(ROI, (51, 51), 0)

        # Insert ROI back into image
        frame[y2:y2+h2, x2:x2+w2] = blur
        cv2.imshow('blur', blur)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
