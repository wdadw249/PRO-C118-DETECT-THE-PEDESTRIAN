import cv2

# Create our body classifier
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    # Convert each frame into grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    people = classifier.detectMultiScale(grayscale, scaleFactor=1.2, minNeighbors=3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

   
    cv2.imshow("Pedestrians", frame)

   
    if cv2.waitKey(1) == 32:  
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
