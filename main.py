import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame from webcam
    _, frame = webcam.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
        face_roi = frame[y:y+h, x:x+w]
        gaze.refresh(face_roi)

        
        text = ""
        if gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        
        cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)

        # Get the coordinates of the left and right pupils
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        # Display the pupil positions if available
        if left_pupil is not None:
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (x, y - 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)
        if right_pupil is not None:
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (x, y - 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)
    
    # Display the annotated frame
    
    cv2.imshow("Demo", frame)

    # Exit loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
