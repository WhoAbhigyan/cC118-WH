import cv2

# Load the video
cap = cv2.VideoCapture("walking.avi")

# Load the pre-trained cascade classifier
body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Loop over the frames of the video
while True:
    # Read the current frame
    ret, frame = cap.read()

    # Check if the video has ended
    if not ret:
        break

    # Convert the frame to grayscale for faster processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the bodies in the current frame
    bodies = body_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the current frame
    cv2.imshow("frame", frame)

    # Check if the user has pressed 'q' to quit
    if cv2.waitKey(0):
        break

# Release the video and destroy the window
cap.release()
cv2.destroyAllWindows()
