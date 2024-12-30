import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Handle OpenCV version compatibility for cv2.findContours
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 3:  # OpenCV 3.x
        _, contours, _ = result
    else:  # OpenCV 4.x
        contours, _ = result

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Real-Time Yellow Boundary Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
