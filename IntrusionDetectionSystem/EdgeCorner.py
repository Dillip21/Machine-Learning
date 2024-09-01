import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge Detection using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Corner Detection using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    # Draw edges on the original frame
    frame_with_edges = cv2.bitwise_and(frame, frame, mask=edges)

    # Draw corners on the original frame
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame_with_edges, (x, y), 3, 255, -1)

    # Display the resulting frame
    cv2.imshow('Edges and Corners', frame_with_edges)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
