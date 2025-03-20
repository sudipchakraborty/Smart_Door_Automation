import cv2
import numpy as np

# ROI coordinates
roi_points = np.array([
    (314, 73),   # Top-left
    (650, 73),   # Top-right
    (650, 714),  # Bottom-right
    (314, 714)   # Bottom-left
], np.int32)

def show_mouse_position(event, x, y, flags, param):
    frame = param.copy()
    
    # Draw ROI
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display mouse pointer position
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dot at pointer
        cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Resized RTSP Stream', frame)


cap = cv2.VideoCapture("rtsp://admin:DPDYWJ@192.168.0.202:554/H.264")
if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

cv2.namedWindow('Resized RTSP Stream')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 1080x720
    resized_frame = cv2.resize(frame, (1080, 720))

    # Set mouse callback function
    cv2.setMouseCallback('Resized RTSP Stream', show_mouse_position, resized_frame)

    # Draw ROI on the resized frame
    cv2.polylines(resized_frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the frame
    cv2.imshow('Resized RTSP Stream', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()