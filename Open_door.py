import sys
import cv2
import numpy as np
import time
import threading
import queue
import requests
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from dotenv import load_dotenv
import os

# Load environment variables from .env file
#pip install python-dotenv
load_dotenv()

# Get values from the env file
rtsp_url = os.getenv("RTSP_URL_Door")
door_url = os.getenv("DOOR_URL")

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8x.pt')

# ROI coordinates (rectangular ROI)
roi_points = np.array([
    (314, 73),    # Top-left
    (583, 73),    # Top-right
    (583, 714),   # Bottom-right
    (314, 714)    # Bottom-left
], np.int32)

# Global flags and variables
running = True
door_triggered = False
detection_start_time = None  # Global timer for continuous detection
door_lock = threading.Lock()

def resize_frame_to(frame, width, height):
    """Resize frame to fixed dimensions."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def reconnect_stream(url, attempts=5, delay=1):
    """Attempt to reconnect to the RTSP stream with a few retries."""
    cap = cv2.VideoCapture(url, apiPreference=cv2.CAP_FFMPEG)
    attempt = 0
    while not cap.isOpened() and attempt < attempts:
        print(f"Reconnect attempt {attempt+1}/{attempts}...")
        time.sleep(delay)
        cap = cv2.VideoCapture(url)
        attempt += 1
    return cap

def capture_thread_func(cap, frame_queue):
    global running
    while running:
        ret, frame = cap.read()
        if ret:
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass

def yolo_thread_func(model, frame_queue, detection_queue, roi_points, width, height, conf_level):
    global running
    # ROI offset
    x1, y1 = roi_points[0]
    x2, y2 = roi_points[2]
    while running:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        resized_frame = resize_frame_to(frame, width, height)
        if y2 > resized_frame.shape[0] or x2 > resized_frame.shape[1]:
            continue
        roi_frame = resized_frame[y1:y2, x1:x2]
        y_hat = model.predict(roi_frame, conf=conf_level, classes=[0], device=0, verbose=False)
        boxes = y_hat[0].boxes.xyxy.cpu().numpy()
        detections = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            conf = float(box[4]) if len(box) > 4 else 1.0
            w = xmax - xmin
            h = ymax - ymin
            xmin_full = xmin + x1
            ymin_full = ymin + y1
            # Format as: ([x, y, w, h], confidence)
            detections.append(([xmin_full, ymin_full, w, h], conf))
        detection_queue.put((resized_frame, detections))

def trigger_door():
    """Hit the door URL to open the door."""
    print("Triggering door open.")
    try:
        response = requests.get(door_url)
        print("Door triggered, response code:", response.status_code)
    except Exception as e:
        print("Error triggering door:", e)

def tracking_thread_func(tracker, detection_queue, display_queue, trails, max_trail_length):
    global running, door_triggered, detection_start_time
    while running:
        try:
            resized_frame, detections = detection_queue.get(timeout=1)
        except queue.Empty:
            continue
        current_time = time.time()
        tracks = tracker.update_tracks(detections, frame=resized_frame)
        active_tracks = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            active_tracks += 1
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cv2.rectangle(resized_frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"ID: {track_id}", (int(l), int(t)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cx = int((l + r) / 2)
            cy = int((t + b) / 2)
            if track_id not in trails:
                trails[track_id] = []
            trails[track_id].append((cx, cy))
            if len(trails[track_id]) > max_trail_length:
                trails[track_id].pop(0)
            for i in range(1, len(trails[track_id])):
                cv2.line(resized_frame, trails[track_id][i-1], trails[track_id][i], (255, 0, 0), 2)

        # Door trigger logic: use a global detection_start_time
        if active_tracks > 0:
            # Start the timer if not already set
            if detection_start_time is None:
                detection_start_time = current_time
            elapsed = current_time - detection_start_time
            # Display the timer above the ROI
            timer_position = (roi_points[0][0], max(roi_points[0][1] - 20, 20))
            cv2.putText(resized_frame, f"Timer: {elapsed:.1f}s", timer_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            # Trigger door only once when elapsed reaches 3 seconds
            if active_tracks!=0:
                if elapsed >= 1.5 and not door_triggered:
                    threading.Thread(target=trigger_door, daemon=True).start()
                    door_triggered = True
        else:
            detection_start_time = None
            door_triggered = False

        # Overlay ROI boundary on the frame
        overlay = resized_frame.copy()
        cv2.polylines(overlay, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.addWeighted(overlay, 0.1, resized_frame, 0.9, 0, resized_frame)
        display_queue.put(resized_frame)


def display_thread_func(display_queue):
    global running
    while running:
        try:
            frame = display_queue.get(timeout=1)
        except queue.Empty:
            continue
        cv2.imshow("RTSP People Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            running = False
            break

def main():
    global running
    width, height = 1080, 720
    conf_level = 0.8
    max_trail_length = 50

    # Initialize DeepSORT tracker and trail storage
    tracker = DeepSort(max_age=30)
    trails = {}

    cap = cv2.VideoCapture(rtsp_url, apiPreference=cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error opening RTSP stream: {rtsp_url}")
        return

    # Create queues for pipeline stages
    frame_queue = queue.Queue(maxsize=5)
    detection_queue = queue.Queue(maxsize=5)
    display_queue = queue.Queue(maxsize=5)

    # Start threads
    capture_thread = threading.Thread(target=capture_thread_func, args=(cap, frame_queue), daemon=True)
    yolo_thread = threading.Thread(target=yolo_thread_func, args=(model, frame_queue, detection_queue, roi_points, width, height, conf_level), daemon=True)
    track_thread = threading.Thread(target=tracking_thread_func, args=(tracker, detection_queue, display_queue, trails, max_trail_length), daemon=True)
    disp_thread = threading.Thread(target=display_thread_func, args=(display_queue,), daemon=True)

    capture_thread.start()
    yolo_thread.start()
    track_thread.start()
    disp_thread.start()

    # Wait for display thread to finish (ESC pressed)
    disp_thread.join()

    running = False
    capture_thread.join()
    yolo_thread.join()
    track_thread.join()

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting main thread.")

if __name__ == "__main__":
    main()
    print("Python version:", sys.version)
