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
load_dotenv()

# Get values from the env file
rtsp_url = os.getenv("RTSP_URL_Door")
door_url = os.getenv("DOOR_URL")

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8x.pt')

# ROI coordinates (rectangular ROI)
roi_points = np.array([
    (372, 401),    # Top-left
    (687, 401),    # Top-right
    (687, 714),    # Bottom-right
    (380, 714)     # Bottom-left
], np.int32)

# Global flags and variables
running = True
door_triggered = False
detection_start_time = None  # Global timer for continuous detection
door_lock = threading.Lock()

def resize_frame_to(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def capture_thread_func(rtsp_url, frame_queue):
    global running
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print(f"Error opening RTSP stream: {rtsp_url}")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("[WARNING] Dropped invalid/corrupt frame")
            time.sleep(0.01)
            continue

        frame = resize_frame_to(frame, 1080, 720)

        if frame_queue.full():
            try:
                _ = frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            frame_queue.put(frame, timeout=0.01)
        except queue.Full:
            pass

        time.sleep(0.03)

def yolo_thread_func(model, frame_queue, detection_queue, roi_points, width, height, conf_level):
    global running
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

        y_hat = model.predict(roi_frame, conf=conf_level, classes=[0], device="cpu", verbose=False)
        boxes = y_hat[0].boxes.xyxy.cpu().numpy()
        detections = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            conf = float(box[4]) if len(box) > 4 else 1.0
            w = xmax - xmin
            h = ymax - ymin
            xmin_full = xmin + x1
            ymin_full = ymin + y1
            detections.append(([xmin_full, ymin_full, w, h], conf))
        detection_queue.put((resized_frame, detections))

def trigger_door():
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

        if active_tracks > 0:
            if detection_start_time is None:
                detection_start_time = current_time
            elapsed = current_time - detection_start_time
            timer_position = (roi_points[0][0], max(roi_points[0][1] - 20, 20))
            cv2.putText(resized_frame, f"Timer: {elapsed:.1f}s", timer_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if not door_triggered:
                threading.Thread(target=trigger_door, daemon=True).start()
                door_triggered = True
        else:
            detection_start_time = None
            door_triggered = False

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
        if key == 27:
            running = False
            break

def main():
    global running
    width, height = 1080, 720
    conf_level = 0.8
    max_trail_length = 50

    tracker = DeepSort(max_age=30)
    trails = {}

    frame_queue = queue.Queue(maxsize=5)
    detection_queue = queue.Queue(maxsize=5)
    display_queue = queue.Queue(maxsize=5)

    capture_thread = threading.Thread(target=capture_thread_func, args=(rtsp_url, frame_queue), daemon=True)
    yolo_thread = threading.Thread(target=yolo_thread_func, args=(model, frame_queue, detection_queue, roi_points, width, height, conf_level), daemon=True)
    track_thread = threading.Thread(target=tracking_thread_func, args=(tracker, detection_queue, display_queue, trails, max_trail_length), daemon=True)
    disp_thread = threading.Thread(target=display_thread_func, args=(display_queue,), daemon=True)

    capture_thread.start()
    yolo_thread.start()
    track_thread.start()
    disp_thread.start()

    disp_thread.join()

    running = False
    capture_thread.join()
    yolo_thread.join()
    track_thread.join()
    print("Exiting main thread.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
