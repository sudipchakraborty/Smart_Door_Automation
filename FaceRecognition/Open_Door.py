import sys
import cv2
import numpy as np
import time
import threading
import requests
from dotenv import load_dotenv
import os
import tensorflow as tf
from keras.models import load_model
import pickle
from mtcnn import MTCNN
import ArcFace
from my_utils import alignment_procedure  # Ensure alignment_procedure is accessible
from queue import Queue

# Load environment variables from .env file
load_dotenv()

# Get RTSP stream URL and door URL from environment variables
rtsp_url = os.getenv("RTSP_URL_DOOR")
door_url = os.getenv("DOOR_URL")  # Fetch gate URL from env DOOR_URL

# ---------------------------
# FACE RECOGNITION SETUP
# ---------------------------
print("Loading Face Recognition models...")
face_rec_model = load_model("models/model.h5", compile=True)
liveness_model = load_model("models/liveness_model.h5", compile=True)
arcface_model = ArcFace.loadModel()
# Use the target input size for ArcFace model (e.g., (112, 112))
target_size = arcface_model.input_shape[1:3]
with open("models/le.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# Set the recognition confidence threshold (0.85 as specified)
CONF_THRESHOLD = 0.85
# Authorized person label
AUTHORIZED_LABEL = "Sudip"


# ---------------------------
# MTCNN Face Detector
# ---------------------------
detector = MTCNN()

# ---------------------------
# Gate Trigger Setup
# ---------------------------
last_trigger_time = 0
COOLDOWN = 10  # seconds

# Variables to prevent spamming
gate_triggered = False
last_authorized_detection_time = 0
AUTHORIZED_RESET_TIMEOUT = 5  # seconds

def open_gate():
    """Send GET request to open the gate."""
    global last_trigger_time
    current_time = time.time()
    if current_time - last_trigger_time < COOLDOWN:
        return
    print("Authorized person detected. Triggering gate open...")
    try:
        response = requests.get(door_url, timeout=5)
        print("Gate triggered, response code:", response.status_code)
        last_trigger_time = current_time
    except Exception as e:
        print("Error triggering gate:", e)

# ---------------------------
# Multithreading Pipeline Setup
# ---------------------------
running = True
frame_queue = Queue(maxsize=5)
display_queue = Queue(maxsize=5)

# Define the fixed Region of Interest (ROI) for faster processing:
# ROI coordinates: top-left (357,164), bottom-right (624,408)
roi_x1, roi_y1 = 357, 164
roi_x2, roi_y2 = 624, 408

def capture_thread_func():
    global running
    # Use FFMPEG backend for RTSP capture
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error opening RTSP stream: {rtsp_url}")
        running = False
        return
    # Optionally, set a low buffer to reduce latency
    cap.set(cv2.CAP_PROP_FPS, 10)
    while running:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("[WARNING] Dropped corrupt or empty frame")
            continue
        if frame_queue.full():
            discarded = frame_queue.get()  # Discard oldest
            frame_queue.put(frame)
        # Resize full frame to 1080x720
        frame = cv2.resize(frame, (1080, 720))
        try:
            frame_queue.put(frame, timeout=1)
        except Exception:
            pass  # If queue is full, drop the frame
    cap.release()

def processing_thread_func():
    global running, gate_triggered, last_authorized_detection_time
    prev_time = time.time()
    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
        frame = frame_queue.get()
        # Draw ROI rectangle on full frame for visual reference
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
        # Crop the ROI: process only this region for face detection
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        # Run MTCNN on the ROI frame
        detections = detector.detect_faces(roi_frame)
        authorized_detected = False
        for detect in detections:
            # 'bbox' is relative to ROI; adjust to full frame coordinates
            bbox = detect['box']  # [x, y, width, height] in ROI coordinates
            x_roi, y_roi, w, h = bbox
            x = roi_x1 + x_roi
            y = roi_y1 + y_roi
            if w <= 0 or h <= 0:
                continue
            # Extract face ROI from full frame
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # --- Liveness Detection ---
            try:
                face_resized = cv2.resize(face, (32, 32))
            except Exception:
                continue
            face_input = np.expand_dims(tf.keras.preprocessing.image.img_to_array(face_resized) / 255.0, axis=0)
            with tf.device('/GPU:0'):
                preds = liveness_model.predict(face_input, verbose=0)[0]
            decision = np.argmax(preds)
            if decision == 0:  # Fake face detected
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Fake", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                continue

            # --- Face Recognition ---
            try:
                norm_face = alignment_procedure(frame, detect['keypoints']['left_eye'],
                                                detect['keypoints']['right_eye'], [x, y, w, h])
            except Exception as e:
                print("Alignment error:", e)
                continue
            try:
                norm_face_resized = cv2.resize(norm_face, target_size)
            except Exception:
                continue

            img_pixels = tf.keras.preprocessing.image.img_to_array(norm_face_resized)
            img_pixels = np.expand_dims(img_pixels, axis=0) / 255.0
            with tf.device('/GPU:0'):
                embedding = arcface_model.predict(img_pixels, verbose=0)[0]
            import pandas as pd
            data_df = pd.DataFrame([embedding], columns=np.arange(512))
            with tf.device('/GPU:0'):
                prediction = face_rec_model.predict(data_df, verbose=0)[0]

            if max(prediction) > CONF_THRESHOLD:
                class_id = int(np.argmax(prediction))
                label = label_encoder.classes_[class_id]
            else:
                label = "Unknown"

            color = (0, 255, 0) if label == AUTHORIZED_LABEL or label == "Sudip" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # If the recognized label is authorized, mark detection
            if label == AUTHORIZED_LABEL:
                authorized_detected = True

        # If authorized face ("Deep") detected and gate hasn't been triggered, trigger it
        if authorized_detected:
            last_authorized_detection_time = time.time()
            if not gate_triggered:
                threading.Thread(target=open_gate, daemon=True).start()
                gate_triggered = True
        else:
            # Reset trigger if authorized face is absent for the reset timeout period
            if time.time() - last_authorized_detection_time > AUTHORIZED_RESET_TIMEOUT:
                gate_triggered = False

        # Optional: Calculate and overlay FPS on the frame
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        try:
            display_queue.put(frame, timeout=1)
        except Exception:
            pass

def display_thread_func():
    global running
    while running:
        if display_queue.empty():
            time.sleep(0.01)
            continue
        frame = display_queue.get()
        cv2.imshow("Gate Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False
            break

def main():
    cap_thread = threading.Thread(target=capture_thread_func, daemon=True)
    proc_thread = threading.Thread(target=processing_thread_func, daemon=True)
    disp_thread = threading.Thread(target=display_thread_func, daemon=True)

    cap_thread.start()
    proc_thread.start()
    disp_thread.start()

    # Wait for the display thread to exit (on key press)
    disp_thread.join()
    time.sleep(1)
    cv2.destroyAllWindows()
    print("[INFO] Inference stopped.")

if __name__ == "__main__":
    main()
    print("Python version:", sys.version)
