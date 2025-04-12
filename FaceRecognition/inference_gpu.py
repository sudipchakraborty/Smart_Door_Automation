import os
import cv2
import time
import queue
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import pickle
from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import ArcFace

# Enable GPU memory growth early
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU memory growth enabled")
    except RuntimeError as e:
        print(f"[WARNING] GPU memory config error: {e}")
else:
    print("[WARNING] No GPU found. Running on CPU.")

# === Argument Parsing ===
default_source = os.getenv('RTSP_URL_LIVING_ROOM', None)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", type=str, default=default_source,
                required=(default_source is None),
                help="Video/webcam source (defaults to env RTSP_URL_LIVING_ROOM)")
ap.add_argument("-m", "--model", type=str, default='models/model.h5')
ap.add_argument("-c", "--conf", type=float, default=0.9)
ap.add_argument("-lm", "--liveness_model", type=str, default='models/liveness_model.h5')
ap.add_argument("-le", "--label_encoder", type=str, default='models/le.pickle')
args = vars(ap.parse_args())

source = int(args["source"]) if args["source"].isnumeric() else args["source"]

# === Load Models ===
face_rec_model = load_model(args['model'], compile=True)
liveness_model = load_model(args['liveness_model'], compile=True)
arcface_model = ArcFace.loadModel()
target_size = arcface_model.input_shape[1:3]
label_encoder = pickle.loads(open(args['label_encoder'], "rb").read())

detector = MTCNN()

# === Video Capture ===
cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_queue = queue.Queue(maxsize=5)
frame_skip = 2
frame_counter = 0

def capture_frames():
    global frame_counter
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Failed to capture frame")
            break
        frame_counter += 1
        if frame_counter % frame_skip == 0 and not frame_queue.full():
            frame_queue.put(frame)

def process_frames():
    prev_time = time.time()
    tracker = None
    tracking = False
    bbox = None

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            resized_frame = cv2.resize(frame, (1080, 720))

            if not tracking:
                detections = detector.detect_faces(resized_frame)
                if len(detections) > 0:
                    detect = detections[0]
                    # After detecting the face
                    bbox = detect['box']
                    bbox = tuple(map(int, bbox))  # Convert all elements to integers
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(resized_frame, bbox)
                    tracking = True
            else:
                success, bbox = tracker.update(resized_frame)
                if not success:
                    tracking = False
                    continue
                else:
                    bbox = list(map(int, bbox))
                    xmin, ymin, w, h = bbox
                    xmax, ymax = xmin + w, ymin + h
                    face = resized_frame[ymin:ymax, xmin:xmax]

                    face_resized = cv2.resize(face, (32, 32))
                    face_input = np.expand_dims(tf.keras.preprocessing.image.img_to_array(face_resized) / 255.0, axis=0)

                    with tf.device('/GPU:0'):
                        preds = liveness_model.predict(face_input, verbose=0)[0]
                    decision = np.argmax(preds)

                    if decision == 0:
                        color = (0, 0, 255)
                        label = 'Fake'
                    else:
                        keypoints = {'left_eye': (xmin + w//3, ymin + h//3), 'right_eye': (xmin + 2*w//3, ymin + h//3)}
                        norm_face = alignment_procedure(resized_frame, keypoints['left_eye'], keypoints['right_eye'], bbox)
                        norm_face_resized = cv2.resize(norm_face, target_size)
                        img_pixels = tf.keras.preprocessing.image.img_to_array(norm_face_resized)
                        img_pixels = np.expand_dims(img_pixels, axis=0) / 255.0

                        with tf.device('/GPU:0'):
                            embedding = arcface_model.predict(img_pixels, verbose=0)[0]
                            prediction = face_rec_model.predict(pd.DataFrame([embedding], columns=np.arange(512)), verbose=0)[0]

                        if max(prediction) > args['conf']:
                            class_id = np.argmax(prediction)
                            label = label_encoder.classes_[class_id]
                            color = (0, 255, 0)
                        else:
                            label = 'Unknown'
                            color = (0, 0, 255)

                    cv2.rectangle(resized_frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(resized_frame, label, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(resized_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Face Recognition", resized_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

# === Launch Threads ===
t1 = threading.Thread(target=capture_frames)
t2 = threading.Thread(target=process_frames)
t1.daemon = True
t2.daemon = True
t1.start()
t2.start()
t1.join()
t2.join()

cap.release()
cv2.destroyAllWindows()
