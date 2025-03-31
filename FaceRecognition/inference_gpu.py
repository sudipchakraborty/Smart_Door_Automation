import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import ArcFace

# Enable memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("[WARNING] No GPU found. Running on CPU.")

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", type=str, required=True, help="path to Video or webcam")
ap.add_argument("-m", "--model", type=str, default='models/model.h5', help="path to .h5 model")
ap.add_argument("-c", "--conf", type=float, default=0.9, help="min prediction confidence")
ap.add_argument("-lm", "--liveness_model", type=str, default='models/liveness_model.h5', help="path to liveness model")
ap.add_argument("-le", "--label_encoder", type=str, default='models/le.pickle', help="path to label encoder")
args = vars(ap.parse_args())

# Video source
source = int(args["source"]) if args["source"].isnumeric() else args["source"]

# Load models
face_rec_model = load_model(args['model'], compile=True)
liveness_model = load_model(args['liveness_model'], compile=True)
arcface_model = ArcFace.loadModel()
target_size = arcface_model.input_shape[1:3]
label_encoder = pickle.loads(open(args['label_encoder'], "rb").read())

detector = MTCNN()
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

executor = ThreadPoolExecutor(max_workers=2)

def process_face(norm_img_roi):
    resized = cv2.resize(norm_img_roi, target_size)
    img_pixels = tf.keras.preprocessing.image.img_to_array(resized)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_norm = img_pixels / 255.0
    with tf.device('/GPU:0'):
        embedding = arcface_model.predict(img_norm, verbose=0)[0]
    return embedding

print("[INFO] Starting Inference...")
prev_time = time.time()
fps = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Failed to read frame")
        break

    resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    detections = detector.detect_faces(resized_frame)

    if len(detections) > 0:
        for detect in detections:
            bbox = detect['box']
            xmin, ymin = int(bbox[0]*2), int(bbox[1]*2)
            xmax, ymax = int((bbox[0]+bbox[2])*2), int((bbox[1]+bbox[3])*2)
            face = frame[ymin:ymax, xmin:xmax]

            # Liveness detection
            face_resized = cv2.resize(face, (32, 32))
            face_input = np.expand_dims(tf.keras.preprocessing.image.img_to_array(face_resized) / 255.0, axis=0)
            with tf.device('/GPU:0'):
                preds = liveness_model.predict(face_input, verbose=0)[0]
            decision = np.argmax(preds)

            if decision == 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(frame, 'Fake', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                keypoints = detect['keypoints']
                norm_face = alignment_procedure(frame, keypoints['left_eye'], keypoints['right_eye'], [xmin, ymin, bbox[2]*2, bbox[3]*2])
                future = executor.submit(process_face, norm_face)
                embedding = future.result()

                data = pd.DataFrame([embedding], columns=np.arange(512))
                with tf.device('/GPU:0'):
                    prediction = face_rec_model.predict(data, verbose=0)[0]

                if max(prediction) > args['conf']:
                    class_id = np.argmax(prediction)
                    name = label_encoder.classes_[class_id]
                    color = (0, 255, 0)
                else:
                    name = 'Unknown'
                    color = (0, 0, 255)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Inference Stopped")