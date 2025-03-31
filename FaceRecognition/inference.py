import argparse, pickle, threading, time, queue
import numpy as np, pandas as pd, cv2, tensorflow as tf
from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import ArcFace

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", required=True)
ap.add_argument("-m", "--model", default='models/model.h5')
ap.add_argument("-c", "--conf", type=float, default=0.9)
ap.add_argument("-lm", "--liveness_model", default='models/liveness.model')
ap.add_argument("-le", "--label_encoder", default='models/le.pickle')
args = vars(ap.parse_args())

# Load Models
face_rec_model = load_model(args["model"], compile=False)
arcface_model = ArcFace.loadModel()
target_size = arcface_model.input_shape[1:3]
liveness_model = tf.keras.models.load_model(args["liveness_model"])
label_encoder = pickle.loads(open(args["label_encoder"], "rb").read())
detector = MTCNN()

# Threaded Video Reader
frame_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()

def read_frames(source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    while not stop_event.is_set():
        success, frame = cap.read()
        if success:
            if not frame_queue.full():
                frame_queue.put(frame)
    cap.release()

# Start thread
source = int(args["source"]) if args["source"].isdigit() else args["source"]
thread = threading.Thread(target=read_frames, args=(source,))
thread.start()

# Frame skip logic
frame_skip = 3
frame_count = 0

# FPS tracking
fps = 0
start_time = time.time()

try:
    while True:
        if frame_queue.empty():
            continue
        img = frame_queue.get()
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        img = cv2.resize(img, (1080, 720))
        detections = detector.detect_faces(img)

        if detections:
            for detect in detections:
                bbox = detect.get('box', [])
                if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                    continue  # skip invalid bbox
                xmin, ymin = int(bbox[0]), int(bbox[1])
                xmax, ymax = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

                # Crop and Liveness
                img_roi = img[ymin:ymax, xmin:xmax]
                if img_roi.size == 0: continue
                try:
                    face = cv2.resize(img_roi, (32, 32)).astype("float32") / 255.0
                except:
                    continue
                face_array = tf.keras.preprocessing.image.img_to_array(face)
                face_prepro = np.expand_dims(face_array, axis=0)
                decision = np.argmax(liveness_model(face_prepro)[0].numpy())

                if decision == 0:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(img, 'Fake', (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    continue

                # Face Alignment & Recognition
                right_eye = detect['keypoints']['right_eye']
                left_eye = detect['keypoints']['left_eye']
                norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                resized = cv2.resize(norm_img_roi, target_size).astype("float32") / 255.0
                img_pixels = np.expand_dims(tf.keras.preprocessing.image.img_to_array(resized), axis=0)
                embedding = arcface_model.predict(img_pixels)[0]

                data = pd.DataFrame([embedding], columns=np.arange(512))
                prediction = face_rec_model.predict(data)[0]
                class_id = prediction.argmax()
                confidence = prediction[class_id]

                if confidence > args["conf"]:
                    pose_class = label_encoder.classes_[class_id]
                    color = (0, 255, 0)
                else:
                    pose_class = "Unknown Person"
                    color = (0, 0, 255)

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, pose_class, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # FPS Overlay
        end_time = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (end_time - start_time))
        start_time = end_time
        cv2.putText(img, f'FPS: {fps:.2f}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    stop_event.set()
    thread.join()
    cv2.destroyAllWindows()
    print("[INFO] Inference ended.")
