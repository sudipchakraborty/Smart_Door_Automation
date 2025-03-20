import sys
import os
import subprocess
import urllib.request
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from IPython.display import Video
from dotenv import load_dotenv
import os
# # Download and prepare the video
# path_zip = 'https://github.com/freedomwebtech/roiinyolo/raw/main/vid1.zip'
# print("Downloading video zip...")
# urllib.request.urlretrieve(path_zip, "vid1.zip")
# print("Unpacking video zip...")
# shutil.unpack_archive('vid1.zip')
load_dotenv()
video_path = os.getenv("RTSP_URL_Door")

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8x.pt')
dict_classes = model.model.names

# Auxiliary functions
def risize_frame(frame, scale_percent):
    """Resize an image by a percentage scale."""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def filter_tracks(centers, patience):
    """Keep only the latest frames for each tracked object."""
    return {k: dict(list(i.items())[-patience:]) for k, i in centers.items()}

def update_tracking(centers_old, obj_center, thr_centers, lastKey, frame, frame_max):
    """Update tracking of objects based on distance and frame number."""
    is_new = 0
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
    lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
    previous_pos = [(k, obj_center) for k, center in lastpos if np.linalg.norm(np.array(center) - np.array(obj_center)) < thr_centers]
    
    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = obj_center
    else:
        id_obj = 'ID' + (str(int(lastKey.split('D')[1]) + 1) if lastKey else '0')
        is_new = 1
        centers_old[id_obj] = {frame: obj_center}
        lastKey = id_obj
    return centers_old, id_obj, is_new, lastKey

# Detection of people in ROI from video file
def detect_people_in_video(video_path, output_video_name='result.mp4',
                           scale_percent=100, conf_level=0.8, thr_centers=20,
                           frame_max=5, patience=100, alpha=0.1, verbose=False):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    print('[INFO] - Original Dim:', (width, height))

    if scale_percent != 100:
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        print('[INFO] - Dim Scaled:', (width, height))

    output_path = "rep_" + output_video_name
    tmp_output_path = "tmp_" + output_path
    VIDEO_CODEC = "MP4V"

    output_video = cv2.VideoWriter(tmp_output_path,
                                   cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                                   fps, (width, height))

    class_IDS = [0]
    centers_old = {}
    count_p = 0
    lastKey = ''
    frames_list = []
    print(f'[INFO] - Verbose during Prediction: {verbose}')

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = video.read()
        if not ret:
            break

        frame = risize_frame(frame, scale_percent)
        area_roi = [np.array([(1250, 400), (750, 400), (700, 800), (1200, 800)], np.int32)]
        ROI = frame[390:800, 700:1300]

        y_hat = model.predict(ROI, conf=conf_level, classes=class_IDS, device=0, verbose=False)
        boxes = y_hat[0].boxes.xyxy.cpu().numpy()
        conf = y_hat[0].boxes.conf.cpu().numpy()
        classes = y_hat[0].boxes.cls.cpu().numpy()

        positions_frame = pd.DataFrame(
            np.hstack((boxes, conf.reshape(-1, 1), classes.reshape(-1, 1))),
            columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class']
        )

        for ix, row in positions_frame.iterrows():
            xmin, ymin, xmax, ymax, _, _ = row[['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class']]
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            center_x, center_y = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
            centers_old, id_obj, is_new, lastKey = update_tracking(centers_old, (center_x, center_y),
                                                                    thr_centers, lastKey, i, frame_max)
            count_p += is_new

            cv2.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            for cx, cy in centers_old[id_obj].values():
                cv2.circle(ROI, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(ROI, f"{id_obj}:{np.round(conf[ix], 2)}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 1)

        cv2.putText(frame, f'Counts People in ROI: {count_p}', (30, 40),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0), 1)

        centers_old = filter_tracks(centers_old, patience)
        overlay = frame.copy()
        cv2.polylines(overlay, pts=area_roi, isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.fillPoly(overlay, area_roi, (255, 0, 0))
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        frames_list.append(frame)
        output_video.write(frame)

    video.release()
    output_video.release()

    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(
        ["ffmpeg", "-i", tmp_output_path, "-crf", "18", "-preset", "veryfast",
         "-hide_banner", "-loglevel", "error", "-vcodec", "libx264", output_path]
    )
    os.remove(tmp_output_path)

    print(f"Processed video saved as: {output_path}")

    # for i in [62, 63, 64, 65, 66]:
    #     if i < len(frames_list):
    #         plt.figure(figsize=(14, 10))
    #         plt.imshow(cv2.cvtColor(frames_list[i], cv2.COLOR_BGR2RGB))
    #         plt.show()
    frac = 0.7 
    cap = cv2.VideoCapture("rep_result.mp4")
    if not cap.isOpened():
        print("Error opening video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Output Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    detect_people_in_video(video_path)
    print("Python version:", sys.version)
