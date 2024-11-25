import torch
import cv2
from datetime import datetime

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


PERSON_CLASS_ID = 0
tracked_people = {}


def update_tracked_people(detections, frame_time):
    """
    Update the tracking dictionary with the detected bounding boxes and times.
    """
    global tracked_people
    updated_people = {}

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if int(class_id) != PERSON_CLASS_ID:
            continue
        #unique id based on bounding boxes
        person_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
        if person_id in tracked_people:
            updated_people[person_id] = tracked_people[person_id]
        else:
            updated_people[person_id] = frame_time

    tracked_people = updated_people