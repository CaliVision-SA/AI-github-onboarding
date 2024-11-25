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


def get_box_color(person_id, frame_time):
    """
    Bounding boxes colors based on the people tracked 
    """
    start_time = tracked_people[person_id]
    duration = (frame_time - start_time).total_seconds()

    if duration <= 2:
        return (0, 0, 255)  
    elif 2 < duration <= 5:
        return (0, 165, 255)  
    else:
        return (0, 255, 0)  
    
def process_video(video_path):
    """
    Process a video, detect humans, and display their bounding boxes with time-based color coding.
    """
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = datetime.now()
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()
        update_tracked_people(detections, frame_time)
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            if int(class_id) != PERSON_CLASS_ID:
                continue

            person_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
            color = get_box_color(person_id, frame_time)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                frame,
                f"{int((frame_time - tracked_people[person_id]).total_seconds())}s",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        cv2.imshow('Human Detection and Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()