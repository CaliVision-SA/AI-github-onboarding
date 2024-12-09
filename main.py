import cv2
import numpy as np
from ultralytics import YOLO
from utils.human_class import humanClass, object_to_id_map
from src.match_bounding_boxes import match_bounding_boxes
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_stream(cap1: cv2.VideoCapture, cap2: cv2.VideoCapture, model1: YOLO, model2: YOLO):
    FSM1 = humanClass()
    FSM2 = humanClass()

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break  # Exit loop if there's a problem with either frame

        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))

        # YOLOv8 detection and tracking
        result1 = model1.track(frame1, persist=True)[0]
        result2 = model2.track(frame2, persist=True)[0]

        boxes1 = result1.boxes
        boxes2 = result2.boxes

        ids1 = boxes1.id.cpu().numpy().astype(int)
        class_ids1 = boxes1.cls.cpu().numpy().astype(int)
        confidences1 = boxes1.conf.cpu().numpy()
        bboxes1 = boxes1.xyxy.cpu().numpy().astype(int)

        ids2 = boxes2.id.cpu().numpy().astype(int)
        class_ids2 = boxes2.cls.cpu().numpy().astype(int)
        confidences2 = boxes2.conf.cpu().numpy()
        bboxes2 = boxes2.xyxy.cpu().numpy().astype(int)

        matched_tracker_ids = match_bounding_boxes(
            frame_content_1=[ids1, class_ids1, confidences1, bboxes1],
            frame_content_2=[ids2, class_ids2, confidences2, bboxes2]
        )

        # Extract landmarks for frame1 and frame2
        landmarks_list_1 = get_landmarks(frame1, ids1, class_ids1, confidences1, bboxes1, pose)
        landmarks_list_2 = get_landmarks(frame2, ids2, class_ids2, confidences2, bboxes2, pose)

        # Annotate frames with bounding boxes and landmarks
        annotate_frame(frame1, ids1, class_ids1, confidences1, bboxes1, landmarks_list_1)
        annotate_frame(frame2, ids2, class_ids2, confidences2, bboxes2, landmarks_list_2)

        cv2.imshow("YOLOv8 + MediaPipe Pose Stream 1", frame1)
        cv2.imshow("YOLOv8 + MediaPipe Pose Stream 2", frame2)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def get_landmarks(frame, ids, class_ids, confidences, bboxes, pose_estimator):
    """
    For each detected person, run MediaPipe pose estimation and return a list of:
    [tracker_id, list_of_landmarks], where list_of_landmarks is a list of tuples (x, y, visibility).

    Coordinates returned will be in the original frame's coordinate space.
    """
    landmarks_per_person = []

    for tracker_id, class_id, conf, bbox in zip(ids, class_ids, confidences, bboxes):
        if class_id == 0:  # person
            x1, y1, x2, y2 = bbox
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Convert to RGB for MediaPipe
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(person_crop_rgb)

            if results.pose_landmarks:
                h, w, _ = person_crop.shape
                current_landmarks = []
                for lm in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to absolute pixel values
                    px = x1 + (lm.x * w)
                    py = y1 + (lm.y * h)
                    # Scale z by the width of the bounding box
                    pz = lm.z * w  
                    current_landmarks.append((px, py, pz, lm.visibility))

                landmarks_per_person.append([tracker_id, current_landmarks])

    return landmarks_per_person

def annotate_frame(frame, ids, class_ids, confidences, bboxes, landmarks_per_person):
    """
    Draw bounding boxes and pose landmarks onto the frame.
    The landmarks_per_person is a list of [tracker_id, landmarks].
    """
    # Convert landmarks_per_person into a dict for easy access
    landmarks_dict = {item[0]: item[1] for item in landmarks_per_person}

    for tracker_id, class_id, conf, bbox in zip(ids, class_ids, confidences, bboxes):
        if class_id == 0:  # person
            human_object = object_to_id_map.get(tracker_id, None)
            if human_object is None:
                human_object = humanClass()
                object_to_id_map[tracker_id] = human_object

            bounding_box_color, delta_time = human_object.detected()

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_box_color, 2)
            cv2.putText(frame, f'ID: {tracker_id} || dt: {delta_time:.2f}s', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bounding_box_color, 2)

            # If we have landmarks for this tracker ID, draw them
            if tracker_id in landmarks_dict:
                for (lx, ly, vis) in landmarks_dict[tracker_id]:
                    if vis > 0.5:
                        cv2.circle(frame, (lx, ly), 5, (0, 255, 0), -1)

# Example usage:
# Initialize video sources
# cap1 = cv2.VideoCapture(r"videos\test_video.mp4")
# cap1 = cv2.VideoCapture(r"videos\test_video.mp4")
cap1 = cv2.VideoCapture(r"C:\Users\sJohnson\Downloads\codebase-20241209T173548Z-001\codebase\mvp.mp4")
cap2 = cv2.VideoCapture(r"C:\Users\sJohnson\Downloads\codebase-20241209T173548Z-001\codebase\mvp.mp4")


# Load YOLO models
model1 = YOLO("yolov8n.pt")
model2 = YOLO("yolov8n.pt")

process_stream(cap1, cap2, model1, model2)
