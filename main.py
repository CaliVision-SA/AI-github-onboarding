import cv2
import numpy as np
from ultralytics import YOLO
from utils.human_class import humanClass, object_to_id_map
from src.match_bounding_boxes import match_bounding_boxes
import mediapipe as mp

# Initialize MediaPipe Pose
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

        # YOLO tracking
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

        # Annotate frames with bounding boxes and pose
        annotate_frame(frame1, ids1, class_ids1, confidences1, bboxes1, pose)
        annotate_frame(frame2, ids2, class_ids2, confidences2, bboxes2, pose)

        cv2.imshow("YOLOv8 + MediaPipe Pose Stream 1", frame1)
        cv2.imshow("YOLOv8 + MediaPipe Pose Stream 2", frame2)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # Release
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def annotate_frame(frame, ids, class_ids, confidences, bboxes, pose_estimator):
    for tracker_id, class_id, conf, bbox in zip(ids, class_ids, confidences, bboxes):
        if class_id == 0:  # Person
            human_object = object_to_id_map.get(tracker_id, None)
            if human_object is None:
                human_object = humanClass()
                object_to_id_map[tracker_id] = human_object

            bounding_box_color, delta_time = human_object.detected()

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_box_color, 2)
            cv2.putText(frame, f'ID: {tracker_id} || dt: {delta_time:.2f}s', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bounding_box_color, 2)

            # Crop the detected person's region
            person_crop = frame[y1:y2, x1:x2]

            # Convert the BGR image to RGB before processing with MediaPipe
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(person_crop_rgb)

            # Draw pose landmarks if detected
            if results.pose_landmarks:
                # Drawing landmarks on the cropped image
                mp_drawing.draw_landmarks(
                    person_crop, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # To visualize on the main frame, we must overlay the annotations back
                # Extract landmarks in crop coordinates and map them to original frame coordinates
                h, w, _ = person_crop.shape
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if landmark.visibility > 0.5:
                        lx = int(landmark.x * w)
                        ly = int(landmark.y * h)
                        # Map back to original frame
                        frame_x = x1 + lx
                        frame_y = y1 + ly
                        cv2.circle(frame, (frame_x, frame_y), 5, (0, 255, 0), -1)

def draw_pose_on_crop(crop_frame, landmarks):
    # If you need to draw the pose directly on the cropped frame
    mp_drawing.draw_landmarks(
        crop_frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

# Initialize video sources
# cap1 = cv2.VideoCapture(r"videos\test_video.mp4")
# cap1 = cv2.VideoCapture(r"videos\test_video.mp4")
cap1 = cv2.VideoCapture(r"C:\Users\sJohnson\Downloads\codebase-20241209T173548Z-001\codebase\mvp.mp4")
cap2 = cv2.VideoCapture(r"C:\Users\sJohnson\Downloads\codebase-20241209T173548Z-001\codebase\mvp.mp4")


# Load YOLO models
model1 = YOLO("yolov8n.pt")
model2 = YOLO("yolov8n.pt")

# Process
process_stream(cap1, cap2, model1, model2)
