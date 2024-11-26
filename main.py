import cv2
from ultralytics import YOLO
from utils.human_class import humanClass, object_to_id_map
import threading
from src.match_bounding_boxes import match_bounding_boxes

def process_stream(cap1, cap2, model1, model2):
    FSM1 = humanClass()
    FSM2 = humanClass()
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break  # Exit loop if there's a problem with either frame

        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))

        # Perform object detection and tracking on both frames
        result1 = model1.track(frame1, persist=True)[0]
        result2 = model2.track(frame2, persist=True)[0]

        boxes1 = result1.boxes  # Boxes object for bbox outputs
        boxes2 = result2.boxes  # Boxes object for bbox outputs

        ids1 = boxes1.id.cpu().numpy().astype(int)  # Tracker IDs
        class_ids1 = boxes1.cls.cpu().numpy().astype(int)  # Class IDs
        confidences1 = boxes1.conf.cpu().numpy()  # Confidence scores
        bboxes1 = boxes1.xyxy.cpu().numpy().astype(int)  # Bounding boxes

        ids2 = boxes2.id.cpu().numpy().astype(int)  # Tracker IDs
        class_ids2 = boxes2.cls.cpu().numpy().astype(int)  # Class IDs
        confidences2 = boxes2.conf.cpu().numpy()  # Confidence scores
        bboxes2 = boxes2.xyxy.cpu().numpy().astype(int)  # Bounding boxes

        matched_tracker_ids = match_bounding_boxes(frame_content_1=[ids1, class_ids1, confidences1, bboxes1], frame_content_2=[ids2, class_ids2, confidences2, bboxes2])

        # Annotate the frames with tracking results
        annotate_frame(frame1, ids1, class_ids1, confidences1, bboxes1)
        annotate_frame(frame2, ids2, class_ids2, confidences2, bboxes2)

        # Display the annotated frames
        cv2.imshow("YOLOv8 Tracking Stream 1", frame1)
        cv2.imshow("YOLOv8 Tracking Stream 2", frame2)

        # Exit on pressing 'q'
        if cv2.waitKey(300) & 0xFF == ord('q'): #making it 300ms to avoid over using your CPU
            break

    # Release resources
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def annotate_frame(frame, ids, class_ids, confidences, bboxes):


        for tracker_id, class_id, conf, bbox in zip(ids, class_ids, confidences, bboxes):
            if class_id == 0:
                human_object = object_to_id_map.get(tracker_id, None)
                if human_object is None:
                    human_object = humanClass()
                    object_to_id_map[tracker_id] = human_object

                bounding_box_color, delta_time = human_object.detected()

                x1, y1, x2, y2 = bbox

                cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_box_color, 2)
                cv2.putText(frame, f'ID: {tracker_id} || delta time: {delta_time:2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, bounding_box_color, 2)

# Initialize webcam captures
cap1 = cv2.VideoCapture(r"videos\test_video.mp4")
cap2 = cv2.VideoCapture(r"videos\test_video.mp4")

# Load the YOLOv8 model
model1 = YOLO("yolov8n.pt")
model2 = YOLO("yolov8n.pt")

# Process streams in sync
process_stream(cap1, cap2, model1, model2)
