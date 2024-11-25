import torch
import cv2
from datetime import datetime


class HumanDetectionModel:
    def __init__(self, model_name='yolov5s', person_class_id=0):
        """
        Initialize the human detection model with a YOLO model.
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.person_class_id = person_class_id
        self.tracked_people = {}

    def update_tracked_people(self, detections, frame_time):
        """
        Update the tracking dictionary
        """
        updated_people = {}

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            if int(class_id) != self.person_class_id:
                continue

            person_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

            if person_id in self.tracked_people:
                updated_people[person_id] = self.tracked_people[person_id]
            else:
                updated_people[person_id] = frame_time

        self.tracked_people = updated_people

    def get_box_color(self, person_id, frame_time):
        """
        Determine the color of the bounding box based on tracking duration.
        """
        start_time = self.tracked_people[person_id]
        duration = (frame_time - start_time).total_seconds()

        if duration <= 2:
            return (0, 0, 255)  
        elif 2 < duration <= 5:
            return (0, 165, 255) 
        else:
            return (0, 255, 0)  

    def process_video(self, video_path):
        """
        Detect humans by processing the video 
        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter('/videos', fourcc, 30.0, (640, 480)) 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = datetime.now()
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            self.update_tracked_people(detections, frame_time)

            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                if int(class_id) != self.person_class_id:
                    continue

                person_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                color = self.get_box_color(person_id, frame_time)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"{int((frame_time - self.tracked_people[person_id]).total_seconds())}s",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            #save the video
            out.write(frame) 

            cv2.imshow('Human Detection and Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()  
        cv2.destroyAllWindows()

