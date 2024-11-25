import torch
import cv2
from datetime import datetime


class HumanDetectionModel:
    def __init__(self, model_name='yolov5s', person_class_id=0): # For development, very often we use the most accurate model just to make development a bit easier. Not sure what the largest yolov5 model is but you would also have checked out yolov8-yolov11 models. They have some very cool features that comes in handy!
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
                continue # Good job checking if the class id is the person class id before proceeding (and using the continue keyword - fancyyyy): 

            person_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}" 
            # This seems to assume that the person will remain stationary within the frame - another way to approach this is to rather use the built in tracker from ultralytics (but I am not sure if yolov5 has that functionality tho - yolov8 does tho)
            # Remember that the x1, y1, x2, y2 coordinates is the coordinates of the bounding box of the person. So x1 and y1 is the pixel values of the top left corner, and x2, y2 is the pixel values of the bottom right corner.
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
        # The duration here will always be 0 (or just about always) because the tracker assumes that the human is completely stationary within the frame.
        # Your implementation here however is 100% perfect! I also did it the same way.

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
        # Get video properties

        # Instead of hardcoding the framerate and the resolution, you could have gathered the actual proparties of the video and used those instead (You can find these properties within the cap object) 
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter('/videos', fourcc, 30.0, (640, 480))  
        # The "/videos" value here should be the file path of the output video (which should include the file extension)

        # If you want to save the video within a folder in your route directory - then you can do that as shown below:
        # out = cv2.VideoWriter('video/output.mp4', fourcc, 30.0, (640, 480))  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = datetime.now()
            results = self.model(frame)

            # An alternative is to only make the model detect people through , but that is not really a big deal tho - I do not think that is increases the processing speed of the program.
            # results = self.model(frame, classes=[0])  # Only detect humans (class ID 0).

            detections = results.xyxy[0].cpu().numpy()

            self.update_tracked_people(detections, frame_time)

            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                if int(class_id) != self.person_class_id: 
                    continue 
                
                # You are running the same two lines of code here twice (once here, and the second time within the "self.update_tracked_people" function).
                # Instead you could rather move the "self.update_tracked_people" function call below this if statement (and remove the if int(class_id)... section that is contained within the "self.update_tracked_people" function).

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

