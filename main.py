from humanDetection import HumanDetectionModel


video_path = "recording.mov"  
human_detection_model = HumanDetectionModel()
human_detection_model.process_video(video_path)