from humanDetection import HumanDetectionModel

# Looks very neat! I like the class oriented approach that you have taken.
video_path = "recording.mov"  
human_detection_model = HumanDetectionModel()
human_detection_model.process_video(video_path)