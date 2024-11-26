import time


object_to_id_map = {}

class humanClass:
    def __init__(self):
        self.initially_detected = time.time()

    def detected(self):
        current_time = time.time()
        delta = current_time - self.initially_detected

        if delta < 2:  # Red
            return (0, 0, 255), delta
        elif 2 <= delta < 5:  # Orange
            return (0, 165, 255), delta
        else:  # Green
            return (0, 255, 0), delta





        