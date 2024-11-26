
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, simpledialog

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    return cap

def get_start_frame():
    # Create a root window to calculate the screen center
    root = tk.Tk()
    root.withdraw()

    start_frame = simpledialog.askinteger("Start Frame", "Enter the start frame:")
    return start_frame

def main():
    # Create the root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Prompt the user to select video files
    video_paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[("Video files", "*.avi;*.mp4")]
    )

    # # Prompt the user to select video files
    # video_paths += filedialog.askopenfilenames(   
    #     title="Select Video Files",
    #     filetypes=[("Video files", "*.avi;*.mp4")]
    # )

    print(video_paths)
    

    if not video_paths:
        print("No video files selected. Exiting.")
        return

    # Extract the current working directory from the first video path
    cwd = os.path.dirname(video_paths[0])

    # Get the start frame from the user
    frame_num = get_start_frame()
    step = 25
    resize_scale = 0.5
    show_corners = 0 

    while True:
        # Load video frames
        if len(video_paths) > 1:
            cap1 = load_video(video_paths[0])   
            cap2 = load_video(video_paths[1])
        else: 
            cap1 = load_video(video_paths[0])
            cap2 = load_video(video_paths[0])

        if cap1 is None or cap2 is None:
            break

        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        res1, frame1 = cap1.read()

        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        res2, frame2 = cap2.read()

        if not res1 or not res2:
            print(f"End of video reached. Exiting.")
            break

        if show_corners:
            ret, corners = cv2.findChessboardCorners(frame1, (8, 5), None)
            frame1corners = frame1
            if ret:
                cv2.drawChessboardCorners(frame1corners, (8, 5), corners, ret)

            frame2corners = frame2
            ret, corners = cv2.findChessboardCorners(frame2, (9, 6), None)
            if ret:
                cv2.drawChessboardCorners(frame2corners, (9, 6), corners, ret)

        frame1_rs = cv2.resize(frame1, (0, 0), None, resize_scale, resize_scale)
        frame2_rs = cv2.resize(frame2, (0, 0), None, resize_scale, resize_scale)

        frames = np.hstack((frame1_rs, frame2_rs))

        font_size = 0.3
        colour = (0, 0, 0)
        cv2.putText(frames, ('Step: ' + str(step)), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
        cv2.putText(frames, 'W/S - Change Step', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
        cv2.putText(frames, 'A/D - Prev/Next', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
        cv2.putText(frames, '1/2/3 - Save 1/2/Both', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
        cv2.putText(frames, 'Q - Quit', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
        cv2.imshow('Synced Frames', frames)
        cv2.setWindowTitle('Synced Frames', ('Frame ' + str(frame_num)))

        key = cv2.waitKey(33)
        if key == ord('d'):
            frame_num += 1 * step
        elif key == ord('a'):
            frame_num -= 1 * step
        elif key == ord('s'):
            step -= 1
            if step < 1:
                step = 1
        elif key == ord('w'):
            step += 1
        elif key == ord('c'):
            if show_corners == 0:
                show_corners = 1
            else:
                show_corners = 0
        elif key == ord('1'):
            try:
                if not os.path.exists(os.path.join(cwd, 'in_calib1')):
                    os.makedirs(os.path.join(cwd, 'in_calib1'))
            except OSError:
                continue
            cv2.imwrite(os.path.join(cwd, 'in_calib1', (str(frame_num) + '.jpg')), frame1)
        elif key == ord('2'):
            try:
                if not os.path.exists(os.path.join(cwd, 'in_calib2')):
                    os.makedirs(os.path.join(cwd, 'in_calib2'))
            except OSError:
                continue
            cv2.imwrite(os.path.join(cwd, 'in_calib2', (str(frame_num) + '.jpg')), frame2)
        elif key == ord('3'):
            try:
                if not os.path.exists(os.path.join(cwd, 'ex_calib1')) and not os.path.exists(
                        os.path.join(cwd, 'ex_calib2')):
                    os.makedirs(os.path.join(cwd, 'ex_calib1'))
                    os.makedirs(os.path.join(cwd, 'ex_calib2'))
            except OSError:
                continue
            cv2.imwrite(os.path.join(cwd, 'ex_calib1', (str(frame_num) + '.jpg')), frame1)
            cv2.imwrite(os.path.join(cwd, 'ex_calib2', (str(frame_num) + '.jpg')), frame2)
        elif key == ord('q'):
            break

    # Release video capture objects
    if cap1 is not None:
        cap1.release()
    if cap2 is not None:
        cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()