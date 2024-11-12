Table of Contents

    Purpose of This Task
    Project Overview
    GitHub Requirements
    AI Project Requirements
    Submission Guidelines
    Resources
    Conclusion

Purpose of This Task
    Hi all! This task simply test your github proficiency, just to ensure that you have the basic skills to work with github within a team.

    Collaborative development is a cornerstone of our work environment. Proficiency with Git and GitHub ensures that everyone can contribute effectively, manage code changes, and maintain project integrity. This task will assess and enhance your ability to:

        1) Use basic Git commands (add, commit, push, etc.)
        2) Work with branches to manage feature development
        3) Collaborate through pull requests and code reviews

AI Engagement


Project Overview
Objective

Create an AI application that:

    1) Detects humans in a video using a pre-trained object detection model (e.g., YOLO)
    2) Implements the logic to track how long each person remains in the frame
    3) Changes the color of each person's bounding box based on the duration they have been in the frame:
        Red: 0-2 seconds
        Orange: 2-5 seconds
        Green: 5 or more seconds

GitHub Requirements

To simulate a real-world collaborative environment, please follow these GitHub practices:

    Clone the Repository
        Clone your forked repository to your local machine.
        This is the url to the github repo: https://github.com/CaliVision-SA/AI-github-onboarding.git

    Create a Branch
        Create a new branch named yourname_test-branch. (eg. of branch name: "Shaun_Johnson-branch")

    Implement the Project
        Develop your AI application on your branch.

    Commit
        When you are done with the project, commit and push your code to your branch.


AI Project Requirements

    Human Detection
        Use a pre-trained object detection model to detect humans in each frame of the video.

    Finite State Machine (FSM)
        Implement the logic to track how long each detected person remains within the frame.
        Time thresholds for state transitions:
            0-2 seconds: Red
            2-5 seconds: Orange
            5+ seconds: Green

    Performance Considerations
        Optimize your code for real-time processing, if possible.


    Tips:
        1) Use an object orientated approach to keeping track of humans
        2) Use a yolo model to do the detections, and use its built in functionality to track people
        3) Please see the video within the "videos" folder to see an example of how the output should look
