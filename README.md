# AI GitHub Onboarding Task

## Table of Contents
1. [Purpose of This Task](#purpose-of-this-task)
2. [Project Overview](#project-overview)
3. [GitHub Requirements](#github-requirements)
4. [AI Project Requirements](#ai-project-requirements)
5. [Submission Guidelines](#submission-guidelines)
6. [Resources](#resources)
7. [Conclusion](#conclusion)

---

## Purpose of This Task

Hi all! This task is designed to test your GitHub proficiency, ensuring you have the basic skills required to work effectively within a team.

Collaborative development is a cornerstone of our work environment. Proficiency with Git and GitHub ensures that everyone can contribute effectively, manage code changes, and maintain project integrity. This task will assess and enhance your ability to:

1. Use basic Git commands (e.g., `add`, `commit`, `push`, etc.).
2. Work with branches to manage feature development.
3. Collaborate through pull requests and code reviews.

---

## Project Overview

Your task is to create an AI application that:

1. Detects humans in a video using a pre-trained object detection model (e.g., YOLO).
2. Implements the logic to track how long each person remains in the frame.
3. Changes the color of each person's bounding box based on the duration they have been in the frame:
   - **Red**: 0-2 seconds
   - **Orange**: 2-5 seconds
   - **Green**: 5+ seconds

---

## GitHub Requirements

To simulate a real-world collaborative environment, please follow these GitHub practices:

1. **Clone the Repository**
   - Clone your forked repository to your local machine.
   - Repository URL: [https://github.com/CaliVision-SA/AI-github-onboarding.git](https://github.com/CaliVision-SA/AI-github-onboarding.git)

2. **Create a Branch**
   - Create a new branch named `<yourname_test-branch>` (e.g., `Shaun_Johnson-branch`).

3. **Implement the Project**
   - Develop your AI application on your branch.

4. **Commit**
   - When you are done with the project, commit and push your code to your branch. We would like to see progressive commits during this exercise - so please make sure to commit and push in increments. 

---

## AI Project Requirements

1. **Human Detection**
   - Use a pre-trained object detection model (e.g., YOLO) to detect humans in each frame of the video.

2. **Finite State Machine (FSM)**
   - Implement logic to track how long each detected person remains within the frame.
   - Time thresholds for state transitions:
     - **Red**: 0-2 seconds
     - **Orange**: 2-5 seconds
     - **Green**: 5+ seconds

3. **Performance Considerations**
   - Optimize your code for real-time processing, if possible.

### Tips:
- Use an object-oriented approach to keep track of humans.
- Use a YOLO model to perform detections and leverage its built-in functionality to track people.

---

## Submission Guidelines

1. Create a pull request from your branch to the main branch of the repository.
2. Add meaningful comments in your pull request to explain your changes.
3. Ensure your code is clean, well-documented, and follows the project's style guidelines.

---

## Resources

- [YOLO Documentation](https://pjreddie.com/darknet/yolo/)
- [GitHub Workflow Guide](https://guides.github.com/introduction/flow/)

---

## Conclusion

This task is an opportunity to demonstrate your GitHub skills and AI development abilities. Please ensure that your implementation adheres to the requirements and guidelines outlined above. Happy coding!
