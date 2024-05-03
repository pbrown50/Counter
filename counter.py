"""
A pull-up counter/form checker

@author: Parker Brown
@version: April 2024
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
import math
import pygame

# Library Constants
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkPoints = mp.solutions.pose.PoseLandmark
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Counter:
    def __init__(self):
        self.count = 0

        # Create the pose detector
        base_options = BaseOptions(model_asset_path='data/pose_landmarker_lite.task')
        options = PoseLandmarkerOptions(base_options=base_options)
        self.detector = PoseLandmarker.create_from_options(options)

        # Load video
        self.video = cv2.VideoCapture(0)

        self.first = time.time()

        self.completed = False
        self.started = False

        self.deadHang = False

    def draw_landmarks_on_body(self, image, detection_result):
        """
        Draws all the landmarks on the body
        Args:
            image (Image): Image to draw on
            detection_result (PoseLandmarkerResult): PoseLandmarker detection results
        """
        # Get a list of the landmarks
        pose_landmarks_list = detection_result.pose_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

             # Save the landmarks into a NormalizedLandmarkList
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       pose_landmarks_proto,
                                       solutions.pose.POSE_CONNECTIONS,
                                       solutions.drawing_styles.get_default_pose_landmarks_style())
    def checkForm(self, image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            leftHand = pose_landmarks[19]
            rightHand = pose_landmarks[20]
            leftElbow = pose_landmarks[13]
            rightElbow = pose_landmarks[14]
            leftHip = pose_landmarks[23]
            rightHip = pose_landmarks[24]
            leftKnee = pose_landmarks[25]
            rightKnee = pose_landmarks[26]
            leftShoulder = pose_landmarks[11]
            rightShoulder = pose_landmarks[12]
            mouth = pose_landmarks[10]
            leftAngle = self.calculate_angle(leftHand, leftElbow, leftShoulder)
            rightAngle = self.calculate_angle(rightHand, rightElbow, rightShoulder)
            messages = []
            repTime = time.time() - self.first
            if leftAngle < 12 and rightAngle < 12:
                self.deadHang = True
            if leftElbow.y > leftHand.y and rightElbow.y > rightHand.y and leftHand.y > mouth.y and rightHand.y > mouth.y and self.completed is False and self.deadHang:
                self.count += 1
                self.completed = True
                self.started = True
                self.deadHang = False
                self.first = time.time()

                pygame.mixer.init()
                pygame.mixer.music.load("data/Goal_Scored.wav")
                pygame.mixer.music.play()
            elif repTime > 4:
                messages.append("DEAD HANG --> CHIN OVER BAR TO COUNT A REP!")

            if self.completed and leftHand.y < mouth.y and rightHand.y < mouth.y:
                self.completed = False

            if self.started:
                if ((leftElbow.x < leftHand.x - 0.07 or leftElbow.x > leftHand.x + 0.07) or (rightElbow.x > rightHand.x + 0.07 and rightElbow.x < rightHand.x - 0.07)):
                    messages.append("KEEP ELBOWS BELOW HANDS FOR OPTIMAL FORM!")

                # Check if there's excessive forward/backward sway, CHATGPT HELPED
                if abs(leftKnee.z - leftHip.z) > 0.06 or abs(rightKnee.z - rightHip.z) > 0.06:
                    messages.append("DON'T SWAY BACK AND FORTH!")

                # Check if there's excessive side-to-side sway, CHATGPT HELPED
                if abs(leftKnee.x - leftHip.x) > 0.015 or abs(rightKnee.x - rightHip.x) > 0.015:
                    messages.append("DON'T SWAY SIDE TO SIDE!")

            y = 150
            for message in messages:
                cv2.putText(image,
                            message,
                            (50, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2)
                y += 50

    # CHAT GPT HELPED WITH THE BELOW METHOD
    def calculate_angle(self, hand, elbow, shoulder):
        # Calculate the lengths of the sides of the triangle
        a = math.sqrt((elbow.x - shoulder.x)**2 + (elbow.y - shoulder.y)**2)
        b = math.sqrt((hand.x - shoulder.x)**2 + (hand.y - shoulder.y)**2)
        c = math.sqrt((hand.x - elbow.x)**2 + (hand.y - elbow.y)**2)

        # Use the Law of Cosines to find the angle at the elbow joint (in radians)
        angle_rad = math.acos((b**2 + c**2 - a**2) / (2 * b * c))

        # Convert radians to degrees
        angle_deg = math.degrees(angle_rad)

        return angle_deg
   
   
    # CHAT GPT USED TO COMBINE PROTOCOL AND RUN METHODS
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """
        self.time = time.time()
        # Initialize protocol message display flag
        show_protocol = True

        # Use the default camera (change the argument if you have multiple cameras)
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            self.time = time.time()
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                break

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the pose
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the pose landmarks
            self.draw_landmarks_on_body(image, results)

            if show_protocol:
                # Display protocol messages
                protocol = [
                    "PROTOCOL:",
                    "-  PLACE CAMERA 5 ft FROM BAR",
                    "-  PLACE CAMERA 5 ft HIGH",
                    "-  PLACE CAMERA PARALLEL TO YOUR BODY",
                    "",
                    "",
                    "   PRESS 'SPACE' WHEN THE ABOVE ACTIONS HAVE",
                    "   BEEN COMPLETED AND START PULL-UPS"
                ]
                y = 150
                for message in protocol:
                    cv2.putText(image,
                                message,
                                (50, y),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(225, 0, 0),
                                thickness=2)
                    y += 50  # Increment y after each message
                # Reset y for the next iteration
                y = 150

                # Break the loop if the user presses 'SPACE'
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    show_protocol = False

            # If not showing protocol, perform pull-up counting
            else:
                # Increase pull-up count if pull-up is detected and correct form
                self.checkForm(image, results)

                # Display pull-up count on screen
                cv2.putText(image,
                            "COUNT: " + str(self.count) + "                                         PRESS 'R' TO RESTART",
                            (50, 100),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(225, 0, 0),
                            thickness=2)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    self.count = 0

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Pose Tracking', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print("COUNT: " + str(self.count) + "    THANKS FOR PLAYING!")
                break    

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":        
    c = Counter()
    c.run()