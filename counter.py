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

# Library Constants
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkPoints = mp.solutions.pose.PoseLandmark
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Counter:
    def __init__(self):
        self.count = -1

        # Create the pose detector
        base_options = BaseOptions(model_asset_path='data/pose_landmarker_lite.task')
        options = PoseLandmarkerOptions(base_options=base_options)
        self.detector = PoseLandmarker.create_from_options(options)

        # Load video
        self.video = cv2.VideoCapture(0)

        self.first = time.time()

        self.completed = False

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
    def checkPullUp(self, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            leftHand = pose_landmarks[19]
            rightHand = pose_landmarks[20]
            leftElbow = pose_landmarks[13]
            rightElbow = pose_landmarks[14]
            mouth = pose_landmarks[10]

            if leftElbow.y > leftHand.y and rightElbow.y > rightHand.y and leftHand.y > mouth.y and rightHand.y > mouth.y and self.completed is False:
                self.count += 1
                self.completed = True

            if self.completed and leftHand.y < mouth.y and rightHand.y < mouth.y:
                self.completed = False

            
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        self.time = time.time()
        # Run until we close the video
        while self.video.isOpened():
            self.time = time.time()

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the pose
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # # Draw time onto screen
            # cv2.putText(image,
            #             "Time: " + str(self.time),
            #             (50, 100),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=1,
            #             color=(0, 255, 0),
            #             thickness=2)

            # Draw the pose landmarks
            self.draw_landmarks_on_body(image, results)
            
            # Increase pull up count if pull up is detected
            self.checkPullUp(results)

            if self.count != -1:
                # Display pull up count on screen
                cv2.putText(image,
                            "count: " + str(self.count),
                            (50, 100),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 255, 0),
                            thickness=2)
            
            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Pose Tracking', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break    

        # Release our video and close all windows
        self.video.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":        
    c = Counter()
    c.run()