import winsound
import cv2
from imutils import face_utils
import sys, os

import numpy as np
sys.path.append(os.getcwd())

class EAR:
    def __init__(self, threshold = 0.20, frame_check = 20):
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.threshold = threshold
        self.frame_check = frame_check
        self.flag = 0

    def eye_aspect_ratio(self, eye):
        A = self.calculate_distance(eye[1], eye[5])
        B = self.calculate_distance(eye[2], eye[4])
        C = self.calculate_distance(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def detect(self, frame, landmark):
        leftEye = landmark[self.lStart:self.lEnd]
        rightEye = landmark[self.rStart:self.rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(np.array(leftEye))
        rightEyeHull = cv2.convexHull(np.array(rightEye))
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, f'EAR: {ear}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if ear < self.threshold:
            self.flag += 1
            print(self.flag)

            if self.flag >= self.frame_check:
                cv2.putText(frame, "****************ALERT!****************", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print ("Drowsy")
                winsound.Beep(440, 100)
        else:
            self.flag = 0


if __name__ == '__main__':
    from FaceRecognitionModule.dlib_model import FaceModule
    obj_drows = EAR(0.4)
    obj = FaceModule()
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        landmarks_tuples_list = obj.find_landmarks(frame)
        obj.show_landmarks(frame, landmarks_tuples_list)
        try:
            obj_drows.detect(frame, landmarks_tuples_list[0])
        except Exception as e:
            print(e)
        cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    vid.release()
    cv2.destroyAllWindows()
    del obj
    del obj_drows
