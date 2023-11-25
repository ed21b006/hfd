import cv2, numpy as np
from imutils.face_utils import FACIAL_LANDMARKS_68_IDXS
import sys, os
sys.path.append(os.getcwd())
from FaceRecognitionModule.dlib_model import FaceModule

class FaceAlign:

    def __init__(self, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None):
        # self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        (self.lStart, self.lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, landmark):
        landmark = np.array(landmark)
        leftEyePts = landmark[self.lStart:self.lEnd]
        rightEyePts = landmark[self.rStart:self.rEnd]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        eyesCenter = (int(leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            int(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        return output


if __name__=='__main__':
    o=FaceAlign()
    obj=FaceModule()
    frame=cv2.imread(r'C:\materials\IITM\3rd year\HFD project\codes\test_images\HarikrishnaRangam.jpeg')
    try:
        landmarks=obj.landmarks_tuples_list if obj.landmarks_tuples_list!=None else obj.find_landmarks(frame)
        landmark = landmarks[0]
        img=o.align(frame, landmark)
    except IndexError:
        ...
        # y=
        # img=frame[y:y+h, x:x+h]
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# from imutils.face_utils import FaceAligner
# from FaceRecognitionModule.dlib_model import FaceModule

# obj = FaceModule()
# fa = FaceAligner(obj.pose_predictor)

# def face_align():
