import os, sys
sys.path.append(os.getcwd())
import cv2
import time
from pathlib import Path
import face_recognition
from variables import DATABASE_PATH

try:
    number = 7
except:
    number = 1

os.makedirs(DATABASE_PATH + f"/person_{number}/")

def start_capturing():

    vid = cv2.VideoCapture(0)
    start = time.time()
    frame_counter = 0
    # It takes around 0.03 secs to capture 1 frame without any operation
    # And 0.2 secs per frame on showing landmarks

    while(True):

        ret, frame = vid.read()
        frame_counter += 1
        save_frame(frame, "{}.png".format(frame_counter))
        
        input_face_locations = face_recognition.face_locations(frame)
        landmarks = face_recognition.face_landmarks(frame, face_locations=input_face_locations)

        try:
            for k,v in landmarks[0].items():
                for points in v:
                    frame=cv2.circle(frame, points, 3, (0,0,255), -1)
        except IndexError:
            pass

        cv2.imshow('frame', frame)
        if frame_counter == 30:
            break
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    end = time.time()
    print(f'Time taken: {(end-start)/frame_counter} per frame. Frames: {frame_counter}. Shape: {frame.shape}')

    vid.release()
    cv2.destroyAllWindows()

    return f"/person_{number}/"

def save_frame(frame, filename):
    cv2.imwrite(DATABASE_PATH + f"/person_{number}/" + filename, frame)

if __name__ == "__main__":
    start_capturing()