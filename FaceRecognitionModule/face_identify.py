import os, sys
import time
sys.path.append(os.getcwd())
from pathlib import Path
import face_recognition
import pickle
from variables import DEFAULT_ENCODINGS_PATH
from collections import Counter

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]

def recognize_faces(image_location, model = "hog", encodings_location = DEFAULT_ENCODINGS_PATH):

    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        print(name, bounding_box)
    return name, bounding_box

if __name__ == '__main__':
    import cv2
    for test_image in os.listdir('test_images'):
        name, box = recognize_faces('test_images/'+test_image)
        img = cv2.imread('test_images/'+test_image)
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0,255,0))
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
