from pathlib import Path
import face_recognition
import pickle
from variables import *

def encode_known_faces(training_directory = DATABASE_PATH, model = "hog", encodings_location = DEFAULT_ENCODINGS_PATH):
    names = []
    encodings = []

    for filepath in Path(training_directory).glob("*/*"):
        name = filepath.parent.name
        print(f"Encoding for {name}")
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model=model)  #1 face detect only
        face_encodings = face_recognition.face_encodings(image, face_locations, model = 'large')

        for encoding in face_encodings: # Only one if only 1 face is available in the image
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
    print("COMPLETED!")