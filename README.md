database folder has all the registered faces in separate folders.

test_images folder has images for testing the model.

variables.py has few variables. It's necessary to first append system path with current directory(codes/) in a file to import variables in that file.

test.py is just for experimenting.

faces/ is the dataset from kaggle.

To install dlib, pip install <dlib_file_in_base_directory_name>. Use python 3.9 only

FaceRecognitionModule has 2 files. 
train.py has a function encode_known_faces which trains all the person's face in the database/ and update the encodings.pkl.
  //modify the code to not train all the person's face for every new person data. Directly append it.
face_identify.py recognizes the faces in the test_images/ and draw rectangles over it and displays.

FaceRegistrationModule has 2 files.
face_capture.py captures the video using webcam and stores 30 frames out of it into database.
face_registration.py is designed to execute face_capture.py as well as train.py.

//NEXT:
Extract all the functions of face_recognition library's api.py.
Add more models to the face_recognition_models/ directory of this library. Add models for different lighting conditions also or modify the existing database by applying filters.
Drowsiness and emotion models, mask, phone, distraction detection.
Alert system.
