import os, sys
sys.path.append(os.getcwd())

from FaceRecognitionModule.train import encode_known_faces
# from face_capture import start_capturing

# person = start_capturing()
encode_known_faces()
