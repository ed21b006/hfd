import os, sys
sys.path.append(os.getcwd())

import dlib, cv2
import pickle
import numpy as np
import variables

class FaceModule:

    def __init__(self, detector_model = 'hog', landmark_model = '68'):

        self.face_encoder = dlib.face_recognition_model_v1(variables.MODELS_PATH + 'dlib_face_recognition_resnet_model_v1.dat')
        self.detector_model = detector_model
        self.last_person_name = None
        self.landmark_model = landmark_model

        self.detections = None
        self.landmarks_tuples_list = None

        if detector_model == 'hog':
            self.detector = dlib.get_frontal_face_detector()
        elif detector_model == 'cnn':
            self.detector = dlib.cnn_face_detection_model_v1(variables.MODELS_PATH + 'mmod_human_face_detector.dat')

        if landmark_model == '68':
            self.pose_predictor = dlib.shape_predictor(variables.MODELS_PATH + 'shape_predictor_68_face_landmarks.dat')
        
        elif landmark_model == '5':
            self.pose_predictor = dlib.shape_predictor(variables.MODELS_PATH + 'shape_predictor_5_face_landmarks.dat')
        

    def save_encodings(self, name_encodings):

        if os.path.exists(variables.DEFAULT_ENCODINGS_PATH):

            with variables.DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
                loaded_encodings = pickle.load(f)

            name_encodings['names'].extend(loaded_encodings['names'])
            name_encodings['encodings'].extend(loaded_encodings['encodings'])

        with variables.DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
            pickle.dump(name_encodings, f)


    def save_frame(self, frame, person_name, filename):
        cv2.imwrite(variables.DATABASE_PATH + person_name + '/' + filename, frame)


    def find_landmarks(self, frame):

        try:
            landmarks_tuples_list = []
            detections = self.detector(frame, 1)
            self.detections = detections

            for detection in detections:
                if self.detector_model == 'hog':
                    landmarks = self.pose_predictor(frame, detection)
                elif self.detector_model == 'cnn':
                    landmarks = self.pose_predictor(frame, detection.rect)

                landmarks_tuples_list.append([(p.x, p.y) for p in landmarks.parts()])

            self.landmarks_tuples_list = landmarks_tuples_list
            return landmarks_tuples_list
    
        except IndexError:
            self.landmarks_tuples_list = []
            return []
            
        
    
    def show_landmarks(self, frame, landmarks_tuples_list):

        for landmarks_as_tuples in landmarks_tuples_list:
            for point in landmarks_as_tuples:
                frame = cv2.circle(frame, point, 3, (0,0,255), -1)


    def capture(self, person_name = 'unknown_name'):
        
        os.makedirs(variables.DATABASE_PATH + person_name, exist_ok=True)

        vid = cv2.VideoCapture(0)
        frame_counter = 0
        
        while(True):

            ret, frame = vid.read()

            if frame_counter % 5 == 0:
                self.save_frame(frame, person_name, "{}.png".format(frame_counter))

            frame_counter += 1
            
            self.find_landmarks(frame)

            self.show_landmarks(frame, self.landmarks_tuples_list)

            cv2.imshow('CAPTURING..', frame)

            if frame_counter == 30:
                break
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        self.last_person_name = person_name

        print(f'FACE CAPTURED FOR {person_name.upper()}')

        vid.release()
        cv2.destroyAllWindows()


    def reset_encodings(self):

        name_encodings = {'names':[], 'encodings':[]}

        with variables.DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
            pickle.dump(name_encodings, f)
    

    def train(self, person_name = None):

        encodings = list()
        names = list()

        if person_name is None:
            person_name = self.last_person_name

        for filename in os.listdir(variables.DATABASE_PATH + person_name):

            image_path = variables.DATABASE_PATH + person_name + '/' + filename
            img = cv2.imread(image_path)

            detections = self.detector(img, 1)

            for detection in detections:
                landmark = self.pose_predictor(img, detection)
                face_descriptor = self.face_encoder.compute_face_descriptor(img, landmark, num_jitters=2)
                encodings.append(np.array(face_descriptor))
                names.append(person_name)
        
        name_encodings = {"names": names, "encodings": encodings}
        
        self.save_encodings(name_encodings)
            
        print("FACE REGISTERED FOR {}".format(person_name.upper()))


    def recognize_face(self, image_path, tolerance = 0.6):

        with variables.DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
            known_encodings = pickle.load(f)

        if len(known_encodings['encodings']) == 0:
            print('NO DATA REGISTERED!')
            return None, []

        img = cv2.imread(image_path)

        detections = self.detector(img, 1)

        for detection in detections:

            landmark = self.pose_predictor(img, detection)
            face_descriptor = self.face_encoder.compute_face_descriptor(img, landmark, num_jitters=2)
            new_encoding = np.array(face_descriptor)

            distances = np.linalg.norm(known_encodings['encodings'] - new_encoding, axis=1)
            
            if min(distances) < tolerance:
                idx = np.argmin(distances)
                print(known_encodings['names'][idx])
                return known_encodings['names'][idx], [detection.left(), detection.top(), detection.right(), detection.bottom()]
            
            else:
                print('UNKNOWN!')
                return None, []



if __name__ == '__main__':

    obj = FaceModule(detector_model = 'hog', landmark_model = '68')
    # obj.reset_encodings()
    person_name = input("Enter person's name: ")
    obj.capture(person_name)
    # for person_name in os.listdir(variables.DATABASE_PATH):
    # obj.train('aryan')
    # for test_image in os.listdir('test_images'):
    #     name, box = obj.recognize_face('test_images/'+test_image)
    #     if box != []:
    #         img = cv2.imread('test_images/'+test_image)
    #         cv2.rectangle(img, box[:2], box[2:], (0,255,0))
    #         cv2.imshow(name, img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

