import os, sys
sys.path.append(os.getcwd())

import dlib, cv2
import pickle
import numpy as np
import variables

class FaceModule():

    def __init__(self, model = '68'):

        self.face_encoder = dlib.face_recognition_model_v1(variables.MODELS_PATH + 'dlib_face_recognition_resnet_model_v1.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.last_person_name = None
        self.model = model

        if model == '68':
            self.pose_predictor = dlib.shape_predictor(variables.MODELS_PATH + 'shape_predictor_68_face_landmarks.dat')
        
        elif model == '5':
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


    def show_landmarks(self, image):

        try:
            detection = self.detector(image, 1)[0]
            landmarks = self.pose_predictor(image, detection)
            landmarks_as_tuples = [(p.x, p.y) for p in landmarks.parts()]

            for point in landmarks_as_tuples:
                image = cv2.circle(image, point, 3, (0,0,255), -1)

        except IndexError:
            pass

        cv2.imshow('CAPTURING..', image)


    def capture(self, person_name = 'unknown_name'):
        
        os.makedirs(variables.DATABASE_PATH + person_name, exist_ok=True)

        vid = cv2.VideoCapture(0)
        frame_counter = 0
        
        while(True):

            ret, frame = vid.read()

            if frame_counter % 5 == 0:
                self.save_frame(frame, person_name, "{}.png".format(frame_counter))

            frame_counter += 1
            
            self.show_landmarks(frame)

            if frame_counter == 30:
                break
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        self.last_person_name = person_name

        print(f'FACE CAPTURED FOR {person_name}!')

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
            
        print("FACE REGISTERED FOR {}!".format(person_name.upper()))


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

    obj = FaceModule(model = '68')
    obj.reset_encodings()
    # person_name = input("Enter person's name: ")
    # obj.capture(person_name)
    for person_name in os.listdir(variables.DATABASE_PATH):
        obj.train(person_name)
    for test_image in os.listdir('test_images'):
        name, box = obj.recognize_face('test_images/'+test_image)
        if box != []:
            img = cv2.imread('test_images/'+test_image)
            cv2.rectangle(img, box[:2], box[2:], (0,255,0))
            cv2.imshow(name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
