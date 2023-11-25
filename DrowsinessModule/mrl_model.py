import numpy as np
from tflite_runtime.interpreter import Interpreter
import os,sys
sys.path.append(os.getcwd())
import variables


class MRL:
    def __init__(self):
        self.interpreter = Interpreter(model_path=variables.MODELS_PATH+'mrl_model.tflite')
        self.interpreter.allocate_tensors()
        self.input_tensor_index = self.interpreter.get_input_details()[0]['index']
        self.output = self.interpreter.tensor(self.interpreter.get_output_details()[0]['index'])

    def predict(self, X_input):
        input_data = X_input.astype(np.float32)
        self.interpreter.set_tensor(self.input_tensor_index, input_data)
        self.interpreter.invoke()
        output = self.output()[0]
        return output

if __name__=='__main__':
    import cv2
    o=MRL()
    img_size = 224
    im=r'C:\materials\IITM\3rd year\HFD project\codes\datasets\mrlEyes_2018_01\s0037\s0037_10257_1_1_1_0_0_01.png'
    img_array = cv2.imread(im,cv2.IMREAD_GRAYSCALE)
    backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    new_array = cv2.resize(backtorgb, (img_size, img_size))
    X_input = np.array(new_array).reshape(1, img_size, img_size, 3)
    X_input = X_input/255.0
    prediction = o.predict(X_input)
    print(prediction)
    del o