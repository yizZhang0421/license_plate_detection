from keras.models import load_model
model=load_model('keras_model/lp_char.h5')
class_label={
        0:'0',
        1:'1',
        2:'2',
        3:'3',
        4:'4',
        5:'5',
        6:'6',
        7:'7',
        8:'8',
        9:'9',
        10:'A',
        11:'B',
        12:'C',
        13:'D',
        14:'E',
        15:'F',
        16:'G',
        17:'H',
        18:'I',
        19:'J',
        20:'K',
        21:'L',
        22:'M',
        23:'N',
        24:'P',
        25:'Q',
        26:'R',
        27:'S',
        28:'T',
        29:'U',
        30:'V',
        31:'W',
        32:'X',
        33:'Y',
        34:'Z'
        }
import cv2
import numpy as np
def ocr(binary_char):
    #binary_char=char
    binary_char = cv2.resize(binary_char, (64, 64), interpolation=cv2.INTER_CUBIC)
    binary_char=np.reshape(binary_char, (64, 64, 1))
    binary_char=binary_char/255.0
    binary_char=np.array([binary_char])
    result=model.predict_classes(binary_char)
    return class_label[result[0]]