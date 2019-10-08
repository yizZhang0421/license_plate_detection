import os, cv2, re
import numpy as np

def ocr(binary_img):
    # binary_img=binary[y:y+h, x:x+w]
    max_match=-1
    result=None
    filename=None
    for file in os.listdir('template'):
        template_img=cv2.threshold(cv2.imread('template/'+file, cv2.IMREAD_GRAYSCALE),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
        template_answer=re.split('[.]', file)[0][-1]
        what_img = cv2.resize(binary_img, (int(round(binary_img.shape[1]*template_img.shape[0]/binary_img.shape[0])), template_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        ret2,what_img = cv2.threshold(what_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if template_img.shape[1]>what_img.shape[1]:
            template_img=template_img[:, int(round((template_img.shape[1]/2)-(what_img.shape[1]/2))):int(round((template_img.shape[1]/2)+(what_img.shape[1]/2)))]
        else:
            what_img=what_img[:, int(round((what_img.shape[1]/2)-(template_img.shape[1]/2))):int(round((what_img.shape[1]/2)+(template_img.shape[1]/2)))]
        match=np.sum(template_img==what_img)
        if match>max_match:
            max_match=match
            result=template_answer
            filename=file
    #print(filename)
    return result

'''
cv2.imshow('test', template_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('test', what_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''