import darknet.darknet as dn
dn.set_gpu(0)
net = dn.load_net(b"darknet/lp/lp.cfg", b"darknet/lp/lp_3000.weights", 0)
meta = dn.load_meta(b"darknet/lp/lp.data")

import cv2, os, re
import numpy as np
from recog_char import recognize_plate
expand=10
path='test/'
for file in os.listdir(path):
    if re.split('[.]', file)[-1]!='jpg':
        continue
    img=cv2.imread(path+file)
    yolo_image, arr = dn.array_to_image(img)
    r=dn.detect_image(net, meta, yolo_image)
    for obj in r:
        x_center=int(round(obj[2][0]))
        y_center=int(round(obj[2][1]))
        width=int(round(obj[2][2]))
        height=int(round(obj[2][3]))
        xmin=x_center-int(round(width/2))
        ymin=y_center-int(round(height/2))
        xmax=x_center+int(round(width/2))
        ymax=y_center+int(round(height/2))
        crop=img[ymin-expand if ymin-expand >=0 else 0:ymax+expand if ymax+expand<img.shape[0] else img.shape[0]-1, 
                 xmin-expand if xmin-expand >=0 else 0:xmax+expand if xmax+expand<img.shape[1] else img.shape[1]-1,
                 :]
        
        crop=cv2.imencode('.jpg', crop)[1].tostring()
        crop = np.fromstring(crop, np.uint8)
        crop = cv2.imdecode(crop, cv2.IMREAD_COLOR)

        recognize_plate(crop)