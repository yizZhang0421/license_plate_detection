import darknet.darknet as dn
dn.set_gpu(0)
net = dn.load_net(b"lp.cfg", b"lp_3000.weights", 0)
meta = dn.load_meta(b"lp.data")

filename_index=1
import cv2, os
expand=10
for file in os.listdir('test'):
    img=cv2.imread('test/'+file)
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
        crop=img[ymin-expand:ymax+expand, xmin-expand:xmax+expand, :]
        cv2.imwrite('croped/'+str(filename_index)+'.jpg', crop)
        filename_index+=1