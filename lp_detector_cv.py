import cv2
net = cv2.dnn.readNetFromDarknet("darknet/lp/lp.cfg", "darknet/lp/lp.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread("test/lp11.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255, (224, 224), (0, 0, 0), False, crop=False) 
net.setInput(blob)
outs = net.forward(output_layers)
outs[0].shape
outs[1].shape
outs[2].shape
outs[2][0]
len(outs)

height, width, channels = img.shape
import numpy as np
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('test.jpg', img)