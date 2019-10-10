import cv2
import numpy as np
from keras_ocr import ocr

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def rotate_bound(image, angle):
    #獲取寬高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # 提取旋轉矩陣 sin cos 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # 計算影象的新邊界尺寸
    nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
    nH = h
 
    # 調整旋轉矩陣
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew(binary_im, origin, max_skew=10):
    height, width= binary_im.shape
    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        binary_im, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )
    
    if isinstance(lines, np.ndarray)==False:
        return binary_im

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return binary_im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            binary_im = cv2.rotate(binary_im, cv2.ROTATE_90_CLOCKWISE)
            origin = cv2.rotate(origin, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            binary_im = cv2.rotate(binary_im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            origin = cv2.rotate(origin, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    binary_im = cv2.warpAffine(binary_im, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    origin = cv2.warpAffine(origin, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return (binary_im, origin)

def recognize_plate(img):
    #img=crop
    # lighter
    img=increase_brightness(img)

    # resize and denoise
    if img.shape[1]<366:
        img = cv2.resize(img, (366, int(round(img.shape[0]*366/img.shape[1]))), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (366, int(round(img.shape[0]*366/img.shape[1]))), interpolation=cv2.INTER_AREA)
    img = cv2.fastNlMeansDenoising(img, None, 35, 7, 21)

    # enhance
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 5))
    img = cv2.erode(img,kernel)

    # gray and binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret2,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # fix tilt
    binary, img=deskew(binary, img)
 
    # find contour
    same_row=[]
    contours, _ = cv2.findContours(binary , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        contour=contours[i]
        x,y,w,h=cv2.boundingRect(contour)
        finded=False
        for row in same_row:
            if y>=row['min_y_top'] and y<=row['min_y_bottom'] and y+h-1>=row['max_y_top'] and y+h-1<=row['max_y_bottom']:
                mask = np.full((binary.shape[0], binary.shape[1]), 255).astype(np.uint8)
                cv2.drawContours(mask, contours, i, color=0, thickness=-1)
                row['member'].append((x,y,w,h,mask))
                finded=True
                break
        if finded==False:
            threshold=10
            d=dict()
            d['min_y_top']=y-threshold
            d['min_y_bottom']=y+threshold
            d['max_y_top']=y+h-1-threshold
            d['max_y_bottom']=y+h-1+threshold
            mask = np.full((binary.shape[0], binary.shape[1]), 255).astype(np.uint8)
            cv2.drawContours(mask, contours, i, color=0, thickness=-1)
            d['member']=[(x,y,w,h,mask)]
            same_row.append(d)
    target_row=same_row[0]
    for row in same_row:
        if len(row['member'])>len(target_row['member']) and img.shape[0]/2>=row['min_y_top'] and img.shape[0]/2<=row['max_y_bottom']:
            target_row=row
    
    for i in range(len(target_row['member'])):
        for j in range(i+1, len(target_row['member'])):
            if target_row['member'][j][0] < target_row['member'][i][0]:
                tmp=target_row['member'][j]
                target_row['member'][j]=target_row['member'][i]
                target_row['member'][i]=tmp

    frame=binary.copy()
    frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    result=''
    for member in target_row['member']:
        x,y,w,h,mask=member
        cv2.rectangle(frame, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        char_in_binary=binary.copy()
        char_in_binary=cv2.threshold(char_in_binary,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
        char_in_binary+=mask
        char_in_binary=cv2.threshold(char_in_binary,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[-1]
        char=char_in_binary[y:y+h, x:x+w]
        result+=ocr(char)
    print(result.upper())
    cv2.imshow('test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


