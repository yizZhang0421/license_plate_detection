import cv2, os
import numpy as np
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

for file in os.listdir('croped'):
    img=cv2.imread('croped/'+file)

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
    same_row={}
    contours, _ = cv2.findContours(binary , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        frame=binary.copy()
        frame=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    break