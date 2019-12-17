import cv2 
import numpy as np 
import os 
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

# ... code

def change_brightness(img,  beta = 30):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    anlpha =  1.0
    img_new = img + beta
    img_new[img_new > 255] = 255
    img_new[img_new < 0] = 0
    return img_new


def detectFace(detector, img):
    output = detector.detect_faces(img)
    # print(len(output))
    if len(output) < 1 :
        return None
    return output

def detectFace_(img):
    detector = MTCNN()
    output = detector.detect_faces(img)
    # print(len(output))
    if len(output) < 1 :
        return None
    del detector
    return output

def cutImg(img, out):
    x1, y1, x2, y2 = out[0], out[1], out[0] + out[2], out[1] + out[3]
    img = img[60:420 , :x1-5, :]
    return img
def getAFrame(path):
    cap = cv2.VideoCapture(path)
    n_frames =  cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, frame = cap.read()
    cap.release()
    return frame
def getFrameM(detector,path):

    #print(path)
    cap = cv2.VideoCapture(path)
    n_frames =  cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(n_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        out = detectFace(detector, frame)
        if out is not None:
            out = out[0]['box']
            # cutImg(frame,out)
            cap.release()
            return out
    cap.release()
    return None
def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def getFrameI(detector, path):
    out = getFrameM(detector, path)
    if  out is not None:
        return out
    cap = cv2.VideoCapture(path)
    n_frames =  cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(n_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        frame = increase_brightness(frame)
        out = detectFace(detector, frame)
        if out is not None:
            out = out[0]['box']
            # cutImg(frame,out)
            cap.release()
            return out
    cap.release()
    return None




# d = MTCNN()
# path = 'F:/Sign Laguage 15 classes/hoa_a_0.avi'

# frame = getAFrame(path)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)

# img = increase_brightness(frame)
# cv2.imshow('frame', img)
# cv2.waitKey(0)





