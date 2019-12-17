from flask import Flask, render_template, Response, request
from utils.load_tf import I3D_Model
from utils.videoto3D import Videoto3D
from utils.FindTop import find_top
from utils.cutImage import *
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
import os

app = Flask(__name__)

from utils.encodingClass import loadLabelN2W
CLASSES27 = loadLabelN2W(filename='encode29.txt')

model_dir27 = 'model29'
md27 = I3D_Model(model_dir27)

# videoto3D = Videoto3D(224, 224, 15)
detector = None

@app.route('/')
def index():
    return render_template('index_ver2.html')

FILE_OUTPUT = 'data/output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

data = []
def gen(predict =  False):

    cap = cv2.VideoCapture(0)
    if predict ==  True:
        numFrame = 0

    global data
    data = []
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break
        if(predict ==  True):
            numFrame += 1
        if predict ==  True and numFrame > 0:
            data.append(frame)
            cv2.putText(frame,"Saving frame", (105,105), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness = 2)
           
        cv2.imwrite('demo.jpg', frame)
        
        if predict ==  True and numFrame == 120:
            numFrame = 0
            predict  = False
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(predict = False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_save')
def video_feed_save():
    return Response(gen(predict = True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def data_hand(data):
    
    if len(data) > 0:
        frames = [int(x * len(data) / 15) for x in range(15)]
        tmp = []
        for i in range(5):
            out = detectFace_(data[i])
            if out is not None:
                out = out[0]['box']
                break
        for i in frames:
            img = cutImg(data[i], out)
            img = cv2.resize(img, (224,224))
            tmp.append(img)
        tmp = np.asanyarray(tmp)
        print(tmp.shape)
        return tmp 
    else:
        return None

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if request.method == 'GET':
        global data
        global Videoto3D
        tmp = data_hand(data)
        data = []
        tmp =  np.expand_dims(tmp, axis=0)
        data_res = []
        pred = md27.predict(tmp)
        rank =  find_top(pred[0], top=5)
        for i in rank:
            dataSending = str(CLASSES27[i] +' : ' + str(pred[0][i]))
            data_res.append(dataSending)
        return render_template('index_ver2.html', data = data_res)


if __name__ == '__main__':
    app.run(debug=True)