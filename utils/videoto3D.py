import cv2
import numpy as np 
from mtcnn.mtcnn import MTCNN

from utils.cutImage import getFrameI, cutImg, detectFace
# from tfoptflow.extract_optFlow import Img2Flow
# from extract_optFlow import Img2Flow

from utils.opticalflow import OpticalFlow  # Open CV >= 4
# from opticalflow import OpticalFlow

# -----------COLAB-----------------------------------
# from cutImage import getFrameI, cutImg, detectFace
# from opticalflowCv3 import OpticalFlow # using colab

#---------------------------------------------------
#from utils.opticalflowCv3 import OpticalFlow # OpenCV >= 3 and <= 4

class Videoto3D:

    def __init__(self,  width , height , depth = 10):

        self.width = width
        self.height = height
        self.depth = depth
        self.flower = OpticalFlow()
        self.detector =  MTCNN()
    
    def video3D(self, filename, color=True, skip=True, mp4 = True):
        # color True : RGB
        # color false: hardwired kenerl 
        #skip: True division give 10 frame
        
        out = getFrameI(self.detector,filename)
        #print(out)
        if out is None:
            return None
        cap = cv2.VideoCapture(filename)

        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT) # give n frame
        
        if (color == True):
            frames = [int(x * nframe / self.depth) for x in range(self.depth)]
            # print(len(frames))
        else:
            frames = [int(x * nframe / self.depth) for x in range(self.depth)]

        framearray = []
        
        for i in range(self.depth):
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, prvs = cap.read()
            
            if prvs is None:
                break
            prvs = cutImg(prvs, out)
            prvs = cv2.resize(prvs, (self.height, self.width))
            
            if color:
                
                if (mp4 != True):
                    (h, w) = prvs.shape[:2]
                    # calculate the center of the image
                    center = (w / 2, h / 2)
                    angle180 = 180
                    scale = 1.0
                    M = cv2.getRotationMatrix2D(center, angle180, scale)
                    prvs = cv2.warpAffine(prvs, M, (w, h))
                # cv2.imshow("d",prvs)
                # cv2.waitKey(0)
                
                framearray.append(prvs)
                
            else:
                if i == self.depth - 1:
                    break
                
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i + 1])
                    ret, next_frame = cap.read()
                    next_frame = cv2.resize(next_frame, (self.height, self.width))
                    
                    image1 = prvs
                    image2 = next_frame
                   
                    flow = self.flower.predict(image1, image2)
                    # flow = np.asanyarray(flow)[0]
                    framearray.append(flow)
        if len(framearray) < self.depth and len(framearray) > 15:
            
            dis = self.depth - len(framearray)
            feature = framearray[-1]
            for i in range(dis): 
                framearray.append(feature)
           
        cap.release()
        framearray = np.asanyarray(framearray)
        
        return framearray
