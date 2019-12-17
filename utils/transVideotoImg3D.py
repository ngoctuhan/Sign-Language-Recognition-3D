import os
import numpy as np
from videoto3D import Videoto3D


path_ClassName = "UCF101_Action_detection_splits"


def TrimVideo(nframe =  20,weight = 128, height = 128,color = True, nclass = 100, path_Video = "" ):

    X = []
    y = []
    count = 0
    #path_Video = "/content/drive/My Drive/data_khiemthinhMP4/data_cutted"
    # path_Video = "/content/drive/My Drive/data-15class"
    # path_Video = os.path.join(path_Video, 'UCF-101')
    videoto3D = Videoto3D(weight,height, nframe)
    for folder in os.listdir(path_Video):
        
        count = count + 1
        print("[INFOR] : Loading.....", folder)
        path_folder = os.path.join(path_Video, folder)

        for filename in os.listdir(path_folder):
            
            if filename.split('.')[1] == 'mov':
                pass
            path_filename = os.path.join(path_folder, filename)

            if color:            
                tmp = videoto3D.video3D(path_filename, True)
                if tmp is not None and tmp.shape == (nframe, weight, height, 3):
                    X.append(tmp)
                    y.append(folder)
            else:
                tmp = videoto3D.video3D(path_filename, False)
                X.append(tmp)
                y.append(folder)
        if count == nclass:
            break
    X = np.asanyarray(X)
    print('[INFOR] : Shape data for train: ', X.shape)
    return (X, y)



