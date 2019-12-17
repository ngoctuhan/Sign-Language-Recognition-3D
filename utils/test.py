# import numpy as np

# print("[INFOR] : Starting loading data form disk...........")
# from transVideotoImg3D import TrimVideo

# # color: True RGB
# # color: False OPT Flow RGB

# (X_train,y_train) = TrimVideo(nframe = 15,weight = 224, height = 224,color = True, path_Video = "F:/StreamVideo Save/data")
# # (X_test, y_test)  = TrimVideo(nframe = 15,weight = 224, height = 224,color = True, path_Video = "/content/drive/My Drive/Sign Language Recognition/data_test")
# print("[INFOR] : Loading Done !")

# from sklearn.preprocessing import LabelBinarizer
# label_binary = LabelBinarizer()
# y_train = label_binary.fit_transform(y_train)
# # y_test = label_binary.transform(y_test)
# for i in range(len(label_binary.classes_)):
#   print(i, " : " , label_binary.classes_[i])


import numpy as np 


W =  np.array([
      [-1, 3, 3, -1], 
      [1, -3,-3, 1],
      [-1 , -3, -3, -1],
      [1, -3, -3, 1],
      [3, -1, -1, 3]])

B = np.array([[-1, 1, 1, -1, -1]])

print(B.dot(W))