import numpy as np 

def find_top(listObject, top = 3):

    listObject = np.asanyarray(listObject)
    index = np.argsort(listObject)

    return np.flip(index[-top: ])

