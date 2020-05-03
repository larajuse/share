import numpy as np
import tensorflow as tf
import hashlib

class S2(object):
    def __init__(self):
        self.__sol1 = "90bbaa0bc5b96d1ea6b4624b5ddebef169be0ed1917c270ebe182182"
        self.__sol2 = "929dd9980f7d25444f148767550b1f5396d8b5e09f1b9ffd9b067d56"
        
    def p1(self, res, verbose=True):
        val = np.round(res, decimals=2)
        code = hashlib.sha224(str(val).encode('utf-8')).hexdigest()
        if verbose:
            if code==self.__sol1:
                print("Correcto")
            else:
                print("Incorrecto")
        else:
            return code==self.__sol1
        
    def p2(self, model, verbose=True):
        my_str = str(model.layers[0].units)+str("sigmoid" in str(model.layers[0].activation))+model.loss
        code = hashlib.sha224(str(my_str).encode('utf-8')).hexdigest()
        if verbose:
            if code==self.__sol2:
                print("Correcto")
            else:
                print("Incorecto")
        else:
            return code==self.__sol2
