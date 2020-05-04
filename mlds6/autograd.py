import numpy as np
import tensorflow as tf
import hashlib
from functools import reduce

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

class S3(object):
    def __init__(self):
        self.__and_gt = lambda i: i[0] and i[1]
        self.__or_gt = lambda i: i[0] or i[1]
        self.__sol3 = "929dd9980f7d25444f148767550b1f5396d8b5e09f1b9ffd9b067d56"
        
    def __rosenblatt(self, X, w, b):
        return np.round((tf.sign(tf.matmul(X, w) + b) + 1)/2)
    
    def p1(self, w, b, verbose=True):
        X = np.random.randint(0, 2, size=(100, 2)).astype(np.float32)
        preds = self.__rosenblatt(X, w, b).flatten()
        asserts = [preds[i]==self.__and_gt(X[i]) for i in range(X.shape[0])]
        res = reduce(lambda a, b: a and b, asserts, True)
        if verbose:
            if res:
                print("Correcto")
            else:
                print("Incorrecto")
        else:
            return res
    def p2(self, w, b, verbose=True):
        X = np.random.randint(0, 2, size=(100, 2)).astype(np.float32)
        preds = self.__rosenblatt(X, w, b).flatten()
        asserts = [preds[i]==self.__or_gt(X[i]) for i in range(X.shape[0])]
        res = reduce(lambda a, b: a and b, asserts, True)
        if verbose:
            if res:
                print("Correcto")
            else:
                print("Incorrecto")
        else:
            return res
    def p3(self, model, verbose=True):
        vals = ["Sequential" in str(model), model.input_shape[1],
                model.output_shape[1], model.layers[1].units,
                "relu" in str(model.layers[1].activation),
                "softmax" in str(model.layers[2].activation),
                "categorical_crossentropy" in str(model.loss),
                "CategoricalAccuracy" in str(model.metrics)]
        code = "".join(map(str, vals))
        if verbose:
            if code==self.__sol3:
                print("Correcto")
            else:
                print("Incorrecto")
        else:
            return code==self.__sol1
