import numpy as np
import pdb

class MEM():

    def __init__(self, DEBUG=0):
        self.DEBUG = DEBUG
        return

    def new(self, shape, type_str):
        self.shape = shape
        if (type_str == "rand"):
            np.random.seed(0)
            self.buffer = np.random.randint(1, 10, shape)
        elif (type_str == "zero"):
            self.buffer = np.zeros(shape)
        elif (type_str == "ones"):
            self.buffer = np.ones(shape)
        return self.buffer

    def __call__(self):
        return
