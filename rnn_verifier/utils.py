import sys
import numpy as np
import os

def normalize(image, means, stds):
    for i in range(len(image)):
        image[i] = (image[i] - means[0])/stds[0]
    return