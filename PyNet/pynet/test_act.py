from net import *
from helpers import *
import numpy as np
import cv2
import math
import random

from activation import *


act = Activation('lerelu')


var = np.array([[-0.1, 0.1], [0.1, -0.1]])

print var, "ACT: ",act.activate(var)

var = np.array([[1, 1], [1, 1]])

print var, "DIF: ",act.diff(var)

