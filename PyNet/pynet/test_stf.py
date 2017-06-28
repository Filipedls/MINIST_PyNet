import numpy as np
#import matplotlib.pyplot as plt
import cv2
from timeit import default_timer as timer
#import pyximport; pyximport.install()


class A:

	def __init__( self, var):
		self.my_var = var

	def change(self):
		self.my_var.append('changed!')



var1 = ['abc']

class_A = A(var1)

class_A.change()

print var1