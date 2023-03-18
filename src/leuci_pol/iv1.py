'''
# Automaticaly generated InvariantVandermonde matrix
# Generated from leuci-pol interpolation loibrary
# (c) Rachel Alcraft, 2023, Birkbeck College, London University
'''

import numpy as np
class InvariantVandermonde(object):
	def __init__(self):
		self.make_mat()
	def get_invariant(self):
		return self.alcraft
	def make_mat(self):
		self.alcraft = np.zeros((8,8))
		self.alcraft[0,0]=1.0
		self.alcraft[0,1]=0.0
		self.alcraft[0,2]=0.0
		self.alcraft[0,3]=0.0
		self.alcraft[0,4]=0.0
		self.alcraft[0,5]=0.0
		self.alcraft[0,6]=0.0
		self.alcraft[0,7]=0.0
		self.alcraft[1,0]=-1.0
		self.alcraft[1,1]=1.0
		self.alcraft[1,2]=0.0
		self.alcraft[1,3]=0.0
		self.alcraft[1,4]=0.0
		self.alcraft[1,5]=0.0
		self.alcraft[1,6]=0.0
		self.alcraft[1,7]=0.0
		self.alcraft[2,0]=-1.0
		self.alcraft[2,1]=0.0
		self.alcraft[2,2]=1.0
		self.alcraft[2,3]=0.0
		self.alcraft[2,4]=0.0
		self.alcraft[2,5]=0.0
		self.alcraft[2,6]=0.0
		self.alcraft[2,7]=0.0
		self.alcraft[3,0]=1.0
		self.alcraft[3,1]=-1.0
		self.alcraft[3,2]=-1.0
		self.alcraft[3,3]=1.0
		self.alcraft[3,4]=0.0
		self.alcraft[3,5]=0.0
		self.alcraft[3,6]=0.0
		self.alcraft[3,7]=0.0
		self.alcraft[4,0]=-1.0
		self.alcraft[4,1]=0.0
		self.alcraft[4,2]=0.0
		self.alcraft[4,3]=0.0
		self.alcraft[4,4]=1.0
		self.alcraft[4,5]=0.0
		self.alcraft[4,6]=0.0
		self.alcraft[4,7]=0.0
		self.alcraft[5,0]=1.0
		self.alcraft[5,1]=-1.0
		self.alcraft[5,2]=0.0
		self.alcraft[5,3]=0.0
		self.alcraft[5,4]=-1.0
		self.alcraft[5,5]=1.0
		self.alcraft[5,6]=0.0
		self.alcraft[5,7]=0.0
		self.alcraft[6,0]=1.0
		self.alcraft[6,1]=0.0
		self.alcraft[6,2]=-1.0
		self.alcraft[6,3]=0.0
		self.alcraft[6,4]=-1.0
		self.alcraft[6,5]=0.0
		self.alcraft[6,6]=1.0
		self.alcraft[6,7]=0.0
		self.alcraft[7,0]=-1.0
		self.alcraft[7,1]=1.0
		self.alcraft[7,2]=1.0
		self.alcraft[7,3]=-1.0
		self.alcraft[7,4]=1.0
		self.alcraft[7,5]=-1.0
		self.alcraft[7,6]=-1.0
		self.alcraft[7,7]=1.0
