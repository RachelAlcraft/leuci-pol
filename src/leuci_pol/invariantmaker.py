"""

"""

import numpy as np
from numpy.linalg import inv
import math

class InvariantMaker(object):
    def __init__(self, dimensions):                
        self.round = 12
        dimX, dimY, dimZ = dimensions[0],1,1
        if len(dimensions) > 1:
            dimY = dimensions[1]
        if len(dimensions) > 2:
            dimZ = dimensions[2]

        simul = np.zeros([dimX*dimY*dimZ,dimX*dimY*dimZ])
        ser = -1
        for i in range(0, dimX):
            for j in range(0, dimY):
                for k in range(0, dimZ):
                    ser += 1
                    sec = -1
                    for ic in range(0, dimX):
                        for jc in range(0, dimY):
                            for kc in range(0, dimZ):
                                sec += 1
                                seCoeff = math.pow(i, ic) * math.pow(j, jc) * math.pow(k, kc)                                
                                simul[ser, sec] = seCoeff

        self.alcraft = np.round(inv(simul),14)

    def save_as_file(self,filename,degree):
        with open(filename,"w") as fw:
            # first the class header stuff
            fw.write("'''\n")
            fw.write("# Automaticaly generated InvariantVandermonde matrix\n")
            fw.write("# Generated from leuci-pol interpolation loibrary\n")
            fw.write("# (c) Rachel Alcraft, 2023, Birkbeck College, London University\n")
            fw.write("'''\n\n")
            fw.write("import numpy as np\n")
            fw.write(f"class InvariantVandermonde(object):\n")
            fw.write("\tdef __init__(self):\n")            
            fw.write("\t\tself.make_mat()\n")                         
            fw.write("\tdef get_invariant(self):\n")
            fw.write("\t\treturn self.alcraft\n")
            shx,shy = self.alcraft.shape
            fw.write("\tdef make_mat(self):\n")
            fw.write(f"\t\tself.alcraft = np.zeros(({shx},{shy}))\n")            
            for i in range(shx):
                for j in range(shy):
                    v = self.alcraft[i,j]
                    #self.invariant[0, 0] = 1.0
                    fw.write(f"\t\tself.alcraft[{i},{j}]={v}\n")



    