"""

"""

import numpy as np

class InvariantVandermonde(object):
    def __init__(self, degree):
        if degree == 5:
            self.make5()
        elif degree == 3:
            self.make3()
        else: #defau;t to linear
            self.make1()
                
    def get_invariant(self):    
        return self.invariant

    def make1(self):    
        self.invariant = np.zeros((8,8))
        self.invariant[0, 0] = 1.0            
        self.invariant[1, 0] = -1.0
        self.invariant[1, 1] = 1.0            
        self.invariant[2, 0] = -1.0            
        self.invariant[2, 2] = 1.0            
        self.invariant[3, 0] = 1.0
        self.invariant[3, 1] = -1.0
        self.invariant[3, 2] = -1.0
        self.invariant[3, 3] = 1.0            
        self.invariant[4, 0] = -1.0            
        self.invariant[4, 4] = 1.0            
        self.invariant[5, 0] = 1.0
        self.invariant[5, 1] = -1.0                        
        self.invariant[5, 4] = -1.0
        self.invariant[5, 5] = 1.0            
        self.invariant[6, 0] = 1.0            
        self.invariant[6, 2] = -1.0            
        self.invariant[6, 4] = -1.0
        self.invariant[6, 6] = 1.0            
        self.invariant[7, 0] = -1.0
        self.invariant[7, 1] = 1.0
        self.invariant[7, 2] = 1.0
        self.invariant[7, 3] = -1.0
        self.invariant[7, 4] = 1.0
        self.invariant[7, 5] = -1.0
        self.invariant[7, 6] = -1.0
        self.invariant[7, 7] = 1.0
    
    def make3(self):    
        #public double[,] inverse3 = new double[64, 64];
        pass

    def make5(self):    
        #public double[,] inverse5 = new double[216, 216];
        pass


    