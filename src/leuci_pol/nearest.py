"""
RSA 27/2/23 nearest Neighbour interpolator

"""

from leuci_xyz import vectorthree as v3
from . import interpolator as pol
class Nearest(pol.Interpolator):
                
    def get_value(self, x, y, z):
        closest_pnt = self.closest(v3.VectorThree(x,y,z))  
        #print(closest_pnt.A, closest_pnt.B,closest_pnt.C)      
        return self.get_fms(closest_pnt.A, closest_pnt.B,closest_pnt.C)
            