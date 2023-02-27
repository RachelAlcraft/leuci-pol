"""
RSA 27/2/23 nearest Neighbour interpolator

"""
from . import interpolator as pol
class Linear(pol.Interpolator):
                
    def get_value(self, x, y, z):
        return 7
    