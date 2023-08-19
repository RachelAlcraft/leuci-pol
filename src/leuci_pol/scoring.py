"""


"""
from abc import ABC, abstractmethod
from leuci_xyz import vectorthree as v3
from leuci_xyz import matrix3d as d3
import math
import numpy as np

### Factory method for creation ##############################################################
def create_scoring(method, interpA, interpB,log_level=0):
    """
    Factory method to create scoring function classes.

    Parameters
    ----------
    method : string
        The name of the scoring function method, least squares, abs val...
    interpA : an interpolator
        The interpolator that contains the full matrix of values which is the template
    interpB : an interpolator
        The interpolator that contains the full matrix of values that we are comparing
    
    log_level : int=0
        How much logging you want to see

    Return
    ------
    scorer

    """
    
    if log_level > 0:
        print("Scoring:",method)
                                        
    scrr = None
    if method == "absval":
        scrr = AbsValueScore(interpA,interpB,log_level)
    elif method == "leastsquares":
        scrr = LeastSquaresScore(interpA,interpB,log_level)    
    else: 
        raise(Exception("Method not known " + method))
    scrr.init()
    return scrr

### Abstract class ############################################################################
class Scoring(ABC):
    def __init__(self, interpA,interpB,log_level=0):
        self.interpA = interpA
        self.interpB = interpB        
        self.log_level = log_level                 
        self.round = 12        
    ############################################################################################################            
    @abstractmethod
    def get_score(self, central, linear, planar):
        pass    
    ############################################################################################################
    # implemented interface that is the same for all abstractions
    def get_slice3d(self, central,linear,planar):
        pass
        
####################################################################################################
### Absolute Value difference scorer
####################################################################################################
class AbsValueScore(Scoring):                
    def init(self):
        pass
    ## implement abstract interface #########################################
    def get_score(self, central, linear, planar):        
        return 0
####################################################################################################
### Least Squares difference scorer
####################################################################################################
class LeastSquaresScore(Scoring):                
    def init(self):
        pass
    ## implement abstract interface #########################################
    def get_score(self, central, linear, planar):        
        return 0
            
    