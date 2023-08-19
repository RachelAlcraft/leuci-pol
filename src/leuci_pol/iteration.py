"""


"""
from abc import ABC, abstractmethod
from leuci_xyz import vectorthree as v3
from leuci_xyz import matrix3d as d3
import math
import numpy as np

### Factory method for creation ##############################################################
def create_iteration(method,scoring,pointsA,pointsB,log_level=0):
    """
    Factory method to create iteration of image registration classes.

    Parameters
    ----------
    method : string
        The name of the scoring function method, least squares, abs val...
    scoring : a scoring function
        a scoring function that contains the interpolators        
    pointsA : 3 xyz points list
        the central, linear, plnar points for the reference image that we are minimising to
    pointsB : 3 xyz points list
        the initial start of the points that we are minimising to
    log_level : int=0
        How much logging you want to see

    Return
    ------
    iteration

    """
    
    if log_level > 0:
        print("Iteration:",method)
                                        
    itan = None
    if method == "straight":
        itan = Straightion(scoring,pointsA,pointsB,log_level)
    elif method == "minim":
        itan = Minimation(scoring,pointsA,pointsB,log_level)
    elif method == "simplex":
        itan = Simplexion(scoring,pointsA,pointsB,log_level)
    elif method == "marquadt":
        itan = Marquadtion(scoring,pointsA,pointsB,log_level)
    elif method == "boxes":
        itan = Boxion(scoring,pointsA,pointsB,log_level)
    else: 
        raise(Exception("Method not known " + method))
    itan.init()
    return itan

### Abstract class ############################################################################
class Iteration(ABC):
    def __init__(self, scoring,pointsA,pointsB,log_level=0):
        self.scoring = scoring
        self.pointsA = pointsA
        self.pointsB = pointsB
        self.log_level = log_level                 
        self.round = 12        
    ############################################################################################################            
    @abstractmethod    
    def is_minim(self, central, linear, planar):
        pass
    @abstractmethod    
    def get_next(self, central, linear, planar):
        pass
    ############################################################################################################
    # implemented interface that is the same for all abstractions
    def get_score(self, central,linear,planar):
        return self.scoring.get_score()
        
####################################################################################################
### Straight superposition of the given points with no minimising efforts - the difference score is irrelevant as is the "seed"
####################################################################################################
class Straightion(Iteration):                
    def init(self):
        pass
    ## implement abstract interface #########################################
    def is_minim(self, central, linear, planar):        
        return True
    def get_next(self, central, linear, planar):        
        return central, linear, planar
####################################################################################################
### This looks for the nearest minima to the given points and adjusts - the difference score is irrelevant as is the "seed"
############################################################################ ########################
class Minimation(Iteration):
    """
    In thisd method we are not looking to minimise the difference between the 2, iunstead we are looking to adjust all given points to the nearest maximum and re-align
    """              
    def init(self):
        pass
    ## implement abstract interface #########################################
    def is_minim(self, central, linear, planar):        
        return True
    def get_next(self, central, linear, planar):        
        return central, linear, planar

####################################################################################################
### Boxes algorithm assumes the minimum is nearby and draws smaller boxes - it is using the scoring function against the "seed" to minimise by adjusting the other
############################################################################ ########################
class Boxion(Iteration):                
    def init(self):
        pass
    ## implement abstract interface #########################################
    def is_minim(self, central, linear, planar):        
        return True
    def get_next(self, central, linear, planar):        
        return central, linear, planar

####################################################################################################
### METHODS I DON'T KNOW PLACE HOLDERS - it is using the scoring function against the "seed" to minimise by adjusting the other
### As far as I know it doesn;t minimise the set, only each one against the seed?
############################################################################ ########################
class Simplexion(Iteration):                
    def init(self):
        pass
    ## implement abstract interface #########################################
    def is_minim(self, central, linear, planar):        
        return True
    def get_next(self, central, linear, planar):        
        return central, linear, planar
##############################################################################
class Marquadtion(Iteration): 
    """
    Resources
    https://www.mathematik.uni-konstanz.de/en/volkwein/python/oppy/What%20is%20oppy/index_whatisoppy.html    
    http://nbn-resolving.de/urn:nbn:de:bsz:352-2-1ofyba49ud2jr5

    """               
    def init(self):
        pass
    ## implement abstract interface #########################################
    def is_minim(self, central, linear, planar):        
        return True
    def get_next(self, central, linear, planar):        
        return central, linear, planar
            
    