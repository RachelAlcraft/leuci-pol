"""
RSA 27/2/23

Test the nearest neighbor interpolator
"""

## Ensure code is importaed in path
from pathlib import Path
CODEDIR = str(Path(__file__).resolve().parent.parent )+ "/src/"
import sys
sys.path.append(CODEDIR)
from pathlib import Path
import leuci_pol.thevenaz as thev
from leuci_xyz import vectorthree as v3
import numpy as np


########## INPUTS #################
vals1d = np.array([1,1,1,1,1,1,1])
vals2d = np.array([[1,1,1],[1,1,1],[1,1,1]])
vals3d = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])

########## EXAMPLE #################
if True:    
    vals1d = np.array([1,1,1,1,1,1,1])
    # simplest creation of an interpolator, add values, specify the axis, ask for a value
    intr1d = thev.Thevenaz(vals1d,3,log_level=1)    
    print("1 =", intr1d.get_value(0)) 
    print("1 =", intr1d.get_value(1)) 
    print("1 =", intr1d.get_value(2)) 
    print("1 =", intr1d.get_value(3)) 
    print("1 =", intr1d.get_value(4))     
    print("half ways")
    print("1 =", intr1d.get_value(0.5))#1-2
    print("1 =", intr1d.get_value(1.5)) #1-3

    intr2d = thev.Thevenaz(vals2d,3,log_level=1)    
    print("1 =", intr2d.get_value(0,0)) 
    print("1 =", intr2d.get_value(1,2)) 
    print("half ways")
    print("1 =", intr2d.get_value(0.5,1.5)) 

    intr3d = thev.Thevenaz(vals3d,3,log_level=1)    
    print("1 =", intr3d.get_value(0,0,0)) 
    print("1 =", intr3d.get_value(1,2,1)) 
    print("half ways")
    print("1 =", intr3d.get_value(0.5,1.5,0.5)) 
    
    
