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
vals3d = np.array([[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]])
vals3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    
########## EXAMPLE #################
if True:        
    intr3d = thev.Thevenaz(vals3d,3,log_level=1)    
                
    print("1 =", intr3d.get_value(0,0,0)) 
    print("2 =", intr3d.get_value(1,0,0)) 
    print("3 =", intr3d.get_value(0,1,0)) 
    print("4 =", intr3d.get_value(1,1,0)) 
    print("5 =", intr3d.get_value(0,0,1)) 
    print("6 =", intr3d.get_value(0,0,1)) 
    print("7 =", intr3d.get_value(0,1,1)) 
    print("8 =", intr3d.get_value(0,1,1)) 

    print("-1 =", intr3d.get_value(-1,-1,-1))     
    
        
    print("Values=\n", intr3d.values) 
    print("Coeffs=\n", intr3d.coeffs) 
    
    
