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
import leuci_pol.interpolator as pol
from leuci_xyz import vectorthree as v3


########## INPUTS #################
degree = 3
#test_methods = ["nearest","linear","cubic","bspline"]
test_methods = ["nearest","mv0","mv1","linear","mv3","bspline"]
test_methods = ["bspline"]
###################################
for interp_method in test_methods:
    print("############# Testing", interp_method, "########################")    
    # the 3d data is in a list of values, with the fastest changing axis first, called F, M, S (fast, medium, slow)
    vals = [[[1,2],[3,4]],[[5,6],[7,8]]]
    #vals = [[[2,1],[1,3]],[[2,2],[2,2]]]
    
    #vals = [[[1,2,3,4,5],[6,7,8,9,10]],[[11,12,13,14,15],[16,17,18,19,20]]]
    
    import numpy as np
    vals = np.copy(vals).astype(float)
    print(vals)
    
    f,m,s = len(vals),len(vals[0]),len(vals[0][0])
    # simplest creation of an interpolator, add values, specify the axis, ask for a value
    intr = pol.create_interpolator(interp_method,vals,F=f,M=m,S=s,log_level=1)    
    all_once = True
    if all_once:
        pnts = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1),(0.5,0.5,0.5),(-0.5,-0.5,-0.5),(-1,-1,-1),(2,2,2)]
        vals = intr.get_values(pnts)
        print(vals)
        print("1 =",vals[0]) 
        print("2 =",vals[1]) 
        print("3 =",vals[2]) 
        print("4 =",vals[3]) 
        print("5 =",vals[4]) 
        print("6 =",vals[5]) 
        print("7 =",vals[6]) 
        print("8 =",vals[7]) 
        print("0.5 =",vals[8])   
        print("-0.5 =",vals[9])   
        print("8(-1) =",vals[10]) 
        print("1(2) =",vals[11]) 
    else:    
        print("1 =", intr.get_value(0,0,0)) 
        print("2 =", intr.get_value(0,0,1)) 
        print("3 =", intr.get_value(0,1,0)) 
        print("4 =", intr.get_value(0,1,1)) 
        print("5 =", intr.get_value(1,0,0)) 
        print("6 =", intr.get_value(1,0,1)) 
        print("7 =", intr.get_value(1,1,0)) 
        print("8 =", intr.get_value(1,1,1)) 

        print("0.5 =", intr.get_value(0.5,0.5,0.5)) 
        print("-0.5 =", intr.get_value(-0.5,-0.5,-0.5)) 
        print("8(-1) =", intr.get_value(-1,-1,-1)) 
        print("1(2) =", intr.get_value(2,2,2)) 


    #if interp_method == "bspline":
    #    print(intr._coeffs)
                        
                        
