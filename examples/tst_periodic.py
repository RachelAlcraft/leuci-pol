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
#test_methods = ["nearest","linear","cubic"]
test_methods = ["nearest","linear","cubic","bspline","mspline"]
test_methods = ["linear","cubic","bspline"]
full_test = False

########## EXAMPLE #################
def nearest_interp(interp_method):    
    # the 3d data is in a list of values, with the fastest changing axis first, called F, M, S (fast, medium, slow)
    vals = [[[1,2],
            [3,4]],
            
            [[5,6],
            [7,8]]]
    f,m,s = 2,2,2
    # simplest creation of an interpolator, add values, specify the axis, ask for a value
    intr = pol.create_interpolator(interp_method,vals,FMS=(f,m,s),log_level=1)    
    print("exact")
    print("1 =", intr.get_value(0,0,0)) 
    #print("2 =", intr.get_value(1,0,0)) 
    print("3 =", intr.get_value(0,1,0)) 
    #print("5 =", intr.get_value(0,0,1)) 
    print("8 =", intr.get_value(1,1,1)) 
    print("half ways")
    print("0.5,0,0 =", intr.get_value(0.5,0,0))#1-2
    print("0,0.5,0 =", intr.get_value(0,0.5,0)) #1-3
    print("0,0,0.5 =", intr.get_value(0,0,0.5)) #1-5
    

    
    if full_test:

        # Some testing
        print(intr.get_pos_from_fms(0,0,0),intr.get_value(0,0,0))
        print(intr.get_pos_from_fms(1,0,0),intr.get_value(1,0,0))
        print(intr.get_pos_from_fms(0,1,0),intr.get_value(0,1,0))
        print(intr.get_pos_from_fms(1,1,0),intr.get_value(1,1,0))
        print(intr.get_pos_from_fms(0,0,1),intr.get_value(0,0,1))
        print(intr.get_pos_from_fms(1,0,1),intr.get_value(1,0,1))
        print(intr.get_pos_from_fms(0,1,1),intr.get_value(0,1,1))
        print(intr.get_pos_from_fms(1,1,1),intr.get_value(1,1,1))
        
        
        print(intr.get_fms_from_pos(0))
        print(intr.get_fms_from_pos(1))
        print(intr.get_fms_from_pos(2))
        print(intr.get_fms_from_pos(3))
        print(intr.get_fms_from_pos(4))
        print(intr.get_fms_from_pos(5))
        print(intr.get_fms_from_pos(6))
        print(intr.get_fms_from_pos(7))

        print("Interp to 0.5,0.5,0.4")
        print("... =", intr.get_value(0.5,0.5,0.4))
        print("Interp to 0.5,0.5,0.5 = ")
        print("... =", intr.get_value(0.5,0.5,0.5))
        print("Interp to 0.5,0.5,0.6 = ")
        print("... =", intr.get_value(0.5,0.5,0.6))
        print("Interp to 1,1,0.9 = ")
        print("... =", intr.get_value(1,1,0.9))
        print("Interp to 1,1,1.9 = ")
        print("... =", intr.get_value(1,1,1.9))

        print("Interp to -0.5,-0.5,-0.5 = ")
        print("... =", intr.get_value(-0.5,-0.5,-0.5))
        print("... =", intr.get_value(1.5,1.5,1.5))
        print("... =", intr.get_value(1.5,1.5,1.6))
        print("... =", intr.get_value(1.6,1.6,1.6))
                        
#######################################################
for interp_method in test_methods:
    print("#### Testing", interp_method, "####")
    nearest_interp(interp_method)
