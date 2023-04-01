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
from leuci_xyz import gridmaker as grid
from leuci_xyz import spacetransform as space




########## INPUTS #################
vals = [[[1,2],[3,4]],[[5,6],[7,8]]]
f,m,s = 2,2,2    
central = v3.VectorThree(1,1,1)
linear = v3.VectorThree(1,1,0)
planar = v3.VectorThree(0,0,0)
width=2
samples=5
method="nearest"

########## EXAMPLE #################
def nearest_slice():
    print("Checking simple nearest slice")
    # the 3d data is in a list of values, with the fastest changing axis first, called F, M, S (fast, medium, slow)        
    itrp = pol.create_interpolator(method,vals,F=f,M=m,S=s,log_level=1)                
    valsxy = itrp.get_projection("xy")
    print("xy", valsxy)
    valsyz = itrp.get_projection("yz")
    print("yz", valsyz)
    valszx = itrp.get_projection("zx")            
    print("zx", valszx)
    
    
                        
nearest_slice()
