"""
RSA 18/3/23

This is a somewhat meta-test case, I have used this to generate the code that is actually used in the library.
This file generates fixed matrices that can be loaded in code as precalculated matrices rather than calcuated on the fly

You can calculate them on the fly if you prefer
"""

## Ensure code is importaed in path
from pathlib import Path
#CODEDIR = ""#str(Path(__file__).resolve().parent.parent )+ "/src/"
MAKE_DIR = str(Path(__file__).resolve().parent)+ "/leuci_pol/"
#import sys
#sys.path.append(CODEDIR)
#from pathlib import Path
from leuci_pol import invariantmaker as mak

#from leuci_xyz import vectorthree as v3
#from leuci_xyz import gridmaker as grid
#from leuci_xyz import spacetransform as space

########## INPUTS #################


########## EXAMPLE #################
def make():
    print("Making some matrices")
    # the 3d data is in a list of values, with the fastest changing axis first, called F, M, S (fast, medium, slow)        
    ################ 1,2,3 and 5 ######################
    mk1 = mak.InvariantMaker((4,4,4))    
    mk1.save_as_file(MAKE_DIR + "iv3.py",3)    
    mk1 = mak.InvariantMaker((2,2,2))    
    mk1.save_as_file(MAKE_DIR + "iv1.py",1)    
    mk1 = mak.InvariantMaker((6,6,6))    
    mk1.save_as_file(MAKE_DIR + "iv5.py",5)
    mk1 = mak.InvariantMaker((3,3,3))    
    mk1.save_as_file(MAKE_DIR + "iv2.py",2)
    
    ################ Some 1d matrices ######################
    #mk1 = mak.InvariantMaker((4,1,1))    
    #mk1.save_as_file(MAKE_DIR + "4_1d.py",4)
    #mk1 = mak.InvariantMaker((2,1,1))    
    #mk1.save_as_file(MAKE_DIR + "2_1d.py",2)
    
    
make()
