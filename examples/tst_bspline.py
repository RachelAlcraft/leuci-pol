"""
RSA 4/2/23

This loads and examines a map file and it's corresponding cif file given the pdb code
It will automatically check what kind of electron ddensity is available - xray or cryo em
"""

## Ensure code is importaed in path
from pathlib import Path
CODEDIR = str(Path(__file__).resolve().parent.parent )+ "/src/"
import sys
DATADIR = str(Path(__file__).resolve().parent )+ "/data/"
CODEDIR = str(Path(__file__).resolve().parent.parent )+ "/src/"
sys.path.append(CODEDIR)
from pathlib import Path
import leuci_xyz.vectorthree as v3
import leuci_xyz.spacetransform as sptr

from leuci_map import maploader as moad
import leuci_map.mapfunctions as mfun
import leuci_xyz.vectorthree as v3

########## INPUTS #################
pdb_code = "6eex"
#interp_methods = ["nearest","linear","cubic","bspline"]
interp_methods = ["linear","bspline"]
########## EXAMPLE #################
def query_pdb():
    print("Showing pdb details", pdb_code)
    po = moad.MapLoader(pdb_code, directory=DATADIR, cif=False)
    if not po.exists():
        po.download()
    po.load()
    po.load_values()
    my_pdb = po.pobj
    al,ac,ap = my_pdb.get_first_three()
    print(ac,al,ap)
    keyc = my_pdb.get_key(ac)            
    coordsc = my_pdb.get_coords_key(keyc)    
    print(keyc,coordsc)    
    cvc = v3.VectorThree().from_coords(coordsc)    
    for interp_method in interp_methods:
        print("####",interp_method,"####")
        mf = mfun.MapFunctions(pdb_code,po.mobj,po.pobj,interp_method)
        print("Start xyz",cvc.get_key())
        c1_crs = mf.get_crs(cvc)
        val = mf.interper.get_value(c1_crs.A,c1_crs.B,c1_crs.C)
        print("Crs",c1_crs.get_key(),val)
        c1_xyz = mf.get_xyz(c1_crs)
        print("return xyz",c1_xyz.get_key())

        # some known values
        print("Crs (10,5,12)",mf.interper.get_value(10,5,12))
        print("Crs (10.5,5.5,12.5)",mf.interper.get_value(10.5,5.5,12.5))        
        print("Crs (29.0584,7.6586,51.7109)",mf.interper.get_value(29.0584,7.6586,51.7109))

    

            
            



                
###################################################
query_pdb()
