"""
RSA 28/2/23

This tests a pdb file for the slice data
"""

## Ensure code is importaed in path
from pathlib import Path
CODEDIR = str(Path(__file__).resolve().parent.parent )+ "/src/"
import sys
DATADIR = str(Path(__file__).resolve().parent )+ "/data/"
CODEDIR = str(Path(__file__).resolve().parent.parent )+ "/src/"
sys.path.append(CODEDIR)
from pathlib import Path

import leuci_pol.interpolator as pol
from leuci_xyz import gridmaker as grid
from leuci_xyz import spacetransform as space
import leuci_xyz.vectorthree as v3
import leuci_map.maploader as moad

########## INPUTS #################
central = v3.VectorThree(1,2,3)
linear = v3.VectorThree(2,2,2)
planar = v3.VectorThree(3,2,3)
pdb_code = "6eex"
width=5
samples=200

########## EXAMPLE #################
def single_slice():
    print("Showing pdb map details", pdb_code)
    po = moad.MapLoader(pdb_code, directory=DATADIR, cif=False)
    if not po.exists():
        po.download()
    po.load()
    if po.em_loaded:        
        print("Loading values", pdb_code)
        po.load_values()
        if po.values_loaded:
            vals = po.mobj.values
            f = po.mobj.F
            m = po.mobj.M
            s = po.mobj.S
            print("Creating slice", pdb_code)
            
            itrp = pol.create_interpolator("nearest",vals,FMS=(f,m,s),log_level=1)    
            spc = space.SpaceTransform(central, linear, planar)
            gm = grid.GridMaker()    
            u_coords = gm.get_unit_grid(width,samples)        
            xyz_coords = spc.convert_coords(u_coords)
            sl_vals = itrp.get_val_slice(xyz_coords)
            print(sl_vals)
                                                                                        
###################################################
single_slice()
