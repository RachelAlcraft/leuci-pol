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
import leuci_map.mapsmanager as mapss
import leuci_map.mapfunctions as mfun
import leuci_xyz.vectorthree as v3

import datetime

def speed_test():

    ########## INPUTS #################
    
    #pdb_codes = ["6eex","1ejg"]
    interp_methods = ["nearest","linear","mv1","cubic","bspline"]
    #derivs = [0,1,2]
    pdb_codes = ["6eex"]    
    #interp_methods = ["bspline"]
    derivs = [0,1,2]
    ###############################
    width=8
    samples=11    
    degree = 3   
    ########## EXAMPLE #################
    for pdb_code in pdb_codes:
        t1 = datetime.datetime.now()
        #po = moad.MapLoader(pdb_code, directory=DATADIR, cif=False)
        #if not po.exists():
        #    po.download()
        #po.load()
        #po.load_values()
        print("Fetching=",pdb_code,datetime.datetime.now()-t1)
        mapman = mapss.MapsManager()
        mapman.set_dir(DATADIR)
        po = mapman.get_or_create(pdb_code,file=1,header=1,values=1)
        t1 = datetime.datetime.now()        
        print("...fetched=",datetime.datetime.now()-t1)
        t1 = datetime.datetime.now()

        my_pdb = po.pobj
        al,ac,ap = my_pdb.get_first_three()
        cc,cl,cp = my_pdb.get_coords(ac),my_pdb.get_coords(al),my_pdb.get_coords(ap)   
        central = v3.VectorThree().from_coords(cc)
        linear = v3.VectorThree().from_coords(cl)
        planar = v3.VectorThree().from_coords(cp) 

        for interp_method in interp_methods:
            t0 = datetime.datetime.now()                       
            print("=============",interp_method,pdb_code,"=================")
            mf = mfun.MapFunctions(pdb_code,po.mobj,po.pobj,interp_method,degree=degree)
            print("Interper=",interp_method,degree,datetime.datetime.now()-t1)

            for deriv in derivs:    
                t1 = datetime.datetime.now()
                if deriv == 0:
                    vals = mf.get_slice(central,linear,planar,width,samples,interp_method,deriv=0)
                    print("Density=",interp_method,datetime.datetime.now()-t1)
                elif deriv == 1:
                    vals = mf.get_slice(central,linear,planar,width,samples,interp_method,deriv=1)
                    print("Radient=",interp_method,datetime.datetime.now()-t1)
                elif deriv == 2:        
                    vals = mf.get_slice(central,linear,planar,width,samples,interp_method,deriv=2)
                    print("Laplacian=",interp_method,datetime.datetime.now()-t1)
            
            t1 = datetime.datetime.now()                
            print("Total=",interp_method,pdb_code,datetime.datetime.now()-t0)

        print("====================================================")
            
###############################################################################################################
## PROFILER ##
import cProfile
cProfile.run('speed_test()')
#speed_test()
                
                



                    


