
from pathlib import Path
CODEDIR = str(Path(__file__).resolve().parent.parent )+ "/src/"
import sys
sys.path.append(CODEDIR)
from pathlib import Path
import leuci_pol.interpolator as pol

from leuci_xyz import spacetransform as space
from leuci_xyz import gridmaker as grid
from leuci_xyz import vectorthree as v3
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
EG_DIR = str(Path(__file__).resolve().parent.parent )+ "/examples/results/"
sys.path.append(EG_DIR)

if True:
  print("### Return type test")
  ########## INPUTS #################
  degree = 3  
  test_methods = ["linear"]
  depth = 4
  width=10
  samples=21
  ###################################
  ## GRID ##
  gm = grid.GridMaker()
  u_coords = gm.get_unit_grid(width,samples)
  u_coords3 = gm.get_unit_grid(width,samples,depth)
    # Create the 3 coordinates for orientation  
  central = v3.VectorThree(0,0,0)  
  linear = v3.VectorThree(1,0,0.5)
  planar = v3.VectorThree(1,1,0)  
    
  spc = space.SpaceTransform(central, linear, planar)
  xyz_coords = spc.convert_coords(u_coords)
  xyz_coords3 = spc.convert_coords(u_coords3)
  ###################################

  for interp_method in test_methods:
      dt1 = datetime.datetime.now()  
      print("####### Testing", interp_method, "########")
      ## the 3d data is in a list of values, 
      ## with the fastest changing axis first, 
      ## called F, M, S (fast, medium, slow)                        
      vals = [[[1,2,3,4,5],[6,7,8,9,10]],[[11,12,13,14,15],[16,17,18,19,20]]]
      np_vals = np.array([[[1,2,3,4,5],[6,7,8,9,10]],[[11,12,13,14,15],[16,17,18,19,20]]])
      f,m,s = len(vals),len(vals[0]),len(vals[0][0])    
      intr = pol.create_interpolator(interp_method,vals,FMS=(f,m,s),log_level=1,as_sd=2)
      print("---creating values", datetime.datetime.now())
      
      dvals_val = intr.get_val_slice(xyz_coords,deriv=0,ret_type="vals")      
      dvals_np = intr.get_val_slice(xyz_coords,deriv=0,ret_type="np")
      dvals_3d = intr.get_val_slice(xyz_coords,deriv=0,ret_type="3d")

      dvals_val3 = intr.get_val_slice3d(xyz_coords3,deriv=0,ret_type="vals")
      dvals_np3 = intr.get_val_slice3d(xyz_coords3,deriv=0,ret_type="np")
      dvals_3d3 = intr.get_val_slice3d(xyz_coords3,deriv=0,ret_type="3d")

      for slice in [dvals_val,dvals_np,dvals_3d,dvals_val3,dvals_np3,dvals_3d3]:
        print(type(slice))
        
      
      #for slice in [dvals_np,dvals_val,dvals_np3,dvals_val3]:
       # print(slice[0])
      
      
      