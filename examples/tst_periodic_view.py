
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
import datetime
EG_DIR = str(Path(__file__).resolve().parent.parent )+ "/examples/results/"
sys.path.append(EG_DIR)

if True:
  print("### Visual Test ###")
  ########## INPUTS #################
  degree = 3  
  test_methods = ["nearest","linear","cubic","bspline"]
  #test_methods = ["cubic","bspline"]
  width=10
  samples=71
  ###################################
  ## GRID ##
  gm = grid.GridMaker()
  u_coords = gm.get_unit_grid(width,samples)
    # Create the 3 coordinates for orientation  
  central = v3.VectorThree(0,0,0)
  linear = v3.VectorThree(1,0,0)
  planar = v3.VectorThree(1,1,0)
  spc = space.SpaceTransform(central, linear, planar)                    
  xyz_coords = spc.convert_coords(u_coords)        
  ###################################

  for interp_method in test_methods:
      dt1 = datetime.datetime.now()  
      print("####### Testing", interp_method, "########")
      ## the 3d data is in a list of values, 
      ## with the fastest changing axis first, 
      ## called F, M, S (fast, medium, slow)
      #vals = [[[1,2],[3,4]],[[5,6],[7,8]]]
      vals = [[[1,2,3,4,5],[6,7,8,9,10]],[[11,12,13,14,15],[16,17,18,19,20]]]      
      f,m,s = len(vals),len(vals[0]),len(vals[0][0])    
      intr = pol.create_interpolator(interp_method,vals,F=f,M=m,S=s,log_level=1,degree=degree)    
      print("---creating values", datetime.datetime.now())
      dvals = intr.get_val_slice(xyz_coords,deriv=0)      
      rvals = intr.get_val_slice(xyz_coords,deriv=1)
      lvals = intr.get_val_slice(xyz_coords,deriv=2)
      
      print("---making plots", datetime.datetime.now())
      data_rads = go.Heatmap(z=rvals,colorscale=['Black','Snow'],showscale=False)      

      minl,maxl = 1000,-1000    
      for i in range(len(lvals)):
        for j in range(len(lvals[0])):        
          minl = min(lvals[i][j],minl)
          maxl = max(lvals[i][j],maxl)              
      l0 = 0.5
      if minl < 0:
        l0 = (0 - minl) / (maxl - minl)
      print(minl, maxl, l0)
      data_laps = go.Heatmap(z=lvals,showscale=False,
                             colorscale=[(0, "rgb(100,0,0)"),(l0/2, "crimson"), (l0, "silver"), (3*l0/2, "blue"),(1, "navy")],)
      
      data_vals = go.Heatmap(z=dvals,showscale=True, 
                          colorscale=[(0, "grey"), (0.1, "snow"), (0.5, "cornflowerblue"),(0.9, "crimson"),(1.0, "rgb(100,0,0)")],)

      fig = make_subplots(rows=1, cols=3,horizontal_spacing=0.05,vertical_spacing=0.05,column_widths=[0.33,0.33,0.33])

      fig.add_trace(data_vals,row=1,col=1)
      fig.add_trace(data_rads,row=1,col=2)
      fig.add_trace(data_laps,row=1,col=3)        
      fig.update_xaxes(showticklabels=False) # hide all the xticks
      fig.update_yaxes(showticklabels=False) # hide all the xticks
      fig.update_yaxes(scaleanchor="x",scaleratio=1)    
      fig.update_xaxes(scaleanchor="y",scaleratio=1)            
      print("---writing images", datetime.datetime.now())
      fig.write_image(EG_DIR + "eg04_" + interp_method + "_" + str(degree) + ".png")
      fig.write_html(EG_DIR + "eg04_" + interp_method + "_" + str(degree) + ".html")      
      print("completed in", datetime.datetime.now()-dt1)
  
  
      