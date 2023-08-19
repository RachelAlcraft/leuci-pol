
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
  for interp_method in ["mv0"]:
      dt1 = datetime.datetime.now()  
      print("####### Testing", interp_method, "########")
      ## the 3d data is in a list of values, 
      ## with the fastest changing axis first, 
      ## called F, M, S (fast, medium, slow)      
      import numpy as np
      
      vals = [[[1,2],[3,4]],[[5,6],[7,8]]]
      #vals = [[[1,2,3,4,5],[6,7,8,9,10]],[[11,12,13,14,15],[16,17,18,19,20]]]      
      f,m,s = len(vals),len(vals[0]),len(vals[0][0])    
      intr = pol.create_interpolator(interp_method,vals,FMS=(f,m,s),log_level=2)
      print("---creating values", datetime.datetime.now())
      projXY = intr.get_projection("xy", -7,3,-2,3)
      projYZ = intr.get_projection("yz", -7,3,-2,3)
      projZX = intr.get_projection("zx", -7,3,-2,3)
      
      print("---making plots", datetime.datetime.now())
      
      data_valsXY = go.Heatmap(z=projXY,showscale=True, 
                          colorscale=[(0, "grey"), (0.1, "snow"), (0.5, "cornflowerblue"),(0.9, "crimson"),(1.0, "rgb(100,0,0)")],)
      data_valsYZ = go.Heatmap(z=projYZ,showscale=False, 
                          colorscale=[(0, "grey"), (0.1, "snow"), (0.5, "cornflowerblue"),(0.9, "crimson"),(1.0, "rgb(100,0,0)")],)
      data_valsZX = go.Heatmap(z=projZX,showscale=False, 
                          colorscale=[(0, "grey"), (0.1, "snow"), (0.5, "cornflowerblue"),(0.9, "crimson"),(1.0, "rgb(100,0,0)")],)

      fig = make_subplots(rows=1, cols=3,horizontal_spacing=0.05,vertical_spacing=0.05,column_widths=[0.33,0.33,0.33])

      fig.add_trace(data_valsXY,row=1,col=1)
      fig.add_trace(data_valsYZ,row=1,col=2)
      fig.add_trace(data_valsZX,row=1,col=3)        
      fig.update_xaxes(showticklabels=False) # hide all the xticks
      fig.update_yaxes(showticklabels=False) # hide all the xticks
      fig.update_yaxes(scaleanchor="x",scaleratio=1)    
      fig.update_xaxes(scaleanchor="y",scaleratio=1)            
      print("---writing images", datetime.datetime.now())
      fig.write_image(EG_DIR + "eg_proj.png")
      fig.write_html(EG_DIR + "eg_proj.html")      
      print("completed in", datetime.datetime.now()-dt1)
  
  
      