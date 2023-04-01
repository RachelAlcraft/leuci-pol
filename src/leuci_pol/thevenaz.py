
"""
****** Thevenaz Spline Convolution Implementation ****************************************
Thevenaz, Philippe, Thierry Blu, and Michael Unser. ?Image Interpolation and Resampling?, n.d., 39.
http://bigwww.epfl.ch/thevenaz/interpolation/
*******************************************************************************
"""

from leuci_xyz import vectorthree as v3
import math
import numpy as np

#from . import invariant as ivm
from . import iv1
from . import iv2
from . import iv3
from . import iv5

import numpy as np

class Thevenaz(object):
    def __init__(self,values, degree,log_level=0):
        self.shape = values.shape
        self.dims = len(self.shape)
        self.degree = degree
        self.log_level = log_level         
        self.values = values.astype(float)
        if log_level > 0:
            print("Thevenaz Interpolator:",self.degree,self.shape,self.dims)    
        self._buffer = 2        
        self.h = 0.0001 #this is the interval for numerical differentiation       
        self.TOLERANCE = 2.2204460492503131e-016 # smallest such that 1.0+DBL_EPSILON != 1.0                                
        self.make_mirror_coeffs()         
    ############################################################################################################    
    def get_radient_numerical(self, x, y, z):                        
        val = self.get_value(x, y, z)
        dx = (self.get_value(x + self.h, y, z) - val) / self.h
        dy = (self.get_value(x, y + self.h, z) - val) / self.h
        dz = (self.get_value(x, y, z + self.h) - val) / self.h
        radient = (abs(dx) + abs(dy) + abs(dz)) / 3
        return radient
    ############################################################################################################    
    def get_laplacian_numerical(self, x, y, z):        
        val = self.get_value(x, y, z)
        xx = self.getDxDx_numerical(x, y, z, val)
        yy = self.getDyDy_numerical(x, y, z, val)
        zz = self.getDzDz_numerical(x, y, z, val)
        return xx + yy + zz 
    ############################################################################################################    
    def getDxDx_numerical(self, x, y, z, val):        
        va = self.get_value(x - self.h, y, z)
        vb = self.get_value(x + self.h, y, z)
        dd = (va + vb - 2 * val) / (self.h * self.h)
        return dd
    ############################################################################################################    
    def getDyDy_numerical(self, x, y, z, val):        
        va = self.get_value(x, y - self.h, z)
        vb = self.get_value(x, y + self.h, z)
        dd = (va + vb - 2 * val) / (self.h * self.h)
        return dd
    ############################################################################################################    
    def getDzDz_numerical(self, x, y, z, val):        
        va = self.get_value(x, y, z - self.h)
        vb = self.get_value(x, y, z + self.h)
        dd = (va + vb - 2 * val) / (self.h * self.h)
        return dd
    ############################################################################################################    
    def get_periodic_value(self,coords):
        new_coords = np.copy(coords)
        for c in len(coords):
            coord = coords[c]
            limit = self.shape[c]
            while coord < 0:
                coord = limit + coord
            while coord >= limit:
                coord = coord - limit
            new_coords[c] = coord        
        return self._npy[new_coords]
    ############################################################################################################    
    def get_exact_value(self,coords):        
        return self._npy[coords]
    ############################################################################################################                                    
    def get_radient(self, x, y, z):
        return self.get_radient_numerical(x,y,z)
    ############################################################################################################                                    
    def get_laplacian(self, x, y, z):
        return self.get_laplacian_numerical(x,y,z)
    ############################################################################################################                                            
    def make_mirror_coeffs(self):
        self.coeffs = np.copy(self.values)
        if len(self.shape) == 3:
            self.make_mirror_coeffs_3d(self.coeffs)
        elif len(self.shape) == 2:
            self.make_mirror_coeffs_2d(self.coeffs)
        elif len(self.shape) == 1:
            self.make_mirror_coeffs_1d(self.coeffs)

    ############################################################################################################                                            
    def make_mirror_coeffs_3d(self,values):
        shp = values.shape
        new_shp = (shp[0],shp[1])
        for i in range(values.shape[2]):
            vals = values[:,:,i:i+1].reshape(new_shp)
            self.make_mirror_coeffs_2d(vals)
    ############################################################################################################                                            
    def make_mirror_coeffs_2d(self,values):
        shp = values.shape
        new_shp = (shp[0])
        for i in range(values.shape[1]):
            vals = values[:,i:i+1].reshape(new_shp)
            self.make_mirror_coeffs_1d(vals)
    ############################################################################################################                                            
    def make_mirror_coeffs_1d(self,values):        
        pole = self.get_pole(self.degree)
        num_poles = len(pole)
        self.convert_to_interp_coeffs(pole, num_poles, len(values), values)
    ############################################################################################################                                                                                                                                                        
    def get_pole(self,degree):        
        #Recover the poles from a lookup table #currently only 3 degree, will I want to calculate all the possibilities at the beginnning, 3,5,7,9?
        pole = []
        if (degree == 9):        
            pole.append(-0.60799738916862577900772082395428976943963471853991)
            pole.append(-0.20175052019315323879606468505597043468089886575747)
            pole.append(-0.043222608540481752133321142979429688265852380231497)
            pole.append(-0.0021213069031808184203048965578486234220548560988624)
        elif (degree == 7):        
            pole.append(-0.53528043079643816554240378168164607183392315234269)
            pole.append(-0.12255461519232669051527226435935734360548654942730)
            pole.append(-0.0091486948096082769285930216516478534156925639545994)
        elif (degree == 5):        
            pole.append(np.sqrt(135.0 / 2.0 - np.sqrt(17745.0 / 4.0)) + np.sqrt(105.0 / 4.0) - 13.0 / 2.0)
            pole.append(np.sqrt(135.0 / 2.0 + np.sqrt(17745.0 / 4.0)) - np.sqrt(105.0 / 4.0) - 13.0 / 2.0)
        else:#then it is 3        
            pole.append(np.sqrt(3.0) - 2.0)
        return pole
    ############################################################################################################                                                
    def convert_to_interp_coeffs(self,pole, num_poles, width, row,mirror=True):        
        #/* special case required by mirror boundaries */
        if (width == 1):         
            #mirror filter and periodic filter, not much can be done if it is only 1 think
            return row        
        lmbda = 1
        n = 0
        k = 0
        #Compute the overall gain
        for k in range(num_poles):
            lmbda = lmbda * (1 - pole[k]) * (1 - 1 / pole[k])        
        #Apply the gain
        for n in range(width):
            row[n] = float(row[n]) * float(lmbda)        
        #loop over the poles            
        for k in range(num_poles):        
            #/* causal initialization */                        
            row[0] = self.initial_causal_coeffs(row, width, pole[k])            
            #/* causal recursion */
            for n in range(1,width):            
                row[n] = row[n]+ pole[k] * row[n - 1]
            #/* anticausal initialization */            
            row[width - 1] = self.initial_anticausal_coeffs(row, width, pole[k])            
            #/* anticausal recursion */
            for n in range(width - 2, -1, -1):
                row[n] = pole[k] * (row[n + 1] - row[n])
        return row
    ############################################################################################################                                                
    def initial_causal_coeffs(self, vals, length, pole):
        #/* begin InitialCausalCoefficient */
        Sum, zn, z2n, iz = 0.,0.,0.,0.
        n, Horizon = 0,0
        #/* this initialization corresponds to mirror boundaries */
        Horizon = length
        if (self.TOLERANCE > 0.0):        
            Horizon = np.ceil(np.log(self.TOLERANCE) / np.log(abs(pole)))
        if (Horizon < length):        
            #/* accelerated loop */
            zn = pole
            Sum = vals[0]
            for n in range(1,int(Horizon)):
                Sum += zn * vals[n]
                zn *= pole       
            return Sum        
        else:#// if (_mirror)
            #// RSA notes this is the mirror condition, when the horizon - how far you need to look ahead for good data - is not as far the data you have
            #/* full loop */
            zn = pole
            iz = 1.0 / pole
            z2n = math.pow(pole, length - 1)
            Sum = vals[0] + z2n * vals[length - 1]
            z2n *= z2n * iz 
            for n in range(1,length - 1):
                Sum += (zn + z2n) * vals[n]
                zn *= pole                
                z2n *= iz
            return (Sum / (1.0 - zn * zn))        
    ############################################################################################################                                                        
    def initial_anticausal_coeffs(self, vals, length, pole):        
        #/* this initialization corresponds to mirror boundaries */
        if (length < 2):
            return 0;
        else:#// if (_mirror)
            return ((pole / (pole * pole - 1.0)) * (pole * vals[length - 2] + vals[length - 1]))
    ############################################################################################################                                            
    def get_coeff(self,xyz):
        try:                        
            value = self.coeffs[xyz]
        except Exception as e:
            print(xyz,str(e))

        return value
    ############################################################################################################                                                                                
    def put_coeff(self,xyz,v):
        self.coeffs[xyz] = v
    ############################################################################################################                                            
    def get_value(self,x,y=None,z=None):
        if len(self.shape) == 1:
            return self.get_value1d(x)
        elif len(self.shape) == 2:
            return self.get_value2d(x,y)
        elif len(self.shape) == 3:
            return self.get_value3d(x,y,z)
    ############################################################################################################                                            
    def get_value1d(self, x):
        weight_length = self.degree + 1
        xIndex = []        
        xWeight = []        
        for s in range(weight_length):            
            xIndex.append(0)                        
            xWeight.append(0)
        i = int(np.floor(x) - np.floor(self.degree / 2))
        for l in range(self.degree+1):                          
            xIndex[l] = i #if 71.1 passed in, for linear, we would want 71 and 72,             
            i=i+1
        if (self.degree == 9):        
            xWeight = self.applyValue9(x, xIndex, weight_length)            
        elif (self.degree == 7):        
            xWeight = self.applyValue7(x, xIndex, weight_length)            
        elif (self.degree == 5):        
            xWeight = self.applyValue5(x, xIndex, weight_length)            
        else:        
            xWeight = self.applyValue3(x, xIndex, weight_length)        
        
        # !!! I have removed the mirror boundary bit, might need to put it back in RSA TODO                                  
        Width2 = 2 * self.shape[0] - 2        
        mirror_convert = [(xIndex,Width2,self.shape[0])]
        for pIndex,pLen2,pLen in mirror_convert:            
            for k in range(len(pIndex)):        
                if pLen == 1:
                    pIndex[k] = 0
                elif pIndex[k] < 0:
                    pIndex[k] = int(-pIndex[k] - pLen2 * ((-pIndex[k]) / pLen2))                
                elif pLen <= pIndex[k]:
                    pIndex[k] = int(pLen2 - pIndex[k])
        w1 = 0.0        
        for i in range(self.degree+1):                
            w1 += xWeight[i] * self.get_coeff(xIndex[i])        
        return w1
    
    def get_value2d(self, x,y):
        weight_length = self.degree + 1
        xIndex,yIndex = [],[]
        xWeight,yWeight = [],[]
        for s in range(weight_length):            
            xIndex.append(0)                        
            xWeight.append(0)
            yIndex.append(0)                        
            yWeight.append(0)
        i = int(np.floor(x) - np.floor(self.degree / 2))
        j = int(np.floor(y) - np.floor(self.degree / 2))
        for l in range(self.degree+1):                          
            xIndex[l] = i #if 71.1 passed in, for linear, we would want 71 and 72,             
            yIndex[l] = j 
            i,j=i+1,j+1
        if (self.degree == 9):        
            xWeight = self.applyValue9(x, xIndex, weight_length)            
            yWeight = self.applyValue9(y, yIndex, weight_length)            
        elif (self.degree == 7):        
            xWeight = self.applyValue7(x, xIndex, weight_length)            
            yWeight = self.applyValue7(y, yIndex, weight_length)            
        elif (self.degree == 5):        
            xWeight = self.applyValue5(x, xIndex, weight_length)            
            yWeight = self.applyValue5(y, yIndex, weight_length)            
        else:        
            xWeight = self.applyValue3(x, xIndex, weight_length)        
            yWeight = self.applyValue3(y, yIndex, weight_length)            
        
        # !!! I have removed the mirror boundary bit, might need to put it back in RSA TODO                          
        Width2 = 2 * self.shape[0] - 2
        Height2 = 2 * self.shape[1] - 2        
        mirror_convert = [(xIndex,Width2,self.shape[0]),(yIndex,Height2,self.shape[1])]
        for pIndex,pLen2,pLen in mirror_convert:
            for k in range(len(pIndex)):        
                if pLen == 1:
                    pIndex[k] = 0
                elif pIndex[k] < 0:
                    pIndex[k] = int(-pIndex[k] - pLen2 * ((-pIndex[k]) / pLen2))                
                elif pLen <= pIndex[k]:
                    pIndex[k] = int(pLen2 - pIndex[k])
        
        w2,w1 = 0.0,0.0
        for j in range(self.degree+1):
            for i in range(self.degree+1):                
                w1 += xWeight[i] * self.get_coeff((xIndex[i],yIndex[j]))
            w2 += yWeight[j] * w1
        return w2
            

    def get_value3d(self, x,y,z):                
        weight_length = self.degree + 1
        xIndex, yIndex, zIndex = [],[],[]        
        xWeight, yWeight,zWeight = [],[],[]        
        for s in range(weight_length):            
            xIndex.append(0)
            yIndex.append(0)
            zIndex.append(0)
            xWeight.append(0)
            yWeight.append(0)
            zWeight.append(0)            
        #Compute the interpolation indices
        i = int(np.floor(x) - np.floor(self.degree / 2))
        j = int(np.floor(y) - np.floor(self.degree / 2))
        k = int(np.floor(z) - np.floor(self.degree / 2))

        for l in range(self.degree+1):                          
            xIndex[l] = i #if 71.1 passed in, for linear, we would want 71 and 72, 
            yIndex[l] = j
            zIndex[l] = k
            i,j,k=i+1,j+1,k+1              
        #/* compute the interpolation weights */
        if (self.degree == 9):        
            xWeight = self.applyValue9(x, xIndex, weight_length)
            yWeight = self.applyValue9(y, yIndex, weight_length)
            zWeight = self.applyValue9(z, zIndex, weight_length)        
        elif (self.degree == 7):        
            xWeight = self.applyValue7(x, xIndex, weight_length)
            yWeight = self.applyValue7(y, yIndex, weight_length)
            zWeight = self.applyValue7(z, zIndex, weight_length)        
        elif (self.degree == 5):        
            xWeight = self.applyValue5(x, xIndex, weight_length)
            yWeight = self.applyValue5(y, yIndex, weight_length)
            zWeight = self.applyValue5(z, zIndex, weight_length)        
        else:        
            xWeight = self.applyValue3(x, xIndex, weight_length)
            yWeight = self.applyValue3(y, yIndex, weight_length)
            zWeight = self.applyValue3(z, zIndex, weight_length)        
        #//applying the mirror boundary condition becaue I am only interpolating within values??
        #// RSA edit actually I want to wrap        
        # !!! I have removed the mirror boundary bit, might need to put it back in RSA TODO                          
        Width2 = 2 * self.shape[0] - 2
        Height2 = 2 * self.shape[1] - 2
        Depth2 = 2 * self.shape[2] - 2                        
        mirror_convert = [(xIndex,Width2,self.shape[0]),(yIndex,Height2,self.shape[1]),(zIndex,Depth2,self.shape[2])]        
        for pIndex,pLen2,pLen in mirror_convert:
            for k in range(len(pIndex)):        
                if pLen == 1:
                    pIndex[k] = 0
                elif pIndex[k] < 0:
                    pIndex[k] = int(-pIndex[k] - pLen2 * ((-pIndex[k]) / pLen2))                
                elif pLen <= pIndex[k]:
                    pIndex[k] = int(pLen2 - pIndex[k])
        #Perform interolation            
        spline_degree = self.degree
        w3 = 0.0
        for k in range(spline_degree+1):
            w2 = 0.0
            for j in range(spline_degree+1):
                w1 = 0.0
                for i in range(spline_degree+1):                
                    w1 += xWeight[i] * self.get_coeff((xIndex[i],yIndex[j],zIndex[k]))
                w2 += yWeight[j] * w1
            w3 += zWeight[k] * w2
        return w3
    ############################################################################################################                                            
    def applyValue3(self,val, idc, weight_length):        
        ws = []
        for i in range(weight_length):
            ws.append(0)
        w = val - idc[1]        
        ws[3] = (1.0 / 6.0) * w * w * w
        ws[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - ws[3]        
        ws[2] = w + ws[0] - 2.0 * ws[3]
        ws[1] = 1.0 - ws[0] - ws[2] - ws[3]
        
        return ws
    ############################################################################################################                                                
    def applyValue5(self,val,idc,weight_length):        
        ws = []
        for i in range(weight_length):
            ws.append(0)
        w = val - idc[2]
        w2 = w * w
        ws[5] = (1.0 / 120.0) * w * w2 * w2
        w2 -= w
        w4 = w2 * w2
        w -= 1.0 / 2.0
        t = w2 * (w2 - 3.0)
        ws[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - ws[5]
        t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0)
        t1 = (-1.0 / 12.0) * w * (t + 4.0)
        ws[2] = t0 + t1
        ws[3] = t0 - t1
        t0 = (1.0 / 16.0) * (9.0 / 5.0 - t)
        t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0)
        ws[1] = t0 + t1
        ws[4] = t0 - t1
        return ws
    ############################################################################################################                                                
    def applyValue7(self,val,idc, weight_length):        
        ws = []
        for i in range(weight_length):
            ws.append(0)
        w = val - idc[3]
        ws[0] = 1.0 - w
        ws[0] *= ws[0]
        ws[0] *= ws[0] * ws[0]
        ws[0] *= (1.0 - w) / 5040.0
        w2 = w * w
        ws[1] = (120.0 / 7.0 + w * (-56.0 + w * (72.0 + w * (-40.0 + w2 * (12.0 + w * (-6.0 + w)))))) / 720.0
        ws[2] = (397.0 / 7.0 - w * (245.0 / 3.0 + w * (-15.0 + w * (-95.0 / 3.0 + w * (15.0 + w * (5.0 + w * (-5.0 + w))))))) / 240.0
        ws[3] = (2416.0 / 35.0 + w2 * (-48.0 + w2 * (16.0 + w2 * (-4.0 + w)))) / 144.0
        ws[4] = (1191.0 / 35.0 - w * (-49.0 + w * (-9.0 + w * (19.0 + w * (-3.0 + w) * (-3.0 + w2))))) / 144.0
        ws[5] = (40.0 / 7.0 + w * (56.0 / 3.0 + w * (24.0 + w * (40.0 / 3.0 + w2 * (-4.0 + w * (-2.0 + w)))))) / 240.0
        ws[7] = w2
        ws[7] *= ws[7] * ws[7]
        ws[7] *= w / 5040.0
        ws[6] = 1.0 - ws[0] - ws[1] - ws[2] - ws[3] - ws[4] - ws[5] - ws[7]
        return ws
    ############################################################################################################                                                
    def applyValue9(self,val,idc,weight_length):
        ws = []
        for i in range(weight_length):
            ws.append(0)
        w = val - idc[4]
        ws[0] = 1.0 - w
        ws[0] *= ws[0]
        ws[0] *= ws[0]
        ws[0] *= ws[0] * (1.0 - w) / 362880.0
        ws[1] = (502.0 / 9.0 + w * (-246.0 + w * (472.0 + w * (-504.0 + w * (308.0 + w * (-84.0 + w * (-56.0 / 3.0 + w * (24.0 + w * (-8.0 + w))))))))) / 40320.0
        ws[2] = (3652.0 / 9.0 - w * (2023.0 / 2.0 + w * (-952.0 + w * (938.0 / 3.0 + w * (112.0 + w * (-119.0 + w * (56.0 / 3.0 + w * (14.0 + w * (-7.0 + w))))))))) / 10080.0
        ws[3] = (44117.0 / 42.0 + w * (-2427.0 / 2.0 + w * (66.0 + w * (434.0 + w * (-129.0 + w * (-69.0 + w * (34.0 + w * (6.0 + w * (-6.0 + w))))))))) / 4320.0
        w2 = w * w
        ws[4] = (78095.0 / 63.0 - w2 * (700.0 + w2 * (-190.0 + w2 * (100.0 / 3.0 + w2 * (-5.0 + w))))) / 2880.0
        ws[5] = (44117.0 / 63.0 + w * (809.0 + w * (44.0 + w * (-868.0 / 3.0 + w * (-86.0 + w * (46.0 + w * (68.0 / 3.0 + w * (-4.0 + w * (-4.0 + w))))))))) / 2880.0
        ws[6] = (3652.0 / 21.0 - w * (-867.0 / 2.0 + w * (-408.0 + w * (-134.0 + w * (48.0 + w * (51.0 + w * (-4.0 + w) * (-1.0 + w) * (2.0 + w))))))) / 4320.0
        ws[7] = (251.0 / 18.0 + w * (123.0 / 2.0 + w * (118.0 + w * (126.0 + w * (77.0 + w * (21.0 + w * (-14.0 / 3.0 + w * (-6.0 + w * (-2.0 + w))))))))) / 10080.0
        ws[9] = w2 * w2
        ws[9] *= ws[9] * w / 362880.0
        ws[8] = 1.0 - ws[0] - ws[1] - ws[2] - ws[3] - ws[4] - ws[5] - ws[6] - ws[7] - ws[9]
        return ws
    ############################################################################################################                                            
    ############################################################################################################                                            
    ############################################################################################################                                                

            
        
        
        
        
        
        
        
        
    """    

        
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public class BetaSpline : Interpolator
            
        public BetaSpline(Single[] bytes, int start, int length, int x, int y, int z, int degree, int copies,bool mirror, bool sample) : base(bytes, start, length, x, y, z,copies)
        {
            TOLERANCE = 2.2204460492503131e-016; // smallest such that 1.0+DBL_EPSILON != 1.0
            _degree = degree;
            _mirror = mirror;
            
            if (_mirror)
            {
                _coefficients = getCubeWhole();
                createCoefficients();
            }
            else if (_sample)
            {
                createSampledCoefficients();
            }
            else
            {
                createPeriodicCoefficients();
            }
        }        
        public override double getValue(double xx, double yy, double zz)
        {            
            if (!isValid(xx, yy, zz))
                return 0;
            double[] xyz = adjustReflection(xx, yy, zz);
            double x = xyz[0];
            double y = xyz[1];
            double z = xyz[2];
                        
            int weight_length = _degree + 1;
            List<int> xIndex = new List<int>();
            List<int> yIndex = new List<int>();
            List<int> zIndex = new List<int>();
            List<double> xWeight = new List<double>();
            List<double> yWeight = new List<double>();
            List<double> zWeight = new List<double>();
            for (int s = 0; s < weight_length; ++s)
            {
                xIndex.Add(0);
                yIndex.Add(0);
                zIndex.Add(0);
                xWeight.Add(0);
                yWeight.Add(0);
                zWeight.Add(0);
            }


            //Compute the interpolation indices
            int i = Convert.ToInt32(Math.Floor(x) - _degree / 2);
            int j = Convert.ToInt32(Math.Floor(y) - _degree / 2);
            int k = Convert.ToInt32(Math.Floor(z) - _degree / 2);

            for (int l = 0; l <= _degree; ++l)
            {
                xIndex[l] = i++; //if 71.1 passed in, for linear, we would want 71 and 72, 
                yIndex[l] = j++;
                zIndex[l] = k++;
            }

            /* compute the interpolation weights */

            if (_degree == 9)
            {
                xWeight = applyValue9(x, xIndex, weight_length);
                yWeight = applyValue9(y, yIndex, weight_length);
                zWeight = applyValue9(z, zIndex, weight_length);
            }
            else if (_degree == 7)
            {
                xWeight = applyValue7(x, xIndex, weight_length);
                yWeight = applyValue7(y, yIndex, weight_length);
                zWeight = applyValue7(z, zIndex, weight_length);
            }
            else if (_degree == 5)
            {
                xWeight = applyValue5(x, xIndex, weight_length);
                yWeight = applyValue5(y, yIndex, weight_length);
                zWeight = applyValue5(z, zIndex, weight_length);
            }
            else
            {
                xWeight = applyValue3(x, xIndex, weight_length);
                yWeight = applyValue3(y, yIndex, weight_length);
                zWeight = applyValue3(z, zIndex, weight_length);
            }

            //applying the mirror boundary condition becaue I am only interpolating within values??
            // RSA edit actually I want to wrap
            int Width2 = 2 * XLen - 2;
            int Height2 = 2 * YLen - 2;
            int Depth2 = 2 * ZLen - 2;
            if (_mirror)
            {
                for (k = 0; k <= _degree; k++)
                {
                    xIndex[k] = (XLen == 1) ? (0) :
                        ((xIndex[k] < 0) ?
                            (-xIndex[k] - Width2 * ((-xIndex[k]) / Width2)) :
                            (xIndex[k] - Width2 * (xIndex[k] / Width2)));
                    if (XLen <= xIndex[k])
                    {
                        xIndex[k] = Width2 - xIndex[k];                    
                    }

                    yIndex[k] = (YLen == 1) ? (0) :
                        ((yIndex[k] < 0) ?
                            (-yIndex[k] - Height2 * ((-yIndex[k]) / Height2)) :
                            (yIndex[k] - Height2 * (yIndex[k] / Height2)));
                    if (YLen <= yIndex[k])
                    {
                        yIndex[k] = Height2 - yIndex[k];                    
                    }

                    zIndex[k] = (ZLen == 1) ? (0) :
                        ((zIndex[k] < 0) ?
                            (-zIndex[k] - Depth2 * ((-zIndex[k]) / Depth2)) :
                            (zIndex[k] - Depth2 * (zIndex[k] / Depth2)));
                    if (ZLen <= zIndex[k])
                    {
                        zIndex[k] = Depth2 - zIndex[k];                    
                    }
                }
            }

            //Perform interolation            
            int splineDegree = _degree;
            double w3 = 0.0;
            for (k = 0; k <= splineDegree; k++)
            {
                double w2 = 0.0;
                for (j = 0; j <= splineDegree; j++)
                {
                    double w1 = 0.0;
                    for (i = 0; i <= splineDegree; i++)
                    {
                        w1 += xWeight[i] * getCoef(xIndex[i], yIndex[j], zIndex[k]);
                    }
                    w2 += yWeight[j] * w1;
                }
                w3 += zWeight[k] * w2;
            }
            return w3;
        }
        private Single[] copyCoef(int buffer, Single[] copyFromCoeffs)
        {
            Single[] copyToCoeffs = new Single[(XLen - (buffer * 2)) * (YLen - (buffer * 2)) * (ZLen - (buffer * 2))];
            int fromCount = 0;
            int toCount = 0;

            for (int k = 0; k < ZLen; k++)
            {                
                for (int j = 0; j < YLen; j++)
                {                    
                    for (int i = 0; i < XLen; i++)
                    {
                        if (i >= buffer && i < XLen - buffer)
                        {
                            if (j >= buffer && j < YLen - buffer)
                            {
                                if (k >= buffer && k < ZLen - buffer)
                                {
                                    copyToCoeffs[toCount] = copyFromCoeffs[fromCount];
                                    ++toCount;
                                }

                            }

                        }
                        ++fromCount;                        
                    }
                }
            }
            return copyToCoeffs;
        }
        private double getCoef(int x, int y, int z)
        {
            int pos = getPosition(x, y, z);
            try
            {
                if (pos >=0 && _coefficients.Length > pos)
                {
                    Single value = _coefficients[pos];
                    return value;
                }
                else
                {
                    return 0;
                }
            }
            catch (Exception e)
            {
                return 0;
            }
            
        }
        private void putCoef(int x, int y, int z, double v)
        {
            int pos = getPosition(x, y, z);
            /*byte[] bv = BitConverter.GetBytes(Convert.ToSingle(v));
            int start = _bStart + pos * 4;
            foreach (byte b in bv)
            {
                _coefficients[start] = b;
                start++;
            }*/
            _coefficients[pos] = Convert.ToSingle(v);
            //if (_coefficients.ContainsKey(pos))
            //    _coefficients[pos] = v;            
        }
        
        private void createSampledCoefficients()
        {
            // we have been given a cube with a buffer, so create as usual then cut
            createCoefficients();
            Single[] copyCoeffs = getCubeEmpty(-1*_buffer);
            copyCoeffs = copyCoef(_buffer, _coefficients);
            _coefficients = copyCoeffs;
            XLen -= (2 * _buffer);
            YLen -= (2 * _buffer);
            ZLen -= (2 * _buffer);
        }
        private void createCoefficients()
        {            
            List<double> pole = getPole(_degree);
            int numPoles = Convert.ToInt32(pole.Count);

            //Convert the samples to interpolation coefficients
            //X-wise
            for (int y = 0; y < YLen; ++y)
            {
                for (int z = 0; z < ZLen; ++z)
                {
                    List<double> row = getRow3d(y, z, XLen);
                    List<double> line = convertToInterpolationCoefficients(pole, numPoles, XLen, row);
                    putRow3d(y, z, line, XLen);
                }
            }
            //Y-wise
            for (int x = 0; x < XLen; ++x)
            {
                for (int z = 0; z < ZLen; ++z)
                {
                    List<double> row = getColumn3d(x, z, YLen);
                    List<double> line = convertToInterpolationCoefficients(pole, numPoles, YLen, row);
                    putColumn3d(x, z, line, YLen);
                }
            }

            //Z-wise
            for (int x = 0; x < XLen; ++x)
            {
                for (int y = 0; y < YLen; ++y)
                {
                    List<double> row = getHole3d(x, y, ZLen);
                    List<double> line = convertToInterpolationCoefficients(pole, numPoles, ZLen, row);
                    putHole3d(x, y, line, ZLen);
                }
            }
        }

        private List<double> getRow3d(int y, int z, int length)
        {
            List<double> row = new List<double>();
            for (int x = 0; x < length; ++x)
                row.Add(getCoef(x, y, z));
            return row;
        }
        private void putRow3d(int y, int z, List<double> row, int length)
        {
            for (int x = 0; x < length; ++x)
                putCoef(x, y, z, row[x]);
        }
        private List<double> getColumn3d(int x, int z, int length)
        {
            List<double> col = new List<double>();
            for (int y = 0; y < length; ++y)
                col.Add(getCoef(x, y, z));
            return col;
        }
        private void putColumn3d(int x, int z, List<double> col, int length)
        {
            for (int y = 0; y < length; ++y)
                putCoef(x, y, z, col[y]);
        }
        private List<double> getHole3d(int x, int y, int length)
        {
            List<double> bore = new List<double>();
            for (int z = 0; z < length; ++z)
                bore.Add(getCoef(x, y, z));
            return bore;
        }
        private void putHole3d(int x, int y, List<double> bore, int length)
        {
            for (int z = 0; z < length; ++z)
                putCoef(x, y, z, bore[z]);
        }

        private List<double> getPole(int degree)
        {
            //Recover the poles from a lookup table #currently only 3 degree, will I want to calculate all the possibilities at the beginnning, 3,5,7,9?
            List<double> pole = new List<double>();
            if (degree == 9)
            {
                pole.Add(-0.60799738916862577900772082395428976943963471853991);
                pole.Add(-0.20175052019315323879606468505597043468089886575747);
                pole.Add(-0.043222608540481752133321142979429688265852380231497);
                pole.Add(-0.0021213069031808184203048965578486234220548560988624);
            }
            else if (degree == 7)
            {
                pole.Add(-0.53528043079643816554240378168164607183392315234269);
                pole.Add(-0.12255461519232669051527226435935734360548654942730);
                pole.Add(-0.0091486948096082769285930216516478534156925639545994);
            }
            else if (degree == 5)
            {
                pole.Add(Math.Sqrt(135.0 / 2.0 - Math.Sqrt(17745.0 / 4.0)) + Math.Sqrt(105.0 / 4.0) - 13.0 / 2.0);
                pole.Add(Math.Sqrt(135.0 / 2.0 + Math.Sqrt(17745.0 / 4.0)) - Math.Sqrt(105.0 / 4.0) - 13.0 / 2.0);
            }
            else//then it is 3
            {
                pole.Add(Math.Sqrt(3.0) - 2.0);
            }
            return pole;
        }

        private List<double> convertToInterpolationCoefficients(List<double> pole, int numPoles, int width, List<double> row)
        {
            /* special case required by mirror boundaries */
            if (width == 1)
            {                
                //mirror filter and periodic filter
                // not much can be done if it is only 1 thivk, it is both mirror and periodic at the same time
                return row; ;                                
            }

            double lambda = 1;
            int n = 0;
            int k = 0;
            //Compute the overall gain
            for (k = 0; k < numPoles; k++)
            {
                lambda = lambda * (1 - pole[k]) * (1 - 1 / pole[k]);
            }
            //Apply the gain
            for (n = 0; n < width; n++)
            {
                row[n] *= lambda;
            }
            //loop over the poles            
            for (k = 0; k < numPoles; k++)
            {
                /* causal initialization */
                if (_mirror)
                {
                    row[0] = InitialCausalCoefficient(row, width, pole[k]);
                }
                else //TODO
                {                    
                    row[0] = InitialCausalCoefficient(row, width, pole[k]);
                }
                /* causal recursion */
                for (n = 1; n < width; n++)
                {
                    row[n] += (double)pole[k] * row[n - 1];
                }
                /* anticausal initialization */
                if (_mirror)
                    row[width - 1] = InitialAntiCausalCoefficient(row, width, pole[k]);
                else //TODO
                    row[width - 1] = InitialAntiCausalCoefficient(row, width, pole[k]);
                /* anticausal recursion */
                for (n = width - 2; 0 <= n; n--)
                {
                    row[n] = pole[k] * (row[n + 1] - row[n]);
                }
            }
            return row;
        }

        private double InitialCausalCoefficient(List<double> c, int dataLength, double pole)

        { /* begin InitialCausalCoefficient */

            double Sum, zn, z2n, iz;
            int n, Horizon;

            /* this initialization corresponds to mirror boundaries */
            Horizon = dataLength;
            if (TOLERANCE > 0.0)
            {
                Horizon = (int)Math.Ceiling(Math.Log(TOLERANCE) / Math.Log(Math.Abs(pole)));
            }
            if (Horizon < dataLength)
            {
                /* accelerated loop */
                zn = pole;
                Sum = c[0];
                for (n = 1; n < Horizon; n++)
                {
                    Sum += zn * c[n];
                    zn *= pole;
                }
                return (Sum);
            }
            else// if (_mirror)
            {// RSA notes this is the mirror condition, when the horizon - how far you need to look ahead for good data - is not as far the data you have
                /* full loop */
                zn = pole;
                iz = 1.0 / pole;
                z2n = Math.Pow(pole, dataLength - 1);
                Sum = c[0] + z2n * c[dataLength - 1];
                z2n *= z2n * iz; //is this a mistake, should it be just *=??? Checked it is how it is in their code. NO TRIED IT.                
                for (n = 1; n <= dataLength - 2; n++)
                {
                    Sum += (zn + z2n) * c[n];
                    zn *= pole;
                    //z2n *= z2n * iz;
                    z2n *= iz;
                }
                return (Sum / (1.0 - zn * zn));
            }
            /*else
            {// RSA notes speculative attempt to make it periodic                
                zn = pole;
                Sum = c[0];
                for (n = 1; n < Horizon; n++)
                {
                    int m = n % dataLength;
                    Sum += zn * c[m];
                    zn *= pole;
                }
                return (Sum);
            }*/
        }
        private double InitialAntiCausalCoefficient(List<double> c, int dataLength, double pole)
        {
            /* this initialization corresponds to mirror boundaries */
            if (dataLength < 2)
                return 0;
            else// if (_mirror)
                return ((pole / (pole * pole - 1.0)) * (pole * c[dataLength - 2] + c[dataLength - 1]));
            //else //wrap to beginning instead
            //    return ((pole / (pole * pole - 1.0)) * (pole * c[1] + c[0]));
        }
        private List<double> applyValue3(double val, List<int> idc, int weight_length)
        {
            List<double> ws = new List<double>();
            for (int i = 0; i < weight_length; ++i)
                ws.Add(0);
            double w = val - (double)idc[1];
            ws[3] = (1.0 / 6.0) * w * w * w;
            ws[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - ws[3];
            ws[2] = w + ws[0] - 2.0 * ws[3];
            ws[1] = 1.0 - ws[0] - ws[2] - ws[3];
            return ws;
        }

        private List<double> applyValue5(double val, List<int> idc, int weight_length)
        {
            List<double> ws = new List<double>();
            for (int i = 0; i < weight_length; ++i)
                ws.Add(0);
            double w = val - (double)idc[2];
            double w2 = w * w;
            ws[5] = (1.0 / 120.0) * w * w2 * w2;
            w2 -= w;
            double w4 = w2 * w2;
            w -= 1.0 / 2.0;
            double t = w2 * (w2 - 3.0);
            ws[0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - ws[5];
            double t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
            double t1 = (-1.0 / 12.0) * w * (t + 4.0);
            ws[2] = t0 + t1;
            ws[3] = t0 - t1;
            t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
            t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
            ws[1] = t0 + t1;
            ws[4] = t0 - t1;
            return ws;
        }

        private List<double> applyValue7(double val, List<int> idc, int weight_length)
        {
            List<double> ws = new List<double>();
            for (int i = 0; i < weight_length; ++i)
                ws.Add(0);
            double w = val - (double)idc[3];
            ws[0] = 1.0 - w;
            ws[0] *= ws[0];
            ws[0] *= ws[0] * ws[0];
            ws[0] *= (1.0 - w) / 5040.0;
            double w2 = w * w;
            ws[1] = (120.0 / 7.0 + w * (-56.0 + w * (72.0 + w * (-40.0 + w2 * (12.0 + w * (-6.0 + w)))))) / 720.0;
            ws[2] = (397.0 / 7.0 - w * (245.0 / 3.0 + w * (-15.0 + w * (-95.0 / 3.0 + w * (15.0 + w * (5.0 + w * (-5.0 + w))))))) / 240.0;
            ws[3] = (2416.0 / 35.0 + w2 * (-48.0 + w2 * (16.0 + w2 * (-4.0 + w)))) / 144.0;
            ws[4] = (1191.0 / 35.0 - w * (-49.0 + w * (-9.0 + w * (19.0 + w * (-3.0 + w) * (-3.0 + w2))))) / 144.0;
            ws[5] = (40.0 / 7.0 + w * (56.0 / 3.0 + w * (24.0 + w * (40.0 / 3.0 + w2 * (-4.0 + w * (-2.0 + w)))))) / 240.0;
            ws[7] = w2;
            ws[7] *= ws[7] * ws[7];
            ws[7] *= w / 5040.0;
            ws[6] = 1.0 - ws[0] - ws[1] - ws[2] - ws[3] - ws[4] - ws[5] - ws[7];
            return ws;
        }

        private List<double> applyValue9(double val, List<int> idc, int weight_length)
        {
            List<double> ws = new List<double>();
            for (int i = 0; i < weight_length; ++i)
                ws.Add(0);
            double w = val - (double)idc[4];
            ws[0] = 1.0 - w;
            ws[0] *= ws[0];
            ws[0] *= ws[0];
            ws[0] *= ws[0] * (1.0 - w) / 362880.0;
            ws[1] = (502.0 / 9.0 + w * (-246.0 + w * (472.0 + w * (-504.0 + w * (308.0 + w * (-84.0 + w * (-56.0 / 3.0 + w * (24.0 + w * (-8.0 + w))))))))) / 40320.0;
            ws[2] = (3652.0 / 9.0 - w * (2023.0 / 2.0 + w * (-952.0 + w * (938.0 / 3.0 + w * (112.0 + w * (-119.0 + w * (56.0 / 3.0 + w * (14.0 + w * (-7.0 + w))))))))) / 10080.0;
            ws[3] = (44117.0 / 42.0 + w * (-2427.0 / 2.0 + w * (66.0 + w * (434.0 + w * (-129.0 + w * (-69.0 + w * (34.0 + w * (6.0 + w * (-6.0 + w))))))))) / 4320.0;
            double w2 = w * w;
            ws[4] = (78095.0 / 63.0 - w2 * (700.0 + w2 * (-190.0 + w2 * (100.0 / 3.0 + w2 * (-5.0 + w))))) / 2880.0;
            ws[5] = (44117.0 / 63.0 + w * (809.0 + w * (44.0 + w * (-868.0 / 3.0 + w * (-86.0 + w * (46.0 + w * (68.0 / 3.0 + w * (-4.0 + w * (-4.0 + w))))))))) / 2880.0;
            ws[6] = (3652.0 / 21.0 - w * (-867.0 / 2.0 + w * (-408.0 + w * (-134.0 + w * (48.0 + w * (51.0 + w * (-4.0 + w) * (-1.0 + w) * (2.0 + w))))))) / 4320.0;
            ws[7] = (251.0 / 18.0 + w * (123.0 / 2.0 + w * (118.0 + w * (126.0 + w * (77.0 + w * (21.0 + w * (-14.0 / 3.0 + w * (-6.0 + w * (-2.0 + w))))))))) / 10080.0;
            ws[9] = w2 * w2;
            ws[9] *= ws[9] * w / 362880.0;
            ws[8] = 1.0 - ws[0] - ws[1] - ws[2] - ws[3] - ws[4] - ws[5] - ws[6] - ws[7] - ws[9];
            return ws;
    
    
    
    
    public class OptBSpline : Interpolator
    {
        protected int _degree;
        protected int _dimsize;
        protected int _points;        
        protected int _opt_count = 2;
        protected BetaSpline? _bsp;
        //protected List<Tuple<int[], BetaSpline>> _cubeList = new List<Tuple<int[], BetaSpline>>();
        protected int _xFloor = 0;
        protected int _yFloor = 0;
        protected int _zFloor = 0;
        protected int _xCeil = 0;
        protected int _yCeil = 0;
        protected int _zCeil = 0;

        // ****** Linear Implementation ****************************************
        public OptBSpline(Single[] bytes, int start, int length, int x, int y, int z, int degree, int points, int copies) : base(bytes, start, length, x, y, z,copies)
        {
            // the degree must be an odd number
            _degree = degree;
            _points = points;
            if (_points > XLen)
                _points = XLen;
            if (_points > YLen)
                _points = YLen;
            if (_points > ZLen)
                _points = ZLen;

            _dimsize = (int)Math.Pow(_points, 3);
            _bsp = null;
        }        
        public override double getValue(double xx, double yy, double zz)
        {
            if (!isValid(xx, yy, zz))
                return 0;
            double[] xyz = adjustReflection(xx, yy, zz);
            double x = xyz[0];
            double y = xyz[1];
            double z = xyz[2];

            // The method of linear interpolation is a version of my own method for multivariate fitting, instead of trilinear interpolation
            // NOTE I could extend this to be multivariate not linear but it has no advantage over bspline - and is slower and not as good 
            // Document is here: https://rachelalcraft.github.io/Papers/MultivariateInterpolation/MultivariateInterpolation.pdf

            //1.  do we need a new cube?
            // a) we do if we don't have one

            // b) we do if we are within points/2 of the edges of the edges
            int myxFloor = (int)Math.Floor(x + (-1 * _buffer) + 1);
            int myyFloor = (int)Math.Floor(y + (-1 * _buffer) + 1);
            int myzFloor = (int)Math.Floor(z + (-1 * _buffer) + 1);            
            //int myxFloor = (int)x - (_points / 2 + _buffer);
            //int myyFloor = (int)y - (_points / 2 + _buffer);
            //int myzFloor = (int)z - (_points / 2 + _buffer);

            if (_seedWidth == -1 || _bsp == null)
            {
                selectBSpline(myxFloor, myyFloor, myzFloor, _points);
            }
            else
            {
                checkBSpline(myxFloor, myyFloor, myzFloor);
            }
            

            //2. Find the values that would be in this cube             
            double xn = x - _xFloor;
            double yn = y - _yFloor;
            double zn = z - _zFloor;

            //3. Apply the interpolaor to find the value
            if (_bsp != null)                
                return _bsp.getValue(xn, yn, zn);           
            else
                return 0;
        }

        public void makeCentreBSpline(double x, double y, double z, int points)
        {
            int xFloor = (int)Math.Floor(x + (-1 * _buffer) + 1);
            int yFloor = (int)Math.Floor(y + (-1 * _buffer) + 1);
            int zFloor = (int)Math.Floor(z + (-1 * _buffer) + 1);
            //int xFloor = (int)x - (points / 2 + _buffer);
            //int yFloor = (int)y - (points / 2 + _buffer);
            //int zFloor = (int)z - (points / 2 + _buffer);
            _xFloor = xFloor;
            _yFloor = yFloor;
            _zFloor = zFloor;            
            int buffered_points = points + (_buffer * 2);
            _xCeil = _xFloor + buffered_points;
            _yCeil = _yFloor + buffered_points;
            _zCeil = _zFloor + buffered_points;
            Single[] vals = getSmallerCubeThevenaz(xFloor, yFloor, zFloor, buffered_points);
            //3. Kind of recursive, make a smaller BSlipe interpolator out of this.                             
            _bsp = new BetaSpline(vals, 0, buffered_points * buffered_points * buffered_points, buffered_points, buffered_points, buffered_points, _degree, _copies, true, false);
        }
        private void checkBSpline(int xFloor, int yFloor, int zFloor)
        {
            bool need = false;
            if (!(Math.Abs(xFloor - _xFloor) < _buffer))
                need = true;
            if (!(Math.Abs(yFloor - _yFloor) < _buffer))
                need = true;
            if (!(Math.Abs(zFloor - _zFloor) < _buffer))
                need = true;
        }
        private void selectBSpline(int xFloor, int yFloor, int zFloor, int points)
        {
            bool need = true;            
            //foreach (var cuver in _cubeList)
            //{
            //    int[] points = cuver.Item1;
            need = false;            
            if (!(Math.Abs(xFloor - _xFloor) < _buffer))
                need = true;
            if (!(Math.Abs(yFloor - _yFloor) < _buffer))
                need = true;
            if (!(Math.Abs(zFloor - _zFloor) < _buffer))
                need = true;

            if (!need && _bsp != null)
            {
                //bsp = cuver.Item2;
                //_xFloor = points[0];
                //_yFloor = points[1];
                //_zFloor = points[2];
                //return bsp;
            }
            //}
            else //which it must be or it returned
            {
                _xFloor = xFloor;
                _yFloor = yFloor;
                _zFloor = zFloor;
                int buffered_points = points + (_buffer * 2);
                _xCeil = _xFloor + buffered_points;
                _yCeil = _yFloor + buffered_points;
                _zCeil = _zFloor + buffered_points;
                Single[] vals = getSmallerCubeThevenaz(xFloor, yFloor, zFloor, buffered_points);
                //3. Kind of recursive, make a smaller BSlipe interpolator out of this.                             
                _bsp = new BetaSpline(vals, 0, buffered_points * buffered_points * buffered_points, buffered_points, buffered_points, buffered_points, _degree, 0, true, false);
                //_cubeList.Add(Tuple.Create(new int[]{ xFloor, yFloor, zFloor }, bsp));

                //if (_cubeList.Count > _opt_count)
                //{
                //    int[] vvv = _cubeList[0].Item1;
                //    _cubeList.RemoveAt(0);
                //}
            }            
        }
    }
    
    """