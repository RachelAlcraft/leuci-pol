"""


"""
from abc import ABC, abstractmethod
from leuci_xyz import vectorthree as v3
import math
import numpy as np

#from . import invariant as ivm
from . import iv1
from . import iv3
from . import iv5




### Factory method for creation ##############################################################
def create_interpolator(method, values, F, M, S, npy=None,degree=-1, log_level=0):
    if log_level > 0:
        print("Interpolator:",method,F,M,S,"degree=",degree)
    intr = None
    if method == "linear":
        intr = Multivariate(values,F, M, S,npy, 1, log_level)
    elif method == "cubic" and degree == 5: #not really cubic, but quintic
        intr = Multivariate(values,F, M, S,npy, 5, log_level)
    elif method == "cubic":
        intr = Multivariate(values,F, M, S,npy, 3, log_level)
    elif method == "bspline" and degree == 5:
        intr = Bspline(values,F, M, S,npy, 5, log_level) # I am being cautious about degrees
    elif method == "bspline":
        intr = Bspline(values,F, M, S,npy, 3, log_level) 
    else: #nearest is default
        intr = Nearest(values,F, M, S,npy,0,log_level)
    intr.init()
    return intr

### Abstract class ############################################################################
class Interpolator(ABC):
    def __init__(self, values, F, M, S, npy=None,degree=-1,log_level=0):
        self.use_jax = False                
        if npy != None:
            self._npy = npy
        else:
            count = 0
            self._npy = np.zeros((F,M,S))
            for k in range(S):
                for j in range(M):
                    for i in range(F):
                        self._npy[i,j,k] = values[count]
                        count += 1
        self._F = F
        self._M = M
        self._S = S
        self.degree = degree
        self._buffer = 28
        self.log_level = log_level         
        self.h = 0.001  #this is the interval for numerical differentiation
        
    ############################################################################################################            
    @abstractmethod
    def get_value(self, x, y, z):
        pass
    
    @abstractmethod
    def get_radient(self, x, y, z):
        pass
    
    @abstractmethod
    def get_laplacian(self, x, y, z):
        pass


    ############################################################################################################
    # implemented interface that is the same for all abstractions
    def get_radient_numerical(self, x, y, z):
        # we don't want to cross a grid bounday, so our gradient should be smaller than the distance to a grid.                
        val = self.get_value(x, y, z)
        dx = (self.get_value(x + self.h, y, z) - val) / self.h
        dy = (self.get_value(x, y + self.h, z) - val) / self.h
        dz = (self.get_value(x, y, z + self.h) - val) / self.h
        radient = (abs(dx) + abs(dy) + abs(dz)) / 3
        return radient
        
    def get_laplacian_numerical(self, x, y, z):                
        val = self.get_value(x, y, z)
        xx = self.getDxDx_numerical(x, y, z, val)
        yy = self.getDyDy_numerical(x, y, z, val)
        zz = self.getDzDz_numerical(x, y, z, val)
        ll = xx + yy + zz 
        if ll > 170: #RSA TODO
            print(x,y,z,ll)
        return ll
        
    def getDxDx_numerical(self, x, y, z, val):        
        va = self.get_value(x - self.h, y, z)
        vb = self.get_value(x + self.h, y, z)
        dd = (va + vb - 2 * val) / (self.h * self.h)
        return dd
        
    def getDyDy_numerical(self, x, y, z, val):        
        va = self.get_value(x, y - self.h, z)
        vb = self.get_value(x, y + self.h, z)
        dd = (va + vb - 2 * val) / (self.h * self.h)
        return dd
        
    def getDzDz_numerical(self, x, y, z, val):        
        va = self.get_value(x, y, z - self.h)
        vb = self.get_value(x, y, z + self.h)
        dd = (va + vb - 2 * val) / (self.h * self.h)
        return dd
                
    def get_fms(self,f,m,s,F=-1,M=-1,S=-1):        
        u_f, u_m, u_s = self.get_adjusted_fms(f,m,s,F,M,S)
        return self._npy[int(u_f),int(u_m),int(u_s)]
        #pos = self.get_pos_from_fms(u_f,u_m,u_s,F,M,S)
        #return self._values[pos]
    
    def get_projection(self,slice):        
        if slice == "xy":        
            return self._npy.max(axis=(2))
        elif slice == "yz":        
            return self._npy.max(axis=(0))
        elif slice == "zx":        
            return self._npy.max(axis=(1))
    
    def get_cross_section(self,slice,layer):        
        if slice == "xy":        
            return self._npy[:,:,layer]
        elif slice == "yz":        
            return self._npy[layer,:,:]            
        elif slice == "zx":        
            return self._npy[:,layer,:]
            
    def get_pos_from_fms(self, f, m, s,F=-1,M=-1,S=-1):
        use_f,use_m,use_s = self._F, self._M, self._S
        if F+M+S != -3:
            use_f,use_m,use_s = F,M,S
        slice_area = use_f * use_m
        pos = s * slice_area        
        pos += use_f * m        
        pos += f        
        return pos
    
    def get_adjusted_fms(self, f, m, s,F=-1,M=-1,S=-1):
        use_f,use_m,use_s = self._F, self._M, self._S
        if F+M+S != -3:
            use_f,use_m,use_s = F,M,S
        u_f = f
        u_m = m
        u_s = s
        # Unit wrap F
        while u_f < 0:
            u_f += use_f
        while u_f >= use_f:
            u_f -= use_f
        # Unit wrap M
        while u_m < 0:
            u_m += use_m
        while u_m >= use_m:
            u_m -= use_m
        # Unit wrap S
        while u_s < 0:
            u_s += use_s
        while u_s >= use_s:
            u_s -= use_s
        return u_f, u_m, u_s
        
    def get_fms_from_pos(self, A):        
        f,m,c = 0,0,0
        left = A
        area = int(self._F * self._M)        
        if left >= area:
            dv,md = divmod(left,area)            
            c = dv
            left = md
        if left >= self. _F:            
            dv,md = divmod(left,self._F)
            m = dv
            left = md
        f = left
        return f,m,c

    def closest(self,point):
        cnrs = self.corners(point)
        dis = 10000
        pnt = None
        for cnr in cnrs:
            mag = point.distance(cnr)
            if mag < dis:
                pnt = cnr
                dis = mag        
        return v3.VectorThree(pnt.A,pnt.B,pnt.C)
    
    def corners(self,point, far=1):
        cnrs = []        
        x = np.ceil(point.A)
        y = np.ceil(point.B)
        z = np.ceil(point.C)
        for f in range(0-far,far):
            for m in range(0-far,far):
                for s in range(0-far,far):
                    cnrs.append(v3.VectorThree(x+f,y+m,z+s)) 
        return cnrs
    
    def get_val_slice(self,unit_coords, deriv = 0):
        vals = []
        for i in range(len(unit_coords)):
            row = []
            for j in range(len(unit_coords[0])):
                vec = unit_coords[i][j]
                if self.log_level > 2:
                    print("Get value", vec.A,vec.B,vec.C)
                if deriv == 2:
                    vec_val = self.get_laplacian(vec.A,vec.B,vec.C)
                elif deriv == 1:
                    vec_val = self.get_radient(vec.A,vec.B,vec.C)
                else:
                    vec_val = self.get_value(vec.A,vec.B,vec.C)
                row.append(vec_val)
            vals.append(row)
        return vals

    def build_cube_around(self, x, y, z, width):        
        # 1. Build the points around the centre as a cube - width points
        vals = []            
        xp,yp,zp = 0,0,0 
        for i in range(int(-1*width/2 + 1), int(width/2 + 1)):
            xp = np.floor(x + i)
            for j in range(int(-1*width/2 + 1), int(width/2 + 1)):      
                yp = np.floor(y + j)
                for k in range(int(-1*width/2 + 1), int(width/2 + 1)):          
                    zp = np.floor(z + k)                        
                    p = self.get_fms(xp, yp, zp)
                    vals.append(p)                    
        npvals = np.array(vals)    
        return npvals

    def mult_vector(self,A, V):     #A and V are numpy arrays
        use_np = True
        if use_np:
            mm = np.matmul(A,V)            
            return mm
        else:   
            length = len(V)
            results = []
            for row in range(length):        
                sum = 0
                for col in range(length):
                    mv = A[row,col]
                    vv = V[col]
                    sum += mv*vv                
                results.append(sum)
            return results
    

####################################################################################################
### NEAREST NEIGHBOUR
####################################################################################################
class Nearest(Interpolator):                
    def init(self):
        pass
    def get_value(self, x, y, z):
        closest_pnt = self.closest(v3.VectorThree(x,y,z))  
        #print(closest_pnt.A, closest_pnt.B,closest_pnt.C)      
        return self.get_fms(closest_pnt.A, closest_pnt.B,closest_pnt.C)    
    
    ## implement abstract interface #########################################
    def get_radient(self, x, y, z):
        return self.get_radient_numerical(x,y,z)
    
    def get_laplacian(self, x, y, z):
        return self.get_laplacian_numerical(x,y,z)
    ## iplement abstract interface ###########################################
####################################################################################################
### Multivariate - Linear and Cubic
####################################################################################################
class Multivariate(Interpolator):                
    def init(self):
        self.points = self.degree + 1
        if self.use_jax:
            self.dimsize = np.power(self.points, 3)
        else:
            self.dimsize = math.pow(self.points, 3)
        if self.degree == 1:
            self.inv = iv1.InvariantVandermonde()
        elif self.degree == 3:
            self.inv = iv3.InvariantVandermonde()
        elif self.degree == 5:
            self.inv = iv5.InvariantVandermonde()
        self.need_new = True
        self._xfloor = -1
        self._yfloor = -1
        self._zfloor = -1
    
    ## implement abstract interface #########################################
    def get_radient(self, x, y, z):
        zn, yn, xn, self.polyCoeffs = self.make_coeffs(x,y,z)                
        dx = self.get_value_multivariate(zn, yn, xn, self.polyCoeffs,wrt=["x"])
        dy = self.get_value_multivariate(zn, yn, xn, self.polyCoeffs,wrt=["y"])
        dz = self.get_value_multivariate(zn, yn, xn, self.polyCoeffs,wrt=["z"])
        return (abs(dx) + abs(dy) + abs(dz)/3)
        
    
    def get_laplacian(self, x, y, z):
        return self.get_laplacian_numerical(x,y,z)
    ## iplement abstract interface ###########################################

    
    def diffWRTx(self,coeffs):        
        x,y,z = coeffs.shape
        partialX = np.zeros((x,y,z))
        for i in range(1, x):
            for j in range(0,y):
                for k in range(0,z):
                    partialX[i-1,j,k] = coeffs[i,j,k]*i
        return partialX

    def diffWRTy(self,coeffs):        
        x,y,z = coeffs.shape
        partialY = np.zeros((x,y,z))
        for i in range(0, x):
            for j in range(1,y):
                for k in range(0,z):
                    partialY[i,j-1,k] = coeffs[i,j,k]*j
        return partialY
    
    def diffWRTz(self,coeffs):        
        x,y,z = coeffs.shape
        partialZ = np.zeros((x,y,z))
        for i in range(0, x):
            for j in range(0,y):
                for k in range(1,z):
                    partialZ[i,j,k-1] = coeffs[i,j,k]*k
        return partialZ
        
                
    def make_coeffs(self, x, y, z):
        recalc = self.need_new
        #we can reuse our last matrix if the points are within the same unit cube
        xFloor = np.floor(x)
        yFloor = np.floor(y)
        zFloor = np.floor(z)
        if not recalc:
            if (xFloor != self.xfloor):
                recalc = True
            if (yFloor != self.yfloor):
                recalc = True
            if (zFloor != self.zfloor):
                recalc = True

        if (recalc):        
            self.xfloor = xFloor
            self.yfloor = yFloor
            self.zfloor = zFloor
            # 1. Build the points around the centre as a cube - 8 points
            vals = self.build_cube_around(x, y, z, self.points)
            #2. Multiply with the precomputed matrix to find the multivariate polynomial
            ABC = self.mult_vector(self.inv.get_invariant(), vals)
            # 3. Put the 8 values back into a cube
            self.polyCoeffs = np.zeros((self.points, self.points, self.points))
            pos = 0
            for i in range(self.points):
                for j in range(self.points):                
                    for k in range(self.points):                    
                        self.polyCoeffs[i, j, k] = ABC[pos]
                        pos+=1
            self.need_new = False        
        #4. Adjust the values to be within this cube
        pstart = (-1 * self.points / 2) + 1        
        xn = x - xFloor - pstart
        yn = y - yFloor - pstart
        zn = z - zFloor - pstart
        return xn,yn,zn,self.polyCoeffs

    def get_value(self, x, y, z):
        return self.get_value_old(x,y,z)
    
    def get_value_old(self, x, y, z):        
        recalc = self.need_new
        #we can reuse our last matrix if the points are within the same unit cube
        xFloor = np.floor(x)
        yFloor = np.floor(y)
        zFloor = np.floor(z)
        if not recalc:
            if (xFloor != self.xfloor):
                recalc = True
            if (yFloor != self.yfloor):
                recalc = True
            if (zFloor != self.zfloor):
                recalc = True

        if (recalc):        
            self.xfloor = xFloor
            self.yfloor = yFloor
            self.zfloor = zFloor
            # 1. Build the points around the centre as a cube - 8 points
            vals = self.build_cube_around(x, y, z, self.points)
            #2. Multiply with the precomputed matrix to find the multivariate polynomial
            ABC = self.mult_vector(self.inv.get_invariant(), vals)
            # 3. Put the 8 values back into a cube
            self.polyCoeffs = np.zeros((self.points, self.points, self.points))
            pos = 0
            for i in range(self.points):
                for j in range(self.points):                
                    for k in range(self.points):                    
                        self.polyCoeffs[i, j, k] = ABC[pos]
                        pos+=1
            self.need_new = False        
        #4. Adjust the values to be within this cube
        pstart = (-1 * self.points / 2) + 1        
        xn = x - xFloor - pstart
        yn = y - yFloor - pstart
        zn = z - zFloor - pstart

        return self.get_value_multivariate(xn,yn,zn,self.polyCoeffs)
                        

    def get_value_new(self, x, y, z):
        # The method of linear interpolation is a version of my own method for multivariate fitting, instead of trilinear interpolation
        # NOTE I could extend this to be multivariate not linear but it has no advantage over bspline - and is slower and not as good 
        # Document is here: https://rachelalcraft.github.io/Papers/MultivariateInterpolation/MultivariateInterpolation.pdf                        
        zn, yn, xn, self.polyCoeffs = self.make_coeffs(x,y,z)                
        return self.get_value_multivariate(zn, yn, xn, self.polyCoeffs,wrt=[])
        
    def get_value_multivariate(self, x, y, z, coeffs,wrt=[]):        
        #This is using a value scheme that makes sens of our new fitted polyCube
        #In a linear case it will be a decimal between 0 and 1    
        # wrt in format [x,x]      means 2nd partioal deriv for x etc....
        calc_mat = coeffs
        for partial in wrt:
            if partial == "x":
                calc_mat = self.diffWRTx(calc_mat)
            elif partial == "y":
                calc_mat = self.diffWRTy(calc_mat)
            elif partial == "z":
                calc_mat = self.diffWRTz(calc_mat)

        value = 0
        ii,jj,kk = calc_mat.shape
        for i in range(ii):        
            for j in range(jj):            
                for k in range(kk):                
                    coeff = calc_mat[i, j, k];                    
                    val = coeff * np.power(z, i) * np.power(y, j) * np.power(x, k)
                    value = value + val                                    
        return value
            
####################################################################################################
### B-Spline
####################################################################################################
class Bspline(Interpolator):                
    """
    ****** Thevenaz Spline Convolution Implementation ****************************************
    Thevenaz, Philippe, Thierry Blu, and Michael Unser. ?Image Interpolation and Resampling?, n.d., 39.
    http://bigwww.epfl.ch/thevenaz/interpolation/
    *******************************************************************************
    """
    def init(self):        
        self.TOLERANCE = 2.2204460492503131e-016 # smallest such that 1.0+DBL_EPSILON != 1.0                
        self.mirror = False                
        self.make_periodic_coeffs() #temp make it just ordinary coeffs
            
    ## implement abstract interface #########################################
    def get_radient(self, x, y, z):
        return self.get_radient_numerical(x,y,z)
    
    def get_laplacian(self, x, y, z):
        return self.get_laplacian_numerical(x,y,z)
    ## iplement abstract interface ###########################################

    def make_periodic_coeffs(self):           
        # we make the coefficients matrix a bit bigger than the vaues and have it wrap, and then cut it back down to the values                 
        self._coeffs =self.extend_vals_with_buffer(self._buffer)
        tmpF = self._F + (2 * self._buffer)
        tmpM = self._M + (2 * self._buffer)
        tmpS = self._S + (2 * self._buffer)                
        self.create_coeffs(tmpF,tmpM,tmpS)        
        self._coeffs = self.reduce_vals_with_buffer(self._buffer, self._coeffs,tmpF,tmpM,tmpS)        
        #self._coeffs = np.copy(self._npy)
        #self.create_coeffs(self._F,self._M,self._S)
        
    def reduce_vals_with_buffer(self,buffer, mynpy,tmpF,tmpM,tmpS):
        smnpy = np.zeros((self._F,self._M,self._S))
        for z in range(self._S):            
            for y in range(self._M):                
                for x in range(self._F):                    
                    u_x, u_y, u_z = x+buffer,y+buffer,z+buffer
                    val = mynpy[u_x,u_y,u_z]
                    smnpy[x,y,z] = val
        return smnpy
        
    def extend_vals_with_buffer(self,buffer):        
        #// 1. Make a buffer padded values cube for periodic values        
        xx,yy,zz = self._F, self._M, self._S
        xx,yy,zz = xx+2*buffer,yy+2*buffer,zz+2*buffer
        mynpy = np.zeros((xx,yy,zz))
        for z in range(zz):
            for y in range(yy):
                for x in range(xx):
                    p = self.get_fms(x-buffer,y-buffer,z-buffer)
                    mynpy[x,y,z] = p                    
        return mynpy
    
                                              
    def create_coeffs(self, thisF, thisM, thisS):
        pole = self.get_pole(self.degree)
        num_poles = len(pole)
        #Convert the samples to interpolation coefficients
        #X-wise
        for y in range(thisM):        
            for z in range(thisS):        
                row = self.get_row3d(y, z, thisF,thisF, thisM, thisS)
                line = self.convert_to_interp_coeffs(pole, num_poles, thisF, row,self.mirror)
                self.put_row3d(y, z, line, thisF,thisF, thisM, thisS)
        #Y-wise
        for x in range(thisF):        
            for z in range(thisS):        
                row = self.get_col3d(x, z, thisM,thisF, thisM, thisS)
                line = self.convert_to_interp_coeffs(pole, num_poles, thisM, row,self.mirror)
                self.put_col3d(x, z, line, thisM,thisF, thisM, thisS)           
        #Z-wise
        for x in range(thisF):
            for y in range(thisM):        
                row = self.get_hole3d(x, y, thisS,thisF, thisM, thisS)
                line = self.convert_to_interp_coeffs(pole, num_poles, thisS, row,self.mirror)
                self.put_hole3d(x, y, line, thisS,thisF, thisM, thisS)
                                        
    
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

    def get_row3d(self,y,z,length,F,M,S):        
        row = []
        for x in range(length):
            row.append(self.get_coeff(x, y, z,F,M,S))
        return row
    
    def get_col3d(self,x,z,length,F,M,S):        
        col = []
        for y in range(length):
            col.append(self.get_coeff(x, y, z,F,M,S))
        return col
    
    def get_hole3d(self,x,y,length,F,M,S):        
        bore = []
        for z in range(length):
            bore.append(self.get_coeff(x, y, z,F,M,S))
        return bore
    
    def put_row3d(self,y,z,row,length,F,M,S):        
        for x in range(length):
            self.put_coeff(x, y, z, row[x],F,M,S)
    
    def put_col3d(self,x,z,col,length,F,M,S):        
        for y in range(length):
            self.put_coeff(x, y, z, col[y],F,M,S)
    
    def put_hole3d(self,x,y,bore,length,F,M,S):        
        for z in range(length):
            self.put_coeff(x, y, z, bore[z],F,M,S)

    def convert_to_interp_coeffs(self,pole, num_poles, width, row,mirror):        
        #/* special case required by mirror boundaries */
        if (width == 1):         
            #mirror filter and periodic filter
            #not much can be done if it is only 1 thivk, it is both mirror and periodic at the same time
            return row        
        lmbda = 1
        n = 0
        k = 0
        #Compute the overall gain
        for k in range(num_poles):
            lmbda = lmbda * (1 - pole[k]) * (1 - 1 / pole[k])        
        #Apply the gain
        for n in range(width):
            row[n] = row[n] * lmbda        
        #loop over the poles            
        for k in range(num_poles):        
            #/* causal initialization */
            if (mirror):
                row[0] = self.initial_causal_coeffs(row, width, pole[k])
            else:# //TODO                       
                row[0] = self.initial_causal_coeffs(row, width, pole[k])            
            #/* causal recursion */
            for n in range(1,width):            
                row[n] = row[n]+ pole[k] * row[n - 1]
            #/* anticausal initialization */
            if (mirror):
                row[width - 1] = self.initial_anticausal_coeffs(row, width, pole[k])
            else:# //TODO
                row[width - 1] = self.initial_anticausal_coeffs(row, width, pole[k])
            #/* anticausal recursion */
            for n in range(width - 2, -1, -1):
                row[n] = pole[k] * (row[n + 1] - row[n])
        return row
    
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
            z2n = np.pow(pole, length - 1)
            Sum = vals[0] + z2n * vals[length - 1]
            z2n *= z2n * iz 
            for n in range(1,length - 2):
                Sum += (zn + z2n) * vals[n]
                zn *= pole                
                z2n *= iz
            return (Sum / (1.0 - zn * zn))        
                
    def initial_anticausal_coeffs(self, vals, length, pole):        
        #/* this initialization corresponds to mirror boundaries */
        if (length < 2):
            return 0;
        else:#// if (_mirror)
            return ((pole / (pole * pole - 1.0)) * (pole * vals[length - 2] + vals[length - 1]))

    def get_coeff(self,x,y,z,F,M,S):        
        # this naturally wraps round
        u_x, u_y, u_z = self.get_adjusted_fms(x,y,z,F=F,M=M,S=S)
        #pos = int(self.get_pos_from_fms(u_x,u_y,u_z,F=F,M=M,S=S))
        value = self._coeffs[u_x,u_y,u_z]
        return value
                                        
    def put_coeff(self,x,y,z,v,F,M,S):
        #u_x, u_y, u_z = self.get_adjusted_fms(x,y,z,F=F,M=M,S=S)
        #pos = int(self.get_pos_from_fms(u_x,u_y,u_z,F=F,M=M,S=S))
        self._coeffs[x,y,z] = v

    def get_value(self, x, y, z):        
        u_x, u_y, u_z = self.get_adjusted_fms(x,y,z)
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
        i = int(np.floor(u_x) - np.floor(self.degree / 2))
        j = int(np.floor(u_y) - np.floor(self.degree / 2))
        k = int(np.floor(u_z) - np.floor(self.degree / 2))

        for l in range(self.degree+1):                          
            xIndex[l] = i #if 71.1 passed in, for linear, we would want 71 and 72, 
            yIndex[l] = j
            zIndex[l] = k
            i,j,k=i+1,j+1,k+1              
        #/* compute the interpolation weights */
        if (self.degree == 9):        
            xWeight = self.applyValue9(u_x, xIndex, weight_length)
            yWeight = self.applyValue9(u_y, yIndex, weight_length)
            zWeight = self.applyValue9(u_z, zIndex, weight_length)        
        elif (self.degree == 7):        
            xWeight = self.applyValue7(u_x, xIndex, weight_length)
            yWeight = self.applyValue7(u_y, yIndex, weight_length)
            zWeight = self.applyValue7(u_z, zIndex, weight_length)        
        elif (self.degree == 5):        
            xWeight = self.applyValue5(u_x, xIndex, weight_length)
            yWeight = self.applyValue5(u_y, yIndex, weight_length)
            zWeight = self.applyValue5(u_z, zIndex, weight_length)        
        else:        
            xWeight = self.applyValue3(u_x, xIndex, weight_length)
            yWeight = self.applyValue3(u_y, yIndex, weight_length)
            zWeight = self.applyValue3(u_z, zIndex, weight_length)        
        #//applying the mirror boundary condition becaue I am only interpolating within values??
        #// RSA edit actually I want to wrap        
        # !!! I have removed the mirror boundary bit, might need to put it back in RSA TODO         
        #Perform interolation            
        spline_degree = self.degree
        w3 = 0.0
        for k in range(spline_degree+1):
            w2 = 0.0
            for j in range(spline_degree+1):
                w1 = 0.0
                for i in range(spline_degree+1):                
                    w1 += xWeight[i] * self.get_coeff(xIndex[i], yIndex[j], zIndex[k],self._F,self._M,self._S)
                w2 += yWeight[j] * w1
            w3 += zWeight[k] * w2
        return w3
    
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
        
        
            
        
        
        
        
        
        