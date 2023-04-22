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
        intr = Multivariate(values,F, M, S,1, log_level)
    ## DEGREE 5 explicitly ##
    elif method == "cubic" and degree == 5: #not really cubic, but quintic
        intr = Multivariate(values,F, M, S,5, log_level)
    elif method == "bspline" and degree == 5:
        intr = Bspline(values,F, M, S, 5, log_level) # I am being cautious about degrees
    ### degree 3 ###
    elif method == "cubic":
        intr = Multivariate(values,F, M, S,3, log_level)    
    elif method == "mspline":
        intr = Mspline(values,F, M, S,3, log_level) 
    elif method == "bspline":
        intr = Bspline(values,F, M, S,3, log_level) 
    else: #nearest is default
        intr = Nearest(values,F, M, S,0,log_level)
    intr.init()
    return intr

### Abstract class ############################################################################
class Interpolator(ABC):
    def __init__(self, values, F, M, S, degree=-1,log_level=0):                             
        self._npy = np.copy(values).astype(float)
        self._F = F
        self._M = M
        self._S = S
        self.degree = degree        
        self.log_level = log_level         
        self.h = 0.0001 #this is the interval for numerical differentiation
        self.padded = False
        self.buffer = 14
        self.round = 12
        self.TOLERANCE = 2.2204460492503131e-016 # smallest such that 1.0+DBL_EPSILON != 1.0                                
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
    def make_radient(self,dx,dy,dz):
        radient = abs(dx) + abs(dy) + abs(dz)
        #radient = dx+dy+dz                                
        return radient
    
    def make_laplacian(self,ddx,ddy,ddz):
        laplacian = ddx + ddy + ddz 
        return laplacian

    def get_radient_numerical(self, x, y, z):                        
        val = self.get_value(x, y, z)
        dx = (self.get_value(x + self.h, y, z) - val) / self.h
        dy = (self.get_value(x, y + self.h, z) - val) / self.h
        dz = (self.get_value(x, y, z + self.h) - val) / self.h        
        return self.make_radient(dx,dy,dz)
        
    def get_laplacian_numerical(self, x, y, z):        
        val = self.get_value(x, y, z)
        xx = self.getDxDx_numerical(x, y, z, val)
        yy = self.getDyDy_numerical(x, y, z, val)
        zz = self.getDzDz_numerical(x, y, z, val)
        return self.make_laplacian(xx,yy,zz)
        
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
    
    def get_projection(self,slice,xmin=-1,xmax=-1,ymin=-1,ymax=-1):
        vals = None        
        if slice == "xy":        
            vals = self._npy.max(axis=(2))
        elif slice == "yz":        
            vals = self._npy.max(axis=(0))
        elif slice == "zx":        
            vals = self._npy.max(axis=(1))
        
        if xmin == -1 and xmax == -1 and ymin == -1 and ymax == -1:
            return vals
        
        imax,jmax = vals.shape
        newvals = np.zeros((xmax-xmin,ymax-ymin))
        for i in range(xmin,xmax):
            for j in range(ymin,ymax):
                x,y = i,j
                a,b = i-xmin,j-ymin
                while x < 0:
                    x += imax
                while y < 0:
                    y += jmax
                x = x%imax
                y = y%jmax
                newvals[a,b] = vals[x,y]
        return newvals
    
    def get_cross_section(self,slice,layer):        
        if slice == "xy":        
            return self._npy[:,:,layer]
        elif slice == "yz":        
            return self._npy[layer,:,:]            
        elif slice == "zx":        
            return self._npy[:,layer,:]
            get_adjusted_fmsbuild
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
        u_f,u_m,u_s = f,m,s        
        addon = 0
        #if self.padded:
        #    u_f,u_m,u_s = u_f+self.buffer,u_m+self.buffer,u_s+self.buffer        
        #    #use_f,use_m,use_s = use_f+self.buffer,use_m+self.buffer,use_s+self.buffer        
        #    addon = self.buffer
        # Unit wrap F
        while u_f < 0+addon:
            u_f += use_f
        while u_f >= use_f:
            u_f -= use_f
        # Unit wrap M
        while u_m < 0+addon:
            u_m += use_m
        while u_m >= use_m:
            u_m -= use_m
        # Unit wrap S
        while u_s < 0+addon:
            u_s += use_s
        while u_s >= use_s:
            u_s -= use_s
        return round(u_f,self.round), round(u_m,self.round), round(u_s,self.round)
        
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
        al,au = math.floor(round(point.A,self.round)),math.ceil(round(point.A,self.round))
        bl,bu = math.floor(round(point.B,self.round)),math.ceil(round(point.B,self.round))
        cl,cu = math.floor(round(point.C,self.round)),math.ceil(round(point.C,self.round))
        a,b,c = 0,0,0
        if round(point.A,self.round)-al <= au-round(point.A,self.round):
            a=al
        else:
            a=au
        if round(point.B,self.round)-bl <= bu-round(point.B,self.round):
            b=bl
        else:
            b=bu
        if round(point.C,self.round)-cl <= cu-round(point.C,self.round):
            c=cl
        else:
            c=cu        
        return v3.VectorThree(a,b,c)
    
    def corners(self,point, far=1):
        cnrs = []        
        x = np.ceil(round(point.A,self.round))
        y = np.ceil(round(point.B,self.round))
        z = np.ceil(round(point.C,self.round))
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
        #vals = []            
        #print("---",x,y,z)
        arr = np.zeros((width*width*width))        
        xp,yp,zp = 0,0,0 
        count = 0
        for i in range(int(-1*width/2 + 1), int(width/2 + 1)):
            xp = np.floor(round(x,self.round)+i)
            for j in range(int(-1*width/2 + 1), int(width/2 + 1)):      
                yp = np.floor(round(y,self.round)+j)
                for k in range(int(-1*width/2 + 1), int(width/2 + 1)):          
                    zp = np.floor(round(z,self.round)+k)            
                    p = self.get_fms(xp, yp, zp)
                    #print(xp,yp,zp)
                    #vals.append(p)                    
                    arr[count] = p
                    count += 1
        #npvals = np.array(vals)    
        #return npvals
        return arr

    def build_cube_aroundx(self, x, y, z, width):        
        # 1. Build the points around the centre as a cube - width points        
        arr = np.zeros((width*width*width))        
        xp,yp,zp = 0,0,0         
        half = int(width/2)        
        count = 0
        for i in range(width):
            xp = np.floor(round(x,self.round))+i-half-1
            for j in range(width):
                yp = np.floor(round(y,self.round))+j-half-1
                for k in range(width):
                    zp = np.floor(round(z,self.round))+k-half-1
                    p = self.get_fms(xp, yp, zp)
                    arr[count] = p
                    count += 1
        return arr
    
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
    def get_value(self, x, y, z):
        u_x, u_y, u_z = self.get_adjusted_fms(x,y,z)
        xn,yn,zn,coeffs = self.make_coeffs(u_x,u_y,u_z)
        return self.get_value_multivariate(zn, yn, xn, coeffs)

    def get_radientx(self, x, y, z):
        return self.get_radient_numerical(x,y,z)
    
    def get_laplacianx(self, x, y, z):
        return self.get_laplacian_numerical(x,y,z)
    
    def get_radient(self, x, y, z):
        u_x, u_y, u_z = self.get_adjusted_fms(x,y,z)
        xn,yn,zn,coeffs = self.make_coeffs(u_x,u_y,u_z)
        dx = self.get_value_multivariate(zn, yn, xn, coeffs,["x"])
        dy = self.get_value_multivariate(zn, yn, xn, coeffs,["y"])
        dz = self.get_value_multivariate(zn, yn, xn, coeffs,["z"])
        return self.make_radient(dx,dy,dz)
            
    def get_laplacian(self, x, y, z):
        u_x, u_y, u_z = self.get_adjusted_fms(x,y,z)
        xn,yn,zn,coeffs = self.make_coeffs(u_x,u_y,u_z)
        ddx = self.get_value_multivariate(zn, yn, xn, coeffs,["x","x"])
        ddy = self.get_value_multivariate(zn, yn, xn, coeffs,["y","y"])
        ddz = self.get_value_multivariate(zn, yn, xn, coeffs,["z","z"])
        return self.make_laplacian(ddx,ddy,ddz)
        
    
    ## iplement abstract interface ###########################################
    
    """
    def get_value_old(self, x, y, z):
        # The method of linear interpolation is a version of my own method for multivariate fitting, instead of trilinear interpolation
        # NOTE I could extend this to be multivariate not linear but it has no advantage over bspline - and is slower and not as good 
        # Document is here: https://rachelalcraft.github.io/Papers/MultivariateInterpolation/MultivariateInterpolation.pdf                        
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
            self.xfloor = np.floor(x)
            self.yfloor = np.floor(y)
            self.zfloor = np.floor(z)
            # 1. Build the points around the centre as a cube - 8 points
            vals = self.build_cube_around(x, y, z, self.points)
            #2. Multiply with the precomputed matrix to find the multivariate polynomial
            ABC = self.mult_vector(self.inv.get_invariant(), vals)
            # 3. Put the 8 values back into a cube
            self.polyCoeffs = np.zeros((self.points, self.points, self.points))
            pos = 0;
            for i in range(self.points):
                for j in range(self.points):                
                    for k in range(self.points):                    
                        self.polyCoeffs[i, j, k] = ABC[pos]
                        pos+=1
            self.need_new = False        
        #4. Adjust the values to be within this cube
        pstart = (-1 * self.points / 2) + 1        
        xn = x - np.floor(x) - pstart
        yn = y - np.floor(y) - pstart
        zn = z - np.floor(z) - pstart

        #5. Apply the multivariate polynomial coefficents to find the value
        #return self.get_value_multivariate(zn, yn, xn, self.polyCoeffs)
        return self.get_value_multivariate(zn, yn, xn, self.polyCoeffs)
    """
        
    def get_value_multivariate(self, x, y, z, coeffs,wrt=[]):        
        #This is using a value scheme that makes sens of our new fitted polyCube
        #In a linear case it will be a decimal between 0 and 1          
        calc_mat = np.copy(coeffs)
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

    def diffWRTx(self,coeffs):        
        x,y,z = coeffs.shape
        partialX = np.zeros((x-1,y,z))
        for i in range(1, x):
            for j in range(0,y):
                for k in range(0,z):
                    partialX[i-1,j,k] = coeffs[i,j,k]*i
        return partialX

    def diffWRTy(self,coeffs):        
        x,y,z = coeffs.shape
        partialY = np.zeros((x,y-1,z))
        for i in range(0, x):
            for j in range(1,y):
                for k in range(0,z):
                    partialY[i,j-1,k] = coeffs[i,j,k]*j
        return partialY
    
    def diffWRTz(self,coeffs):        
        x,y,z = coeffs.shape
        partialZ = np.zeros((x,y,z-1))
        for i in range(0, x):
            for j in range(0,y):
                for k in range(1,z):
                    partialZ[i,j,k-1] = coeffs[i,j,k]*k
        return partialZ

    def make_coeffs(self, x, y, z):                                        
        # 1. Build the points around the centre as a cube - 8 points
        vals = self.build_cube_around(x, y, z, self.points)
        #2. Multiply with the precomputed matrix to find the multivariate polynomial
        polyCoeffs = self.mult_vector(self.inv.get_invariant(), vals)
        polyCoeffs = polyCoeffs.reshape((self.points,self.points,self.points))
        #ABC = self.mult_vector(self.inv.get_invariant(), vals)        
        # 3. Put the 8 values back into a cube                
        #polyCoeffs = np.zeros((self.points, self.points, self.points))
        #pos = 0
        #for i in range(self.points):
        #    for j in range(self.points):                
        #        for k in range(self.points):                    
        #            polyCoeffs[i, j, k] = ABC[pos]
        #            pos+=1        
        
        #4. Adjust the values to be within this cube        
        pstart = (-1 * self.points / 2) + 1        
        xn = round(x - np.floor(x) - pstart,self.round)
        yn = round(y - np.floor(y) - pstart,self.round)
        zn = round(z - np.floor(z) - pstart,self.round)
        if (xn > 1 or yn > 1 or zn > 1) and self.points==2:
            print(xn,yn,zn,x,y,z,self.points)
        elif xn < 0 or yn < 0 or zn < 0 and self.points==2:
            print(xn,yn,zn,x,y,z,self.points)
        elif (xn > 2 or yn > 2 or zn > 2) and self.points==4:
            print(xn,yn,zn,x,y,z,self.points)
        elif (xn < 1 or yn < 1 or zn < 1) and self.points==4:
            print(xn,yn,zn,x,y,z,self.points)
        
        return xn,yn,zn,polyCoeffs
    
            
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
        self.padded = True
        self.make_periodic_coeffs() 

                    
    ## implement abstract interface #########################################
    def get_radient(self, x, y, z):
        return self.get_radient_numerical(x,y,z)
    
    def get_laplacian(self, x, y, z):
        return self.get_laplacian_numerical(x,y,z)
    ## iplement abstract interface ###########################################

    def simple_copy_coeffs(self):      
        self._coeffs = np.copy(self._npy)

    def make_periodic_coeffs(self):
        # we make the coefficients matrix a bit bigger than the vaues and have it wrap, and then cut it back down to the values                         
        if not self.padded:
            self.simple_copy_coeffs()            
            self.create_coeffs_copy(self._F,self._M,self._S)                    
        else:
            self._coeffs =self.extend_vals_with_buffer(self.buffer)
            #print(self._coeffs)            
            self.create_coeffs_copy(self._F,self._M,self._S)        
            #print(self._coeffs)
            #self._coeffs = self.reduce_vals_with_buffer(self.buffer, self._coeffs)
        #print(self._npy)
        #print(self._coeffs)
                
    def reduce_vals_with_buffer(self,buffer, mynpy):        
        xx,yy,zz = mynpy.shape
        smnpy = mynpy[buffer:xx-buffer,buffer:yy-buffer,buffer:zz-buffer]
        return smnpy
        
    def extend_vals_with_buffer(self,buffer):        
        #// 1. Make a buffer padded values cube for periodic values        
        xxa,yya,zza = self._F, self._M, self._S        
        padded = np.pad(self._npy,[(buffer,buffer),(buffer,buffer),(buffer,buffer)])        
        # pad z
        for z in range(buffer):
            for y in range(-1*buffer, self._M+buffer):
                for x in range(-1*buffer, self._F+buffer):
                    p = self.get_fms(x-buffer,y-buffer,z-buffer)
                    padded[x,y,z] = p                            
        for z in range(zza,zza+buffer):
            for y in range(-1*buffer, self._M+buffer):
                for x in range(-1*buffer, self._F+buffer):
                    p = self.get_fms(x-buffer,y-buffer,z-buffer)                    
                    padded[x,y,z] = p                                            
        # pad y
        for y in range(buffer):
            for z in range(-1*buffer, self._S+buffer):
                for x in range(-1*buffer, self._F+buffer):
                    p = self.get_fms(x-buffer,y-buffer,z-buffer)
                    padded[x,y,z] = p                            
        for y in range(yya,yya+buffer):
            for z in range(-1*buffer, self._S+buffer):
                for x in range(-1*buffer, self._F+buffer):
                    p = self.get_fms(x-buffer,y-buffer,z-buffer)                    
                    padded[x,y,z] = p                                    
        # pad x
        for x in range(buffer):
            for z in range(-1*buffer, self._S+buffer):
                for y in range(-1*buffer, self._M+buffer):
                    p = self.get_fms(x-buffer,y-buffer,z-buffer)
                    padded[x,y,z] = p                            
        for x in range(xxa,xxa+buffer):
            for z in range(-1*buffer, self._S+buffer):
                for y in range(-1*buffer, self._M+buffer):
                    p = self.get_fms(x-buffer,y-buffer,z-buffer)                    
                    padded[x,y,z] = p                                    
        
        
        return padded
                                                                  
    def create_coeffs_inplace(self, thisF, thisM, thisS):
        pole = self.get_pole(self.degree)
        num_poles = len(pole)
        #Convert the samples to interpolation coefficients
        #X-wise
        for y in range(thisM):        
            for z in range(thisS):                        
                row = self._coeffs[:,y:y+1,z:z+1].reshape([thisF])
                self.convert_to_interp_coeffs_inplace(pole, num_poles, thisF, row)
        #Y-wise
        for x in range(thisF):        
            for z in range(thisS):                        
                row = self._coeffs[x:x+1,:,z:z+1].reshape([thisM])
                self.convert_to_interp_coeffs_inplace(pole, num_poles, thisM, row)
        #Z-wise
        for x in range(thisF):
            for y in range(thisM):                        
                row = self._coeffs[x:x+1,y:y+1,:].reshape([thisS])
                line = self.convert_to_interp_coeffs_inplace(pole, num_poles, thisS, row)

    def create_coeffs_copy(self, thisF, thisM, thisS):
        pole = self.get_pole(self.degree)
        num_poles = len(pole)
        #Convert the samples to interpolation coefficients
        #X-wise
        for y in range(thisM):        
            for z in range(thisS):        
                row = self.get_row3d(y, z, thisF,thisF, thisM, thisS)
                line = self.convert_to_interp_coeffs(pole, num_poles, thisF, row)
                self.put_row3d(y, z, line, thisF,thisF, thisM, thisS)                
        #Y-wise
        for x in range(thisF):        
            for z in range(thisS):        
                row = self.get_col3d(x, z, thisM,thisF, thisM, thisS)
                line = self.convert_to_interp_coeffs(pole, num_poles, thisM, row)
                self.put_col3d(x, z, line, thisM,thisF, thisM, thisS)                           
        #Z-wise
        for x in range(thisF):
            for y in range(thisM):        
                row = self.get_hole3d(x, y, thisS,thisF, thisM, thisS)
                line = self.convert_to_interp_coeffs(pole, num_poles, thisS, row)
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
    
    def convert_to_interp_coeffs(self,pole, num_poles, width, row):        
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

    def convert_to_interp_coeffs_inplace(self,pole, num_poles, width, row):        
        #/* This version is periodic ONLY */
        if (width == 1):#not much can be done if it is only 1 think            
            return row        
        lmbda = 1
        #Compute the overall gain
        for k in range(num_poles):
            lmbda = lmbda * (1 - pole[k]) * (1 - 1 / pole[k])        
        #Apply the gain
        for n in range(width):
            row[n] = row[n] * lmbda        
        #loop over the poles            
        for k in range(num_poles):                                
            #/* causal recursion */            
            for n in range(0,width):
                m = n
                if m < 0:
                    m = -1*width%abs(n)
                n1 = m-1                
                row[m] = row[m]+ pole[k] * row[n1]                                    
            
            #/* anticausal recursion */            
            for n in range(width-1, -1, -1):
                n1 = n+1
                if n1 >= width:
                    n1 = n1%width
                m = n1-1
                row[m] = pole[k] * (row[n1] - row[m])
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
            #// RSA notes is this a mirror condition? when the horizon - how far you need to look ahead for good data - is not as far the data you have
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
        u_x, u_y, u_z = x,y,z        
        if self.padded:
            u_x, u_y, u_z = u_x+self.buffer,u_y+self.buffer,u_z+self.buffer        
        u_x, u_y, u_z = self.get_adjusted_fms(u_x, u_y, u_z)
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
        i = int(np.floor(round(u_x,self.round)) - np.floor(self.degree / 2))
        j = int(np.floor(round(u_y,self.round)) - np.floor(self.degree / 2))
        k = int(np.floor(round(u_z,self.round)) - np.floor(self.degree / 2))

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
                    xc,yc,zc = xIndex[i],yIndex[j],zIndex[k]
                    xc,yc,zc = self.get_adjusted_fms(xc,yc,zc)
                    cc = self.get_coeff(xc,yc,zc,self._F,self._M,self._S)
                    #print(xIndex[i],yIndex[j],zIndex[k])
                    #print(xc,yc,zc,cc)                    
                    w1 += xWeight[i] * cc
                w2 += yWeight[j] * w1
            w3 += zWeight[k] * w2
        return w3
    
    def applyValue3(self,val, idc, weight_length):        
        ws = []
        for i in range(weight_length):
            ws.append(0)
        w = val - idc[1]
        if w > 1:
            print("w=",w)
        if w < 0:
            print("w=",w)
        # Making it linear
        #ws[3] = 1*w/40
        #ws[0] = 1*(1-w) / 40
        #ws[2] = 39*w/40
        #ws[1] = 39*(1-w) / 40        
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
        
####################################################################################################
### R-Spline
#################################################################################################### 
class Mspline(Bspline):
    def init(self):                
        #self.make_periodic_coeffs(True) 
        #self.make_periodic_coeffs(False) 
        self.padded = False
        self.make_periodic_coeffs()
        #self.simple_copy_coeffs() #temp make it just ordinary coeffs
    
    

    
            
        
        
        
        
        
        
        
        
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