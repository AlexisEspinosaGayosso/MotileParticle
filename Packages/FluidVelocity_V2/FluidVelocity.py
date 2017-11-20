"""
Basic module for fluid velocity tools
"""
#Importing necessary 'typical' modules
import os #For interacting with operating system
import numpy as np #Fundamental package for scientific computing
from scipy.integrate import ode #Fundamental package for numerical integration of ordinary differenttial equations
from scipy.interpolate import RectBivariateSpline #Function for interpolating arrays
from scipy.interpolate import interp2d #Function for interpolating arrays

#Adding the root path for our own modules
from os import path
import sys
sys.path.append(path.abspath('../../../../Python')) #IMPORTANT. Put as many .. to reach the level of the Python directory and then we can use the directory Packages below for importing our modules and packages

#Adding our own modules
#None.

class analyticalToyCylinderFlow(object):
    """
    This object hosts the analytical expressions of fields
    """
    def __init__(self):        
        self.xLimits=[-10.0, 10.0]
        self.yLimits=[-10.0, 10.0]
        self.fieldType='analyticalToyCylinder'
        
                
    #Defining the potential flow theory velocity field
    def velocityField(self, X):
        """
        Analytical flow around a cylinder
        Dc is cylinder diameter
        Re is the Reynolds number
        Returns a vector with local velocity
        """
        #The cylindrical parameters:
        rr=np.sqrt(np.power(X[0], 2)+np.power(X[1], 2))
        theta=np.arctan2(X[1], X[0])
        #Potential flow part
        Dc=1.0 #The cylinder diameter is forced to be 1.0
        Uinf=1.0
        Rc=Dc/2.0
        VrF=Uinf*(1-np.power(Rc, 2)/np.power(rr, 2))*np.cos(theta) #Radial component
        VtF=-Uinf*(1+np.power(Rc, 2)/np.power(rr, 2))*np.sin(theta) #Tangential component
        #Modified flow part
        ReyM=0.1
        epsilonM=0.5*ReyM
        deltaM=1/(np.log(4)-0.5772-np.log(epsilonM)+0.5)
        VrFM=deltaM*(np.log(rr)-0.5+0.5/(np.power(rr, 2)))*np.cos(theta)
        VtFM=-deltaM*(np.log(rr)+0.5-0.5/(np.power(rr, 2)))*np.sin(theta)
        #Final velocity field 
        VrF=VrF+VrFM
        VtF=VtF+VtFM
        #Conversion to cartesian coordinates
        VxF=VrF*np.cos(theta)-VtF*np.sin(theta)
        VyF=VrF*np.sin(theta)+VtF*np.cos(theta)
        return np.array([VxF,  VyF,  0], dtype=np.dtype('d'))
    #End for analytical flow velocity

    #Defining the potential flow theory velocity field
    def gradientField(self, X):
        """
        Analytical Gradient of flow around a cylinder
        Dc is cylinder diameter
        Re is the Reynolds number
        Returns a matrix with local velocity gradient
        """
        #The cylindrical parameters:
        rr=np.sqrt(np.power(X[0], 2)+np.power(X[1], 2))
        theta=np.arctan2(X[1], X[0])
        #Potential flow part
        Dc=1.0 #The cylinder diameter is forced to be 1.0
        Uinf=1.0
        Rc=Dc/2.0
        #OJO, this needs to be properly defined.
        #OJO, this is just a test
        #return np.array([[0,   3*X[1],  0], [3*X[0],   0,  0], [0,   0,  0]], dtype=np.dtype('d'))
        return np.array([[0,   3*X[1],  0], [0,   0,  0], [0,   0,  0]], dtype=np.dtype('d'))
    #End for analytical flow gradient
    
class simpleShearFlow(object):
    def __init__(self):        
        self.xLimits=[-100.0, 100.0]
        self.yLimits=[-100.0, 100.0]
        self.fieldType='simpleShearFlow'
    #Defining the simple shear Flow
    def velocityField(self, X): #As in Hinch1979
        """
        Analytical flow , simple Shear. (Hinch1979)
        """

        #The simple shear flow
        VxF=X[1]
        VyF=0
        return np.array([VxF,  VyF,  0], dtype=np.dtype('d'))
    #End for analytical flow velocity

    #Defining the potential flow theory velocity field
    def gradientField(self, X): #As in Hinch 1979
        """
        Analytical Gradient , simple Shear. (Hinch1979)
        """
        #The simple shear flow
        S=1
        return np.array([[0,  S,  0], [0,  0,  0], [0,  0,  0]], dtype=np.dtype('d'))
    #End for analytical  gradient
    
class zeroFlow(object):
    def __init__(self):        
        self.xLimits=[-100.0, 100.0]
        self.yLimits=[-100.0, 100.0]
        self.fieldType='zeroFlow'
    #Defining the simple shear Flow
    def velocityField(self, X): 
        """
        Zero Flow
        """

        #Zero Flow
        VxF=0
        VyF=0
        return np.array([VxF,  VyF,  0], dtype=np.dtype('d'))
    #End for analytical flow velocity

    #Defining the potential flow theory velocity field
    def gradientField(self, X):
        """
        Zero Flow
        """
        return np.array([[0,  0,  0], [0,  0,  0], [0,  0,  0]], dtype=np.dtype('d'))
    #End for analytical  gradient 

class creepingCylinderFlow(object):
    """
    Creeping flow theory of Skinner1975
    """
    #The constructor
    def __init__(self, Reynolds):
        if Reynolds is None:
            self.Reynolds=0
        else:
            self.Reynolds=Reynolds
            
        #Some basic parametes needed for Skinner calculations
        self.eps=0.5*self.Reynolds
        self.delta=1.0/(np.log(4)-0.5772-np.log(self.eps)+0.5)
        self.a=self.delta-0.8669*self.delta**3
        self.b=-0.5+self.delta/4
        self.xLimits=[-3.1/self.Reynolds, 3.1/self.Reynolds]
        self.yLimits=[-3.1/self.Reynolds, 3.1/self.Reynolds]
        
        self.fieldType='creepingCylinderFlow'
    #End of the constructor
    #Defining the simple shear Flow    
    
    def velocityField(self, X):
        """
        Analytical flow defined by Skinner1975 as in Espinosa2012
        Returns a vector with local velocity
        """                
        
        #Rescaling the dimensions with the collector diameter
        dR_dr=0.5
        x=X/dR_dr
        
        #The cylindrical parameters:
        rr=np.sqrt(np.power(x[0], 2)+np.power(x[1], 2))
        theta=np.arctan2(x[1], x[0])
        #Skinner expressions for velocity in cartesian coordinates
        VxF=(
                   32*self.a*rr*np.cos(2*theta)
                   -self.eps*self.a**2*np.cos(3*theta)
                   +64*self.a*rr**3*np.log(rr)
                   -32*self.a*rr**3*np.cos(2*theta)
                   +16*self.eps*self.b*np.cos(3*theta)
                   +self.eps*self.a**2*rr**4*np.cos(3*theta)
                   -16*self.eps*self.b*rr**2*np.cos(theta)
                   +16*self.eps*self.b*rr**4*np.cos(theta)
                   -16*self.eps*self.b*rr**2*np.cos(3*theta)
                   -4*self.eps*self.a**2*rr**4*np.cos(3*theta)*np.log(rr)
                   +8*self.eps*self.a**2*rr**4*np.cos(theta)*(np.log(rr))**2) /(64*rr**3)
                  
        VyF=-(
                   self.eps*self.a**2*np.sin(3*theta)
                   -32*self.a*rr*np.sin(2*theta)
                   -16*self.eps*self.b*np.sin(3*theta)
                   +32*self.a*rr**3*np.sin(2*theta)
                   -self.eps*self.a**2*rr**4*np.sin(3*theta)
                   -16*self.eps*self.b*rr**2*np.sin(theta)
                   +16*self.eps*self.b*rr**4*np.sin(theta)
                   +16*self.eps*self.b*rr**2*np.sin(3*theta)
                   +4*self.eps*self.a**2*rr**4*np.sin(3*theta)*np.log(rr)
                   +8*self.eps*self.a**2*rr**4*np.sin(theta)*(np.log(rr))**2)/(64*rr**3)
                   
        return np.array([VxF,  VyF,  0], dtype=np.dtype('d'))
    #End for analytical flow velocity

    #Defining the potential flow theory velocity field
    def gradientField(self, X):
        """
        Analytical Gradient defined by Skinner1975 as in Espinosa2012
        Returns a vector with local velocity
        """
        #
        
        #Rescaling the dimensions with the collector diameter
        dR_dr=0.5
        x=X/dR_dr
        
        #The cylindrical parameters:
        rr=np.sqrt(np.power(x[0], 2)+np.power(x[1], 2))
        theta=np.arctan2(x[1], x[0])
        #Skinner expressions for velocity in cartesian coordinates
        Gcart11=(
                         32*self.a*rr**3*np.cos(theta)
                         -64*self.a*rr*np.cos(3*theta)
                         +3*self.eps*self.a**2*np.cos(4*theta)
                         +32*self.a*rr**3*np.cos(3*theta)
                         +16*self.eps*self.b*rr**4
                         -48*self.eps*self.b*np.cos(4*theta)
                         -3*self.eps*self.a**2*rr**4*np.cos(4*theta)
                         +8*self.eps*self.a**2*rr**4*(np.log(rr))**2
                         +32*self.eps*self.b*rr**2*np.cos(4*theta)
                         +8*self.eps*self.a**2*rr**4*np.log(rr)
                         +4*self.eps*self.a**2*rr**4*np.cos(4*theta)*np.log(rr)
                         )/(64*rr**4)
                         
        Gcart12=(
                         24*self.a*rr**3*np.sin(theta)
                         -16*self.a*rr*np.sin(3*theta)
                         -12*self.eps*self.b*np.sin(4*theta)
                         +(3*self.eps*self.a**2*np.sin(4*theta))/4
                         +8*self.a*rr**3*np.sin(3*theta)
                         -(3*self.eps*self.a**2*rr**4*np.sin(4*theta))/4
                         +8*self.eps*self.b*rr**2*np.sin(2*theta)
                         +8*self.eps*self.b*rr**2*np.sin(4*theta)
                         +4*self.eps*self.a**2*rr**4*np.sin(2*theta)*np.log(rr)
                         +self.eps*self.a**2*rr**4*np.sin(4*theta)*np.log(rr)
                         )/(64*rr**4)
                         
                     
        Gcart21=-(
                          48*self.eps*self.b*np.sin(4*theta)
                          +64*self.a*rr*np.sin(3*theta)
                          +32*self.a*rr**3*np.sin(theta)
                          -3*self.eps*self.a**2*np.sin(4*theta)
                          -32*self.a*rr**3*np.sin(3*theta)
                          +3*self.eps*self.a**2*rr**4*np.sin(4*theta)
                          +32*self.eps*self.b*rr**2*np.sin(2*theta)
                          -32*self.eps*self.b*rr**2*np.sin(4*theta)
                          +16*self.eps*self.a**2*rr**4*np.sin(2*theta)*np.log(rr)
                          -4*self.eps*self.a**2*rr**4*np.sin(4*theta)*np.log(rr)
                          )/(64*rr**4)
                        
                          
                          
        Gcart22=-(
                          -32*self.a*rr**3*np.cos(theta)
                          +64*self.a*rr*np.cos(3*theta)
                          +3*self.eps*self.a**2*np.cos(4*theta)
                          +32*self.a*rr**3*np.cos(3*theta)
                          +16*self.eps*self.b*rr**4
                          -48*self.eps*self.b*np.cos(4*theta)
                          -3*self.eps*self.a**2*rr**4*np.cos(4*theta)
                          +8*self.eps*self.a**2*rr**4*(np.log(rr))**2
                          +32*self.eps*self.b*rr**2*np.cos(4*theta)
                          +8*self.eps*self.a**2*rr**4*np.log(rr)
                          +4*self.eps*self.a**2*rr**4*np.cos(4*theta)*np.log(rr)
                          )/(64*rr**4)
                          
                              
        return np.array([[Gcart11/dR_dr,   Gcart12/dR_dr,  0], [Gcart21/dR_dr,   Gcart22/dR_dr,  0], [0,   0,  0]], dtype=np.dtype('d'))
    #End for analytical flow gradient

class OpenFOAMCylinderFlow(object):
    """
    Object that will contain the sampled fields from an OpenFOAM sampled solution    
    """
    #The constructor of the class
    def  __init__(self, 
                        OpenFOAMDir=None, 
                        Reynolds=None, 
                        dR_dr=None):
        """
        Constructor of the class
        dR_dr When dR_dr=0.5: this is a factor needed to rescale the coordinates for the nondimensional r to be r=1 at the surface in cylindrical coordinates
              When dR_dr=1 (default): No factor is applied and r=0.5 at the surface of the collector
        """
        if OpenFOAMDir is None:
            self.OpenFOAMDir=os.getcwd()            
        else:
            self.OpenFOAMDir=OpenFOAMDir
        if Reynolds is None:
            self.Reynolds=0
        else:
            self.Reynolds=Reynolds
        if dR_dr is None:
            self.dR_dr=1.0
        else:
            self.dR_dr=dR_dr
            
        self.fieldType='OpenFOAMCylinderFlow'
            
        #Reading the OpenFOAM sampling
        self.readAndFillMatrices() #Calling the reader of the sampled fields
        
        #Creating the interpolation functions using the spline interpolator (Not using because of overshoots)
##        self.uxf=RectBivariateSpline(self.X, self.Y, self.UxDNS)
##        self.uyf=RectBivariateSpline(self.X, self.Y, self.UyDNS)
##        self.dudxf=RectBivariateSpline(self.X, self.Y, self.dudxDNS)
##        self.dudyf=RectBivariateSpline(self.X, self.Y, self.dudyDNS)
##        self.dvdxf=RectBivariateSpline(self.X, self.Y, self.dvdxDNS)
##        self.dvdyf=RectBivariateSpline(self.X, self.Y, self.dvdyDNS)
        #Creating the interpolation functions using the interp2d interpolator
        #kindHere='cubic' #(Not using because of overshoots)
        kindHere='linear'
        self.uxf=interp2d(self.X, self.Y, np.transpose(self.UxDNS), kind=kindHere)
        self.uyf=interp2d(self.X, self.Y, np.transpose(self.UyDNS), kind=kindHere)
        self.dudxf=interp2d(self.X, self.Y, np.transpose(self.dudxDNS), kind=kindHere)
        self.dudyf=interp2d(self.X, self.Y, np.transpose(self.dudyDNS), kind=kindHere)
        self.dvdxf=interp2d(self.X, self.Y, np.transpose(self.dvdxDNS), kind=kindHere)
        self.dvdyf=interp2d(self.X, self.Y, np.transpose(self.dvdyDNS), kind=kindHere)

    #End of constructor

    def estimateSampleSizeWithDX(self):
        """
        This looks weird, but the size is estimated with the same process to define the sample points, all this is the in the matlab scripts
        """
        RCyl=0.5
        if (self.Reynolds>46.999):
            #The sampling spacing
            dxfine=0.005
            dxcoarse=0.05
            dxsupercoarse=0.5
            #The limits between sampling
            xlimfine=[-0.65, 0.65]
            xlimcoarse=[-10, 10]
            xlimsupercoarse=[-100, 12]
            ylimfine=[-0.65, 0.65]
            ylimcoarse=[-10, 10]
            ylimsupercoarse=[-12, 12]
            toleSurface=1E-5
            #Supercoarse in -x
            x=np.arange(xlimsupercoarse[0], xlimcoarse[0]-dxsupercoarse/2, dxsupercoarse) #Avoiding last point
            y=np.arange(ylimsupercoarse[0], ylimcoarse[0]-dxsupercoarse/2, dxsupercoarse) #Avoiding last point
            XA=x
            YA=y
            #Coarse in -x
            x=np.arange(xlimcoarse[0], xlimfine[0]-dxcoarse/2, dxcoarse) #Avoiding last point
            y=np.arange(ylimcoarse[0], ylimfine[0]-dxcoarse/2, dxcoarse) #Avoiding last point
            XB=np.concatenate((XA, x), axis=0)
            YB=np.concatenate((YA, y), axis=0)
            del XA
            del YA
            #Fine in x
            x=np.arange(xlimfine[0], xlimfine[1]-dxfine/2, dxfine) #Avoiding last point
            y=np.arange(ylimfine[0], ylimfine[1]-dxfine/2, dxfine) #Avoiding last point
            XC=np.concatenate((XB, x), axis=0)
            YC=np.concatenate((YB, y), axis=0)
            del XB
            del YB
            #Coarse in x
            x=np.arange(xlimfine[1], xlimcoarse[1]-dxcoarse/2, dxcoarse) #Avoiding last point
            y=np.arange(ylimfine[1], ylimcoarse[1]-dxcoarse/2, dxcoarse) #Avoiding last point
            XD=np.concatenate((XC, x), axis=0)
            YD=np.concatenate((YC, y), axis=0)
            del XC
            del YC
            #Supercoarse in x
            x=np.arange(xlimcoarse[1], xlimsupercoarse[1]+dxsupercoarse/2, dxsupercoarse) #Including last point
            y=np.arange(ylimcoarse[1], ylimsupercoarse[1]+dxsupercoarse/2, dxsupercoarse) #Including last point
            self.X=np.concatenate((XD, x), axis=0)
            self.Y=np.concatenate((YD, y), axis=0)
            del XD
            del YD
        elif(self.Reynolds==1 or self.Reynolds==10): #Not working properly yet because my samples for Re=1,10 have some points missing
            #The sampling spacing
            dxfine=0.005
            dxcoarse=0.05
            dxsupercoarse=0.5
            #The limits between sampling
            xlimfine=[-0.65, 0.65]
            xlimcoarse=[-10, 0.8]
            xlimsupercoarse=[-50, 0.8]
            ylimfine=[-0.65, 0.65]
            ylimcoarse=[-10, 10]
            ylimsupercoarse=[-12, 12]
            toleSurface=1E-5
            #Supercoarse in -x
            x=np.arange(xlimsupercoarse[0], xlimcoarse[0]-dxsupercoarse/2, dxsupercoarse) #Avoiding last point
            y=np.arange(ylimsupercoarse[0], ylimcoarse[0]-dxsupercoarse/2, dxsupercoarse) #Avoiding last point
            XA=x
            YA=y
            #Coarse in -x
            x=np.arange(xlimcoarse[0], xlimfine[0]-dxcoarse/2, dxcoarse) #Avoiding last point
            y=np.arange(ylimcoarse[0], ylimfine[0]-dxcoarse/2, dxcoarse) #Avoiding last point
            XB=np.concatenate((XA, x), axis=0)
            YB=np.concatenate((YA, y), axis=0)
            del XA
            del YA
            #Fine in x
            x=np.arange(xlimfine[0], xlimfine[1]-dxfine/2, dxfine) #Avoiding last point
            y=np.arange(ylimfine[0], ylimfine[1]-dxfine/2, dxfine) #Avoiding last point
            XC=np.concatenate((XB, x), axis=0)
            YC=np.concatenate((YB, y), axis=0)
            del XB
            del YB
            #Coarse in x
            x=np.arange(xlimfine[1], xlimcoarse[1]-dxcoarse/2, dxcoarse) #Avoiding last point
            y=np.arange(ylimfine[1], ylimcoarse[1]-dxcoarse/2, dxcoarse) #Avoiding last point
            XD=np.concatenate((XC, x), axis=0)
            YD=np.concatenate((YC, y), axis=0)
            del XC
            del YC
            #Supercoarse in x
            x=np.arange(xlimcoarse[1], xlimsupercoarse[1]+dxsupercoarse/2, dxsupercoarse) #Including last point
            y=np.arange(ylimcoarse[1], ylimsupercoarse[1]+dxsupercoarse/2, dxsupercoarse) #Including last point
            self.X=np.concatenate((XD, x), axis=0)
            self.Y=np.concatenate((YD, y), axis=0)
            del XD
            del YD
        #Final sizes
        self.Nxfin=np.size(self.X)
        self.Nyfin=np.size(self.Y)
        return

    def estimateSampleSizeWithM(self, MU):
        """
        The size is defined with MU
        """
        #The official increments in sampling
        dxfine=0.005
        dxcoarse=0.05
        dxsupercoarse=0.5
        #Defining from MU
        tamanoM=np.shape(MU)
        i=0
        j=0
        xHere=np.array([MU[0, 0]], dtype=np.float)
        yHere=np.array([MU[0, 1]], dtype=np.float)
        i+=1
        j+=1
        for k in range(0, tamanoM[0]):
            if (MU[k, 1] >= yHere[j-1] + dxfine/2) :
                yHere=np.concatenate((yHere, [MU[k, 1]]), axis=0)
                j+=1
            if (MU[k, 0]>=xHere[i-1]+dxfine/2) :
                xHere=np.concatenate((xHere, [MU[k, 0]]), axis=0)
                i+=1
        #End of the cycle for reading the coordinates
        self.X=xHere
        self.Y=yHere
        #Final sizes
        self.Nxfin=np.size(self.X)
        self.Nyfin=np.size(self.Y)
        return
 
    def createArrays(self):
        self.EMPTY_FLAG=1234567.0
        self.UxDNS=np.ones((self.Nxfin, self.Nyfin), dtype=np.float)*self.EMPTY_FLAG
        self.UyDNS=np.ones((self.Nxfin, self.Nyfin), dtype=np.float)*self.EMPTY_FLAG
        self.dudxDNS=np.ones((self.Nxfin, self.Nyfin), dtype=np.float)*self.EMPTY_FLAG
        self.dudyDNS=np.ones((self.Nxfin, self.Nyfin), dtype=np.float)*self.EMPTY_FLAG
        self.dvdxDNS=np.ones((self.Nxfin, self.Nyfin), dtype=np.float)*self.EMPTY_FLAG
        self.dvdyDNS=np.ones((self.Nxfin, self.Nyfin), dtype=np.float)*self.EMPTY_FLAG
        
        #The radius and angle arrays
        XM, YM = np.meshgrid(self.X, self.Y, sparse=False, indexing='ij')
        self.RDNS=np.sqrt(np.power(XM, 2)+np.power(YM, 2))
        self.ThetaDNS=np.arctan2(YM, XM)

    #Function for reading coordinates,velocity field from an OpenFOAM sample file
    def readAndFillMatrices(self):
        """
        Reads the coordinates and velocity matrix
        """
        caseDir=os.path.join(self.OpenFOAMDir, 'Re_'+str(self.Reynolds))
        fileDir=os.path.join(caseDir,'sets','20000')
        if (self.Reynolds<=10):
            self.xLimits=[-50.0, 0.8]
            self.yLimits=[-11.5, 11.5]
            UFile='StreamLine_'+'cellPoint_UShortMean_CorrectedNoY12.xy'
            GFile='StreamLine_'+'cellPoint_uxdx_uxdy_uydx_uydy_CorrectedNoY12.xy'
        else:
            self.xLimits=[-100.0, 12.0]
            self.yLimits=[-12.0, 12.0]
            UFile='StreamLine_'+'cellPoint_URealMean.xy'
            GFile='StreamLine_'+'cellPoint_uxdx_uxdy_uydx_uydy.xy'        
        UFile=os.path.join(fileDir, UFile)
        GFile=os.path.join(fileDir, GFile)
        
        #The official increments in sampling
        dxfine=0.005
        dxcoarse=0.05
        dxsupercoarse=0.5
        
        #Some parameters
        xcCyl=0.0
        ycCyl=0.0
        RCyl=0.5
        
        #Sampling coordinates
        #I'm not using the factr dR_dr because we are going to work with collectors of NonDimensional Diameter=1, as in OpenFOAM fields
        MU=np.loadtxt(UFile)
        MG=np.loadtxt(GFile)
        MU[:, 0]-=-xcCyl
        MU[:, 1]-=ycCyl
       
        #Size of the OpenFOAM matrix data
        self.MUshape=np.shape(MU)
        self.MGshape=np.shape(MG)
        
        #Generating the matrices that will keep the data ordered for interpolation
        self.estimateSampleSizeWithM(MU) #Defining the sample points directly from M
        #self.estimateSampleSizeWithDX()  #Defining the sample points from parameters (BUT This one is failing as my sample points are a mess for Re=1,10). This was the method I was using with Matlab before

        #Create and initialize the arrays that will keep the data for the interpolation
        self.createArrays()
        
        #Now adding the values into the matrices (Filling just half of the data in Y, later will be reflected)
        eps=dxfine/2        
        for k in range(0, self.MUshape[0]):
            #Finding the index in the matrix
            i=np.searchsorted(self.X, MU[k, 0]-eps)
            j=np.searchsorted(self.Y, MU[k, 1]-eps)            
            self.UxDNS[i, j]=MU[k, 3]
            self.UyDNS[i, j]=MU[k, 4]
            self.dudxDNS[i, j]=MG[k, 3]
            self.dudyDNS[i, j]=MG[k, 5] #The gradient fields are messed up and not in the expected order, so using MG[k, 5] instead of MG[k, 4]
            self.dvdxDNS[i, j]=MG[k, 4] #The gradient fields are messed up and not in the expected order, so using MG[k, 4] instead of MG[k, 5]
            self.dvdyDNS[i, j]=MG[k, 6]
        #End for k
       
        #Anihilating the values inside the cylinder and its surface
        toleSurface=1E-5
        InteriorX,  InteriorY=np.where(self.RDNS<=RCyl+toleSurface)
        self.UxDNS[InteriorX[:], InteriorY[:]]=0.0
        
        #Reflecting the data in Y
        Nycentre=int(self.Nyfin//2)+1-1 #Filling just half of the data in Y, later will be reflected (-1 due to the 0 indexing in Python)
        self.Y[Nycentre::-1]=-self.Y[Nycentre::1]
        self.UxDNS[:, Nycentre::-1]=self.UxDNS[:, Nycentre::1]
        self.UyDNS[:, Nycentre::-1]=-self.UyDNS[:, Nycentre::1]
        self.dudxDNS[:, Nycentre::-1]=self.dudxDNS[:, Nycentre::1]
        self.dudyDNS[:, Nycentre::-1]=-self.dudyDNS[:, Nycentre::1]
        self.dvdxDNS[:, Nycentre::-1]=-self.dvdxDNS[:, Nycentre::1]
        self.dvdyDNS[:, Nycentre::-1]=self.dvdyDNS[:, Nycentre::1]
        
        #Setting a zero Y velocity for the symmetry line
        self.Y[Nycentre]=0.0
        self.UyDNS[:, Nycentre]=0.0
        self.dudyDNS[:, Nycentre]=0.0
        self.dvdyDNS[:, Nycentre]=0.0
        
        #Estimating the non filled values
        nonFilledX,  nonFilledY=np.where(self.UxDNS==self.EMPTY_FLAG)
        nonFilledX=np.array(nonFilledX, ndmin=2, dtype=np.int)
        nonFilledY=np.array(nonFilledY, ndmin=2, dtype=np.int)
        self.nonFilled=np.concatenate((np.transpose(nonFilledX), np.transpose(nonFilledY)), axis=1)
        
                        
        return
        
    def velocityField(self, X):
        """
        OpenFOAM interpolated values
        """                
        
        #Interpolating the velocities
        VxF=self.uxf(X[0], X[1])
        VyF=self.uyf(X[0], X[1])

                   
        return np.array([VxF,  VyF,  0], dtype=np.dtype('d'))
    #End for analytical flow velocity

    #Defining the potential flow theory velocity field
    def gradientField(self, X):
        """
        OpenFOAM interpolated values
        """
        Gcart11=self.dudxf(X[0], X[1])
        Gcart12=self.dudyf(X[0], X[1])
        Gcart21=self.dvdxf(X[0], X[1])
        Gcart22=self.dvdyf(X[0], X[1])
                              
        return np.array([[Gcart11,   Gcart12,  0], [Gcart21,   Gcart22,  0], [0,   0,  0]], dtype=np.dtype('d'))
    #End for analytical flow gradient
    
#End of the OpenFOAM class
    
