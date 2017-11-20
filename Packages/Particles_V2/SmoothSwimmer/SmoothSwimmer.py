"""
Module for smooth swimmers
"""

#Importing necessary 'typical' modules
import os #For interacting with operating system
import numpy as np #Fundamental package for scientific computing
from scipy.integrate import ode #Fundamental package for numerical integration of ordinary differenttial equations

#Adding the root path for our own modules
from os import path
import sys
sys.path.append(path.abspath('../../../../Python')) #IMPORTANT. Put as many .. to reach the level of the Python directory and then we can use the directory Packages below for importing our modules and packages

#Adding our own modules
#import Packages.FluidVelocity_V2.FluidVelocity as FV
import Packages.Particles_V2.Basic.ParticleBasic as PB

#Defining the class particleBasic
class smoothSwimmer(PB.particleBasicBrownianRotation):
    """
    Derived class for particles with smooth swimming capabilities
    """
    #Constructor of the class
    def __init__(self,
                        time=None, 
                        position=None,
                        velocity=None,
                        semiDiameters=None, 
                        eulerAngles=None,
                        eulerAngularVel=None,
                        integratorType=None, 
                        willFreeze=None, 
                        readFluidVelocity=None, 
                        readFluidGradient=None, 
                        xLimits=None, 
                        yLimits=None, 
                        rotationalDiff=None,
                        randomSeed=None,
                        swimmingSpeed=None):
        #Internal properties  with default values
               
        PB.particleBasicBrownianRotation.__init__(self, time=time, position=position, velocity=velocity, semiDiameters=semiDiameters,  eulerAngles=eulerAngles,  
                                               eulerAngularVel=eulerAngularVel, integratorType=integratorType, willFreeze=willFreeze, readFluidVelocity=readFluidVelocity, readFluidGradient=readFluidGradient, 
                                               xLimits=xLimits, yLimits=yLimits, rotationalDiff=rotationalDiff, randomSeed=randomSeed) #Calling the constructor of the particleBasicBrownianRotation class
        if swimmingSpeed is None:
            self.swimmingSpeed=0.0
        else:
            self.swimmingSpeed=swimmingSpeed #Swimming speed in the direction of the orientation
    #End of the constructor of the smoothSwimmer class
       
    #Function for defining the perfect tracer behaviour dy/dt = f(t, y)
    def dYdt_smoothSwimmer(self, t,  Y):
        """
        Computes the derivative dYdT for a smooth swimmer
        """
        positionHere=np.array([Y[0],  Y[1],  Y[2]], dtype=np.dtype('d'))
        eulerAnglesHere=np.array([Y[6],  Y[7],  Y[8]], dtype=np.dtype('d'))                
        RHere=self.setRotationMatrix(eulerAnglesHere) #The rotation Matrix Here
        orientationBodyHere=np.matrix([[1], [0], [0]],  dtype=np.dtype('d')) #Orientation vector in the body coordinate system
        orientationWorldHere=np.matmul(RHere.transpose(), orientationBodyHere) #Orientation vector in the world coordinate system
        #print('Shape of self.orientationWorld is', np.shape(self.orientationWorld))
        orientationHere=np.array(np.array(orientationWorldHere.transpose()).flatten(),  dtype=np.dtype('d'))    #World Orientation vector flattened & saved in a stack of orientations
        #print('Shape of self.orientation is', np.shape(self.orientation))
        #print(self.orientation)
        velocityHere=self.swimmingSpeed*orientationHere
        accelerationHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        eulerAngularVelHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        eulerAngularAccHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        return np.concatenate((velocityHere,  accelerationHere, eulerAngularVelHere, eulerAngularAccHere))
    #End dYdT_perfectTracer

    #Function for defining the system dy/dt = f(t, y)
    def dYdt_system(self, t,  Y,  dt):
        """
        Defines the derivative function to use
        """
        #dYdtHere=self.dYdt_perfectTracer(t, Y)
        dYdtHere=self.dYdt_perfectRotatingTracer(t, Y)+self.dYdt_BrownianRotation(t, Y, dt)+self.dYdt_smoothSwimmer(t, Y)
        return dYdtHere
    #End dYdT_system        
    
#End of the derived class smoothSwimmer definition
