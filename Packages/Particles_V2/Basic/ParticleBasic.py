"""
Module for basic particles
"""

#Importing necessary 'typical' modules
import os #For interacting with operating system
import numpy as np #Fundamental package for scientific computing
from scipy.integrate import ode #Fundamental package for numerical integration of ordinary differenttial equations

#Adding the root path for our own modules
from os import path
import sys
sys.path.append(path.abspath('../../../../Python')) #IMPORTANT. Put as many .. to reach the level of the Python directory and then we can use the directory Packages below for importing our modules and packages

#Defining the class particleBasic
class particleBasic(object):
    """
    Basic class for particles with elliptical shape
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
                        yLimits=None):
        #Internal properties  with default values
               
        if semiDiameters is None: #Semidiamters [a,b,c] that define the ellipsoid in the local coordinates system x**2/a**2+y**2/b**2+z**2/c**2=1
            self.semiDiameters=np.array([0.05, 0.05, 0.05], dtype=np.dtype('d')) 
        else:
            self.semiDiameters=semiDiameters 
                  
        #The geometric factors
        self.B1=(self.semiDiameters[1]**2-self.semiDiameters[2]**2)/ (self.semiDiameters[1]**2+self.semiDiameters[2]**2)
        self.B2=(self.semiDiameters[2]**2-self.semiDiameters[0]**2)/ (self.semiDiameters[2]**2+self.semiDiameters[0]**2)
        self.B3=(self.semiDiameters[0]**2-self.semiDiameters[1]**2)/ (self.semiDiameters[0]**2+self.semiDiameters[1]**2)
        self.radius=np.mean(self.semiDiameters) #The equivalent radius (Used for the contact, but contact needs to be improved to take into account the real ellipsoid shape instead of this spherical approximation)
        self.diameter=self.radius*2
            
        if time is None: #Creating the "stack" of times
            self.time=np.array([[0]], dtype=np.dtype('d')) 
        else:
            self.time=np.array([[time]], dtype=np.dtype('d')) 
            
        if position is None: #Creating the "stack" of Position of the centre of the particle
            self.position=np.array([[-2,  0,  0]], dtype=np.dtype('d')) 
        else:
            self.position=np.array([position], dtype=np.dtype('d'))
        
        self.calc_rAndtheta() #Inside the function, the radial and angular coordinates of the centre are set
                        
        if velocity is None: #Creating the stack of  Velocity of the centre of the particle
            self.velocity=np.array([[0,  0,  0]], dtype=np.dtype('d'))
        else:
            self.velocity=np.array([velocity], dtype=np.dtype('d')) 
        
        if eulerAngles is None: #Creating the stack Euler angles using Hinch1979 and Jesek1994 convention
            self.eulerAngles=np.array([[0,  0,  0]], dtype=np.dtype('d'))
        else:
            self.eulerAngles=np.array([eulerAngles],  dtype=np.dtype('d'))
            
        if eulerAngularVel is None: #Creating the stack of Euler angular velocities
            self.eulerAngularVel=np.array([[0,  0,  0]], dtype=np.dtype('d'))
        else:
            self.eulerAngularVel=np.array([eulerAngularVel],  dtype=np.dtype('d'))
                
        self.orientation=np.array([self.getOrientation()],  dtype=np.dtype('d'))
        #print('Shape of self.orientation is', np.shape(self.orientation))
        #print(self.orientation)
        if integratorType is None:
            self.integratorType='dopri5'
        else:
            self.integratorType=integratorType

        self.Frozen=False #By default the particle is not initialy frozen            
        if willFreeze is None:
            self.willFreeze = True
        else:
            self.willFreeze=willFreeze
        
        self.checkFreeze() #Freezing if the initial condition dictates that
        
        if readFluidVelocity is None:
            def zeroVelocity(X):
                VHere=np.array([0,  0,  0], dtype=np.dtype('d'))
                return VHere
            self.readFluidVelocity = zeroVelocity
        else:
            self.readFluidVelocity = readFluidVelocity
            
        if readFluidGradient is None:
            def zeroGradient(X):
                GHere=np.array([[0,  0,  0], [0,  0,  0], [0,  0,  0]], dtype=np.dtype('d'))
                return GHere
            self.readFluidGradient = zeroGradient
        else:
            self.readFluidGradient = readFluidGradient
            
        if xLimits is None: #Limits in X space for the particle to move
            self.xLimits=np.array([-100, 2], dtype=np.dtype('d')) 
        else:
            self.xLimits=xLimits
            
        if yLimits is None: #Limits in Y space for the particle to move
            self.yLimits=np.array([-2, 2], dtype=np.dtype('d')) 
        else:
            self.yLimits=yLimits
            
        self.checkInside() #Checking if the particle is inside the solution domain
        #Setting the particle to be ready for integration
        #self.setIntegrator()
    #End of the constructor of the class
    
    #This function defines a rotation matrix given any set of Euler angles (It is used in the constructor and in the dY/dt functions)
    def setRotationMatrix(self, givenEulerAngles):
        phi=givenEulerAngles[ 0]
        theta=givenEulerAngles[1]
        psi=givenEulerAngles[2]
        R=np.matrix([
                [np.cos(phi)*np.cos(psi)-np.sin(phi)*np.cos(theta)*np.sin(psi), 
                 np.sin(phi)*np.cos(psi)+np.cos(phi)*np.cos(theta)*np.sin(psi),
                 np.sin(theta)*np.sin(psi)],
                [-np.sin(psi)*np.cos(phi)-np.cos(theta)*np.cos(psi)*np.sin(phi),
                 -np.sin(phi)*np.sin(psi)+np.cos(theta)*np.cos(phi)*np.cos(psi),
                 np.sin(theta)*np.cos(psi)],
                [np.sin(phi)*np.sin(theta),
                 -np.sin(theta)*np.cos(phi),
                 np.cos(theta)]
                ],  dtype=np.dtype('d'))
        return R
    
    def getOrientation(self):
        R=self.setRotationMatrix(self.eulerAngles[-1, :]) #The rotation matrix
        orientationBody=np.matrix([[1], [0], [0]],  dtype=np.dtype('d')) #Orientation vector in the body coordinate system
        orientationWorld=np.matmul(R.transpose(), orientationBody) #Orientation vector in the world coordinate system
        orientation=np.array(np.array(orientationWorld.transpose()).flatten(),  dtype=np.dtype('d'))    #World Orientation vector flattened & saved in a stack of orientations
        return orientation
    
    #Function that calculates the radial and angular coordinates
    def calc_rAndtheta(self):
        self.r=np.sqrt(self.position[-1, 0]**2+self.position[-1, 1]**2) #Radial coordinate in cylindrical
        self.theta=np.arctan2(self.position[-1, 1], self.position[-1, 0]) #Angular coordinate in cylindrical

    #Function that sets the euler angles any time after the declaration of the particle
    def set_eulerAngles(self, givenEulerAngles):
            self.eulerAngles[-1, :]=np.array(givenEulerAngles,  dtype=np.dtype('d')) 
    
    #Function that checks if the particle should be Frozen
#AEGNeed: this function needs to be updated in order to take into account elliptical shape rather than an spherical equivalent
    def checkFreeze(self) :
        if self.willFreeze :
            if self.r<=1.0/2+self.radius :
                self.Frozen = True
                print('Particle was Frozen')
            
            
    #Function that checks if the particle is inside the solution domain or not
    def checkInside(self) :
            if self.position[-1,0 ]>self.xLimits[0] and self.position[-1,0 ]<self.xLimits[1] and self.position[-1,1 ]>self.yLimits[0] and self.position[-1,1 ]<self.yLimits[1]:
                self.Inside = True
            else:
                self.Inside=False
                print('Particle went out of domain')
                
    #Function for defining the perfect tracer behaviour dy/dt = f(t, y)
    def dYdt_perfectRotatingTracer(self, t,  Y):
        """
        Computes the derivative dYdT for a perfect tracer particle
        """
        positionHere=np.array([Y[0],  Y[1],  Y[2]], dtype=np.dtype('d'))
        fluidVelHere=self.readFluidVelocity(positionHere)
        velocityHere=fluidVelHere
        accelerationHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        
        eulerAnglesHere=np.array([Y[6],  Y[7],  Y[8]], dtype=np.dtype('d'))
                
        RHere=self.setRotationMatrix(eulerAnglesHere) #The rotation Matrix Here
        fluidGradientHere=self.readFluidGradient(positionHere)
        #Symmetric and Antisymmetric parts of the gradient
        EWorld=0.5*(fluidGradientHere + fluidGradientHere.transpose())
        OWorld=0.5*(fluidGradientHere - fluidGradientHere.transpose())
        #Transforming into the local body coordinate system
        EBody=np.matmul(np.matmul(RHere, EWorld), RHere.transpose())
        #print('Shape of EBody is', np.shape(EBody))
        OBody=np.matmul(np.matmul(RHere, OWorld), RHere.transpose())
        #print('Shape of OBody is', np.shape(OBody))
        
        #Obtaining the local angular velocities
        w1=self.B1*EBody[2, 1]+OBody[2, 1] #Using + following Hinch1979
        w2=self.B2*EBody[0, 2]+OBody[0, 2]
        w3=self.B3*EBody[1, 0]+OBody[1, 0]
        
        #Equations of the angular velocity of the Euler angles
        phi=eulerAnglesHere[ 0]
        theta=eulerAnglesHere[1]
        psi=eulerAnglesHere[2]
        smallNumber=1E-16
        phiDot=(w1*np.sin(psi)+w2*np.cos(psi))/(np.sin(theta)+smallNumber)
        thetaDot=w1*np.cos(psi)-w2*np.sin(psi)
        psiDot=w3-phiDot*np.cos(theta)
        eulerAngularVelHere=np.array([phiDot,  thetaDot,  psiDot], dtype=np.dtype('d'))
        eulerAngularAccHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        return np.concatenate((velocityHere,  accelerationHere,  eulerAngularVelHere,  eulerAngularAccHere))
    #End dYdT_perfectTracer

    #Function for defining the system dy/dt = f(t, y)
    def dYdt_system(self, t,  Y):
        """
        Defines the derivative function to use
        """
        dYdtHere=self.dYdt_perfectRotatingTracer(t, Y)
        return dYdtHere
    #End dYdT_system
        
    #Function for setting or resetting the integration
    def setIntegrator(self):
        """
        Initializes or reinitializes the integrator and initial conditions
        """
        self.INTG=ode(self.dYdt_system).set_integrator(self.integratorType) #defining integrator
        #print('Shape of self.position is', np.shape(self.position))
        #print('Shape of last row of self.position is', np.shape(self.position[-1,0: ]))
        initialConditions=np.concatenate((self.position[-1, 0:], self.velocity[-1,0: ], self.eulerAngles[-1,0: ], self.eulerAngularVel[-1,0: ]))
        #print('Shape of initialConditions is', np.shape(initialConditions))
        self.INTG.set_initial_value(initialConditions,  self.time[-1, 0]).set_f_params() #Initializing
    #End initializeIntegrator
    
    #Funtion for integrating the trajectory
    def integrateTrajectory(self, tEnd, dt):
        """
        Integrates trajectory until tEnd or unitle Froze or Outside the analysis domain
        """
        while self.INTG.successful() and (not self.Frozen) and self.INTG.t < tEnd and self.Inside:
            self.INTG.integrate(self.INTG.t+dt)
            #print('Shape of self.position is', np.shape(self.position))
            #print('Shape of y is', np.shape(self.INTG.y))
            #print('Shape of y chunk is', np.shape(self.INTG.y[:3]))
            self.position=np.append(self.position, np.array([self.INTG.y[0:3]], dtype=np.dtype('d')), axis=0) #Appending just the position elements from the internal y of the integrator
            #print('Shape of appended self.position is', np.shape(self.position))
            self.velocity=np.append(self.velocity, np.array([self.INTG.y[3:6]], dtype=np.dtype('d')), axis=0) #Appending just the position elements from the internal y of the integrator
            self.eulerAngles=np.append(self.eulerAngles, np.array([self.INTG.y[6:9]], dtype=np.dtype('d')), axis=0) #Appending just the position elements from the internal y of the integrator
            self.eulerAngularVel=np.append(self.eulerAngularVel, np.array([self.INTG.y[9:12]], dtype=np.dtype('d')), axis=0) #Appending just the position elements from the internal y of the integrator
            self.orientation=np.append(self.orientation, np.array([self.getOrientation()], dtype=np.dtype('d')), axis=0) #Appending just the position elements from the internal y of the integrator
            
            #print('Shape of self.time is', np.shape(self.time))
            #print('Shape of t is', np.shape(self.INTG.t))
            #print('Shape of t chunk is', np.shape(np.array([[self.INTG.t]], dtype=np.dtype('d'))))
            self.time=np.append(self.time, np.array([[self.INTG.t]], dtype=np.dtype('d')), axis=0)
            #print('Shape of appended self.time is', np.shape(self.time))
            
            self.calc_rAndtheta() #Updating cylindrical coordinates
            self.checkFreeze() #Checking for Frozen condition
            self.checkInside() #Checking if the particle is still inside the solution domain
#End of class particleBasic definition

#---------------------------------------------------------------------------------------------
#This class should be the base class for the swimming classes    
class particleBasicBrownianRotation(particleBasic):
    """
        Derived class with random reorientation due to Brownian Rotation
    """
    #The constructor
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
                        randomSeed=None):
        #Internal properties  with default values
        particleBasic.__init__(self, time=time, position=position, velocity=velocity, semiDiameters=semiDiameters,  eulerAngles=eulerAngles,  
                                               eulerAngularVel=eulerAngularVel, integratorType=integratorType, willFreeze=willFreeze, readFluidVelocity=readFluidVelocity, readFluidGradient=readFluidGradient, 
                                               xLimits=xLimits, yLimits=yLimits) #Calling the constructor of the particleBasic class
        if rotationalDiff is None:
            self.rotationalDiff=0.0
        else:
            self.rotationalDiff=rotationalDiff #rotationalDiffusivity (Brownian motion)
        if randomSeed is None:
            self.randomSeed=0
        else:
            self.randomSeed=int(np.around(randomSeed+10)) #rotationalDiffusivity (Brownian motion)
    #End of the constructor of this derived class
    
    #Function for defining the perfect tracer behaviour dy/dt = f(t, y)
    def dYdt_BrownianRotation(self, t,  Y,  dt):
        """
        Computes the derivative dYdT due to Brownian rotation
        """
        velocityHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        accelerationHere=np.array([0,  0,  0], dtype=np.dtype('d'))
               
        #Obtaining the random angles of rotation
        if (self.randomSeed != 0):
            np.random.seed(self.randomSeed+np.size(self.time)) #Seeding the random number in order to get the same values from the random utilities during the current time step (as Runge-Kutta uses this function many times per time step)
        else:
            np.random.seed() #Random seeding with the system value
            
        chooser=np.round(np.random.uniform(0, 1)) #Random number (either 0 or 1)
        #Following the Brownian motion rotational noise proposed in Rusconi2014
        if (self.rotationalDiff!=0):
            phiDotRDM=np.random.normal(0, 2*self.rotationalDiff/dt)*((chooser+1)%2)
            thetaDotRDM=np.random.normal(0, 2*self.rotationalDiff/dt)*chooser
            psiDotRDM=np.random.normal(0, 2*self.rotationalDiff/dt)*(chooser%2)
        else:
            phiDotRDM=0
            thetaDotRDM=0
            psiDotRDM=0
        eulerAngularVelHere=np.array([phiDotRDM,  thetaDotRDM,  psiDotRDM], dtype=np.dtype('d'))
        eulerAngularAccHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        return np.concatenate((velocityHere,  accelerationHere,  eulerAngularVelHere,  eulerAngularAccHere))
    #End dYdT_BrownianRotation
    
    #Function for defining the system dy/dt = f(t, y). This uses the original perfectRotatingTracer + the BrownianRotation defined here. Needs dt as an additional parameter
    def dYdt_system(self, t,  Y, dt):
        """
        Defines the derivative function to use
        """
        dYdtHere=self.dYdt_perfectRotatingTracer(t, Y)+self.dYdt_BrownianRotation(t, Y, dt)
        return dYdtHere
    #End dYdT_system
    
    #Function for setting or resetting the integration. Needs dt as an additional parameter
    def setIntegrator(self, dt):
        """
        Initializes or reinitializes the integrator and initial conditions
        """
        self.INTG=ode(self.dYdt_system).set_integrator(self.integratorType) #defining integrator
        #print('Shape of self.position is', np.shape(self.position))
        #print('Shape of last row of self.position is', np.shape(self.position[-1,0: ]))
        initialConditions=np.concatenate((self.position[-1, 0:], self.velocity[-1,0: ], self.eulerAngles[-1,0: ], self.eulerAngularVel[-1,0: ]))
        #print('Shape of initialConditions is', np.shape(initialConditions))
        self.INTG.set_initial_value(initialConditions,  self.time[-1, 0]).set_f_params(dt) #Initializing
    #End initializeIntegrator



#---------------------------------------------------------------------------------------------
#This class is just for testing
class particleBasicSimpleShear(particleBasic):
    """
        Derived class with algebraic simplifications for the simple shear flow of Hinch1979
    """
    #The constructor
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
                        yLimits=None):
        #Internal properties  with default values
        particleBasic.__init__(self, time=time, position=position, velocity=velocity, semiDiameters=semiDiameters,  eulerAngles=eulerAngles,  
                                               eulerAngularVel=eulerAngularVel, integratorType=integratorType, willFreeze=willFreeze, readFluidVelocity=readFluidVelocity, readFluidGradient=readFluidGradient, 
                                               xLimits=xLimits, yLimits=yLimits) #Calling the constructor of the particleBasic class
    #End of the constructor of this derived class
    
    #Function for defining the perfect tracer behaviour dy/dt = f(t, y)
    def dYdt_perfectRotatingTracer(self, t,  Y):
        """
        Computes the derivative dYdT for a perfect tracer particle
        """
        positionHere=np.array([Y[0],  Y[1],  Y[2]], dtype=np.dtype('d'))
        fluidVelHere=self.readFluidVelocity(positionHere)
        velocityHere=fluidVelHere
        accelerationHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        
        eulerAnglesHere=np.array([Y[6],  Y[7],  Y[8]], dtype=np.dtype('d'))
        phi=eulerAnglesHere[0]
        theta=eulerAnglesHere[1]
        psi=eulerAnglesHere[2]
        
        #Shape constants
        alpha=0.25*(self.B2-self.B1)
        beta=0.25*(self.B2+self.B1)
        gamma=0.25*self.B3
        
        #Equations of the angular velocity
        thetaDot=0.5*alpha*np.sin(2*theta)*np.sin(2*phi)+0.5*beta*(-np.sin(2*theta)*np.sin(2*phi)*np.cos(2*psi)-2*np.sin(theta)*np.cos(2*phi)*np.sin(2*psi))
        phiDot=-0.5+alpha*np.cos(2*phi)+beta*(-np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)+np.cos(2*phi)*np.cos(2*psi))
        psiDot=-alpha*np.cos(theta)*np.cos(2*phi)+beta*((np.cos(theta)**2 )*np.sin(2*phi)*np.sin(2*psi)-np.cos(theta)*np.cos(2*phi)*np.cos(2*psi))+\
                      gamma*((np.cos(theta)**2)*np.sin(2*phi)*np.sin(2*psi)-2*np.cos(theta)*np.cos(2*phi)*np.cos(2*psi)+np.sin(2*phi)*np.sin(2*psi))
        eulerAngularVelHere=np.array([phiDot,  thetaDot,  psiDot], dtype=np.dtype('d'))
        eulerAngularAccHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        return np.concatenate((velocityHere,  accelerationHere,  eulerAngularVelHere,  eulerAngularAccHere))
    #End dYdT_perfectRotatingTracer

#---------------------------------------------------------------------------------------------
#This class is just for testing    
class particleBasicSimpleShearDegenerated(particleBasic):
    """
        Derived class with algebraic simplifications for the simple shear flow of Hinch1979
    """
    #The constructor
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
                        yLimits=None):
        #Internal properties  with default values
        particleBasic.__init__(self, time=time, position=position, velocity=velocity, semiDiameters=semiDiameters,  eulerAngles=eulerAngles,  
                                               eulerAngularVel=eulerAngularVel, integratorType=integratorType, willFreeze=willFreeze, readFluidVelocity=readFluidVelocity, readFluidGradient=readFluidGradient, 
                                               xLimits=xLimits, yLimits=yLimits) #Calling the constructor of the particleBasic class
    #End of the constructor of this derived class
    
    #Function for defining the perfect tracer behaviour dy/dt = f(t, y)
    def dYdt_perfectRotatingTracer(self, t,  Y):
        """
        Computes the derivative dYdT for a perfect tracer particle
        """
        positionHere=np.array([Y[0],  Y[1],  Y[2]], dtype=np.dtype('d'))
        fluidVelHere=self.readFluidVelocity(positionHere)
        velocityHere=fluidVelHere
        accelerationHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        
        eulerAnglesHere=np.array([Y[6],  Y[7],  Y[8]], dtype=np.dtype('d'))
        phi=eulerAnglesHere[0]
        theta=eulerAnglesHere[1]
        psi=eulerAnglesHere[2]
        
        #Shape constants
        alpha=0.25*(self.B2-self.B1)
        beta=0.25*(self.B2+self.B1)
        gamma=0.25*self.B3
        
        #Equations of the angular velocity
        thetaDot=0.5*alpha*np.sin(2*theta)*np.sin(2*phi)
        phiDot=-0.5+alpha*np.cos(2*phi)
        psiDot=-alpha*np.cos(theta)*np.cos(2*phi)
        eulerAngularVelHere=np.array([phiDot,  thetaDot,  psiDot], dtype=np.dtype('d'))
        eulerAngularAccHere=np.array([0,  0,  0], dtype=np.dtype('d'))
        return np.concatenate((velocityHere,  accelerationHere,  eulerAngularVelHere,  eulerAngularAccHere))
    #End dYdT_perfectRotatingTracer

