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
import Packages.Particles_V2.SmoothSwimmer.SmoothSwimmer as SS

#Defining the class particleBasic
class erraticSwimmer(SS.smoothSwimmer):
    """
    Derived class for particles with erratic swimming capabilities
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
                        swimmingSpeed=None,
                        randomSeed=None, 
                        meanRunTime=None,
                        stdDevRunTime=None, 
                        tumbleType=None, 
                        meanTumbleAngle=None, 
                        stdDevTumbleAngle=None, 
                        meanReverseAngle=None, 
                        stdDevReverseAngle=None, 
                        meanFlickAngle=None, 
                        stdDevFlickAngle=None, 
                        reversePct=None):
        #Internal properties  with default values
               
        SS.smoothSwimmer.__init__(self, time=time, position=position, velocity=velocity, semiDiameters=semiDiameters,  eulerAngles=eulerAngles,  
                                               eulerAngularVel=eulerAngularVel, integratorType=integratorType, willFreeze=willFreeze, readFluidVelocity=readFluidVelocity, readFluidGradient=readFluidGradient, 
                                               xLimits=xLimits, yLimits=yLimits, rotationalDiff=rotationalDiff, randomSeed=randomSeed,  swimmingSpeed=swimmingSpeed) #Calling the constructor of the particleBasicBrownianRotation class
        if meanRunTime is None:
            self.meanRunTime=0.0
        else:
            self.meanRunTime=meanRunTime #Mean run time
        if stdDevRunTime is None:
            self.stdDevRunTime=0.2*self.meanRunTime
        else:
            self.stdDevRunTime=stdDevRunTime #Standard Deviation of run time
        if tumbleType is None:
            self.tumbleType='Tumble'
        else:
            self.tumbleType=tumbleType
            
        if meanTumbleAngle is None:
            self.meanTumbleAngle=60.0/180*np.pi
        else:
            self.meanTumbleAngle=meanTumbleAngle #Mean tumble angle in Radians
        if stdDevTumbleAngle is None:
            self.stdDevTumbleAngle=0.2*self.meanTumbleAngle
        else:
            self.stdDevTumbleAngle=stdDevTumbleAngle #Standard Deviation of tumble angle in Radians
            
        if meanReverseAngle is None:
            self.meanReverseAngle=180.0/180*np.pi
        else:
            self.meanReverseAngle=meanReverseAngle #Mean reverse angle in Radians
        if stdDevReverseAngle is None:
            self.stdDevReverseAngle=0.2*self.meanReverseAngle
        else:
            self.stdDevReverseAngle=stdDevReverseAngle #Standard Deviation of reverse angle in Radians
            
        if meanFlickAngle is None:
            self.meanFlickAngle=90.0/180*np.pi
        else:
            self.meanFlickAngle=meanFlickAngle #Mean flick angle in Radians
        if stdDevFlickAngle is None:
            self.stdDevFlickAngle=0.2*self.meanFlickAngle
        else:
            self.stdDevFlickAngle=stdDevFlickAngle #Standard Deviation of flick angle in Radians
            
        if reversePct is None:
            self.reversePct=1.0/3.0
        else:
            self.reversePct=reversePct #The percentage of duration of a reverse trajectory with respect to a run trajectory
    #End of the constructor of the erraticSwimmer class
    
    #Function for changing the orientation angles with a tumble, flick or reverse
    def doChangeInAngle(self, meanChangeHere, stdDevHere):
        """
        Function to adjusting the orientation angle with tumbles, flicks or reverses
        """
        #Obtaining the random angles of rotation       
        if (self.randomSeed != 0):
            np.random.seed(self.randomSeed+np.size(self.time)) #Seeding the random number in order to get the same values from the random utilities during the current time step (as Runge-Kutta uses this function many times per time step)
        else:
            np.random.seed() #Random seeding with the system value
        chooser=np.round(np.random.uniform(0, 1)) #Random number (either 0 or 1)
        if stdDevHere==0:
            phiChange=meanChangeHere*((chooser+1)%2)
            thetaChange=meanChangeHere*chooser
            psiChange=meanChangeHere*(chooser%2)
        else:
            phiChange=np.random.normal(meanChangeHere, stdDevHere)*((chooser+1)%2)
            thetaChange=np.random.normal(meanChangeHere, stdDevHere)*chooser
            psiChange=np.random.normal(meanChangeHere, stdDevHere)*(chooser%2)

#        #Use the following lines for Test with changes only in the plane
#        phiChange=np.random.normal(meanChangeHere, stdDevHere)
#        thetaChange=0
#        psiChange=0
        directionChange=np.array([phiChange,  thetaChange,  psiChange], dtype=np.dtype('d'))
        self.eulerAngles[-1, :]=self.eulerAngles[-1, :]+directionChange        
    #End doTumble
    

    #Funtion for integrating the trajectory
    def integrateTrajectory(self, tEnd, dt):
        """
        Integrates trajectory until tEnd
        """
        #Defining parameters before integrating trajectories
        currentRunTime=0.0
        runTypeHere='Run'
        if (self.randomSeed != 0):
            np.random.seed(self.randomSeed+np.size(self.time)) #Seeding the random number in order to get the same values from the random utilities during the current time step (as Runge-Kutta uses this function many times per time step)
        else:
            np.random.seed() #Random seeding with the system value
        meanRunTimeHere=np.random.normal(self.meanRunTime, self.stdDevRunTime) #Adjusting the mean run time with a normal distribution
                
        #Integrating the trajectories
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
            self.time=np.append(self.time, np.array([[self.INTG.t]], dtype=np.dtype('d')), axis=0)
 
            #Deciding if it is time to tumble or keep running (as in Kiorbe and Jackson 2001 without considering chemotaxis)            
            currentRunTime=currentRunTime+dt
            testHere=np.random.uniform(0, 1) 
            probabilityHere=currentRunTime/meanRunTimeHere            
                       
            #---------------------------------------------------------------------------
            if (probabilityHere<testHere):
                #print(runTypeHere)
                runTypeHere=runTypeHere
            else:
                if (self.tumbleType=='Tumble'):
                    self.doChangeInAngle(self.meanTumbleAngle, self.stdDevTumbleAngle)
                    self.setIntegrator(dt)
                    currentRunTime=0.0 #Resetting the counter of the run time
                    meanRunTimeHere=np.random.normal(self.meanRunTime, self.stdDevRunTime) #Adjusting the mean run time with a normal distribution
                    #print("Tumbling")
                elif (self.tumbleType=='ReverseFlick'):
                    if (runTypeHere=='Run'):
                        self.doChangeInAngle(self.meanReverseAngle, self.stdDevReverseAngle)
                        #print("Reversing")
                        runTypeHere='Reverse'
                        meanRunTimeHere=np.random.normal(self.meanRunTime*self.reversePct, self.stdDevRunTime*self.reversePct) #Adjusting the mean run time with a normal distribution for the next swim (a Reverse)
                    elif (runTypeHere=='Reverse'):
                        self.doChangeInAngle(self.meanFlickAngle, self.stdDevFlickAngle)                                            
                        #print("Flicking")
                        runTypeHere='Run'
                        meanRunTimeHere=np.random.normal(self.meanRunTime, self.stdDevRunTime) #Adjusting the mean run time with a normal distribution for the next swim (a Run)
                    self.setIntegrator(dt)
                    currentRunTime=0.0 #Resetting the counter of the run time
                    
                                        
            #------------------------------------------------------
            self.orientation=np.append(self.orientation, np.array([self.getOrientation()], dtype=np.dtype('d')), axis=0) #Appending just the position elements from the internal y of the integrator
            self.calc_rAndtheta() #Updating cylindrical coordinates
            self.checkFreeze() #Checking for Frozen condition
            self.checkInside() #Checking if the particle is still inside the solution domain
#End of class particleBasic definition    
    
#End of the derived class erraticSwimmer definition
