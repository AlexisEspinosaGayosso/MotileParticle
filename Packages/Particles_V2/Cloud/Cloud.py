"""
Module for basic particles
"""

#Importing necessary 'typical' modules
import os #For interacting with operating system
import shutil #For interacting with operating system
from datetime import datetime #For catching the date for the directory names
import numpy as np #Fundamental package for scientific computing
from scipy.integrate import ode #Fundamental package for numerical integration of ordinary differenttial equations

#Adding the root path for our own modules
from os import path
import sys
sys.path.append(path.abspath('../../../../Python')) #IMPORTANT. Put as many .. to reach the level of the Python directory and then we can use the directory Packages below for importing our modules and packages

#Adding our own modules
import Packages.Particles_V2.Basic.ParticleBasic as PB
import Packages.Particles_V2.SmoothSwimmer.SmoothSwimmer as SS
import Packages.Particles_V2.ErraticSwimmer.ErraticSwimmer as ES
import Packages.Particles_V2.HelicalSwimmer.HelicalSwimmer as HS
import Packages.FluidVelocity_V2.FluidVelocity as FV

#Defining the class particleBasic
class generalCloud(object):
    """
    General class for creating a cloud of particles
    """
    #Constructor of the class
    def __init__(self,
                        #Parameters for the CLOUD
                        cloudName=None,
                        Nparticles=None,
                        particleType=None,
                        Ntrajectories=None,
                        fluidField=None,
                        resultsDir=None, 
                        saveResults=None,
                        overwrite=None,
                        #Parameters used for the basic class
                        randomSeed=None,
                        time=None, 
                        position=None,
                        velocity=None,
                        semiDiameters=None, 
                        eulerAngles=None,
                        eulerAngularVel=None,
                        integratorType=None, 
                        willFreeze=None, 
                        ##readFluidVelocity=None,  #Will be deduced from the fluidField received above
                        ##readFluidGradient=None, #Will be deduced from the fluidField received above
                        xLimits=None, 
                        yLimits=None,
                        #Additional parameters for the Smooth Swimmer
                        swimmingSpeed=None, 
                        #Additional parameters for the Brownian Rotation                        
                        rotationalDiff=None,
                        #Additional parameters for the Erratic swimmers
                        meanRunTime=None,
                        stdDevRunTime=None, 
                        tumbleType=None, 
                        meanTumbleAngle=None, 
                        stdDevTumbleAngle=None, 
                        meanReverseAngle=None, 
                        stdDevReverseAngle=None, 
                        meanFlickAngle=None, 
                        stdDevFlickAngle=None, 
                        reversePct=None, 
                        #Additional parameters for the Helical swimmers
                        pitch=None,
                        radius=None,
                        angularFreq=None):
        
        #Internal properties  with default values
        if cloudName is None: #Type of particles in the cloud
            self.cloudName='DefaultCloud'
        else:
            self.cloudName=cloudName
        if Nparticles is None: #Number of particles in the cloud
            self.Nparticles=1
        else:
            self.Nparticles=Nparticles
            
        self.Nfrozen=0 #Number of particles in the cloud that became frozen
            
        if particleType is None: #Type of particles in the cloud
            self.particleType='SmoothSwimmer'
        else:
            self.particleType=particleType
            
        if Ntrajectories is None: #Number of particles whose complete trajectories are going to be saved
            self.Ntrajectories=1
        else:
            self.Ntrajectories=Ntrajectories
            
        if fluidField is None: #Type of particles in the cloud
            print('No fluidField was given for cloud ',  self.cloudName)
            print('Cannot continue without this. Exiting')
            exit()
        else:
            self.fluidField=fluidField
        
        if saveResults is None: #Number of particles whose complete trajectories are going to be saved
            self.saveResults=False
        else:
            self.saveResults=saveResults
            
        if resultsDir is None: #Number of particles whose complete trajectories are going to be saved
            self.resultsDir=os.path.join('.','Results')
        else:
            self.resultsDir=resultsDir
            
        if overwrite is None: #Number of particles whose complete trajectories are going to be saved
            self.overwrite=False
        else:
            self.overwrite=overwrite
        
        #The rest of the parameters are just received as they are
        if randomSeed is None:
            self.randomSeed=0
        else:
            self.randomSeed=randomSeed
        self.time=time
        self.position=position
        self.velocity=velocity
        self.semiDiameters=semiDiameters
        if eulerAngles is None:
            self.eulerAngles=[0.0,  0.0,  0.0]
        else:
            self.eulerAngles=eulerAngles
        self.eulerAngularVel=eulerAngularVel
        self.integratorType=integratorType
        self.willFreeze=willFreeze
        
        if xLimits is None:
            self.xLimits=self.fluidField.xLimits
        else:
            xLeft=np.maximum(self.fluidField.xLimits[0], xLimits[0])
            xRight=np.minimum(self.fluidField.xLimits[1], xLimits[1])
            self.xLimits=[xLeft, xRight]        
        if yLimits is None:
            self.yLimits=self.fluidField.yLimits
        else:
            yLeft=np.maximum(self.fluidField.yLimits[0], yLimits[0])
            yRight=np.minimum(self.fluidField.yLimits[1], yLimits[1])
            self.yLimits=[yLeft,yRight]
                        #Additional parameters for the Smooth Swimmer
        self.swimmingSpeed=swimmingSpeed
                        #Additional parameters for the Brownian Rotation                        
        self.rotationalDiff=rotationalDiff
                        #Additional parameters for the Erratic swimmers
        self.meanRunTime=meanRunTime
        self.stdDevRunTime=stdDevRunTime
        self.tumbleType=tumbleType
        self.meanTumbleAngle=meanTumbleAngle
        self.stdDevTumbleAngle=stdDevTumbleAngle
        self.meanReverseAngle=meanReverseAngle
        self.stdDevReverseAngle=stdDevReverseAngle
        self.meanFlickAngle=meanFlickAngle
        self.stdDevFlickAngle=stdDevFlickAngle
        self.reversePct=reversePct
                        #Additional parameters for the Helical swimmers
        self.pitch=pitch
        self.radius=radius
        self.angularFreq=angularFreq
                           
    #Function for creating the results directories
    def createResultsDirectories(self):
        if (self.saveResults):
            if (not os.path.isdir(self.resultsDir)):
                os.makedirs(self.resultsDir)            
            self.cloudDir=os.path.join(self.resultsDir, self.cloudName)            
            if (os.path.isdir(self.cloudDir)):
                if (self.overwrite):
                    shutil.rmtree(self.cloudDir)
                else:
                    print('Results for this cloud:',  self.cloudName)
                    print('Already exist and overwrite parameter is False')
                    print('Backup the existing results with other name before running')
                    print('Or set overwrite to =False')
                    print('Exiting now')
                    exit()                    
            else:
                os.makedirs(self.cloudDir)
            self.initialFinalDir=os.path.join(self.cloudDir,'InitialFinal')
            self.trajectoriesDir=os.path.join(self.cloudDir,'CompleteTrajectories')
            if (not os.path.isdir(self.initialFinalDir)):
                os.makedirs(self.initialFinalDir)
            if (not os.path.isdir(self.trajectoriesDir)):
                os.makedirs(self.trajectoriesDir)
                
    def saveFrozenPct(self):
        fileName=os.path.join(self.cloudDir,'FrozenPct.dat')
        recordHere=np.array([self.Nparticles,  self.Nfrozen, float(self.Nfrozen)/float(self.Nparticles) ], dtype=np.dtype('d'))
        recordHere=np.reshape(recordHere,(1, 3))
        with open(fileName, 'ab') as file_handle:        
            np.savetxt(file_handle,recordHere)
            
    def saveInitialFinalRecord(self, part):
        recordIni=np.concatenate((part.time[0, 0:], part.position[0, 0:], part.velocity[0,0: ], part.eulerAngles[0,0: ], part.eulerAngularVel[0,0: ], part.orientation[0, 0:]))
        recordFin=np.concatenate((part.time[-1, 0:], part.position[-1, 0:], part.velocity[-1,0: ], part.eulerAngles[-1,0: ], part.eulerAngularVel[-1,0: ], part.orientation[-1, 0:]))
        recordHere=np.concatenate((recordIni, recordFin))
        if part.Frozen:
            recordHere=np.concatenate((recordHere, [1.0]))
        else:
            recordHere=np.concatenate((recordHere, [0.0]))
        recordHere=recordHere.reshape(1, np.size(recordHere))
        fileName=os.path.join(self.initialFinalDir,'InitialFinal.dat')
        with open(fileName, 'ab') as file_handle:        
            np.savetxt(file_handle,recordHere)
            
    def saveTrajectory(self, part, i):
        fileName=os.path.join(self.trajectoriesDir,'Trajectory_'+str(i)+'.dat')        
        with open(fileName, 'ab') as file_handle:        
            np.savetxt(file_handle,np.transpose(part.time))
            np.savetxt(file_handle,np.transpose(part.position))
            np.savetxt(file_handle,np.transpose(part.velocity))
            np.savetxt(file_handle,np.transpose(part.eulerAngles))
            np.savetxt(file_handle,np.transpose(part.orientation))
            
    #Function for integrating the trajectories of the N particles
    def integrateCloud(self, tEnd,  dt):
         #Preparation for saving results
        self.createResultsDirectories()
        
        #Opening the result files        
        
        for i in range (0, self.Nparticles):
            #The random Initial orientation
            if i==0:
                eulerAnglesIni=self.eulerAngles
            else:
                #Obtaining the random angles of rotation
                if (self.randomSeed != 0):
                    np.random.seed(self.randomSeed+i) #Seeding the random number in order to get the same values from the random utilities during the current time step (as Runge-Kutta uses this function many times per time step)
                else:
                    np.random.seed() #Random seeding with the system value
                                
                #Following the Brownian motion rotational noise proposed in Rusconi2014
                phi=np.random.uniform(0, 2*np.pi)
                theta=np.random.uniform(0, 2*np.pi)
                psi=np.random.uniform(0, 2*np.pi)
                eulerAnglesIni=np.array([phi,  theta,  psi], dtype=np.dtype('d'))
                
            if self.particleType=='SmoothSwimmer':
                pHere=SS.smoothSwimmer(
                    time=self.time, 
                    position=self.position, 
                    velocity=self.velocity, 
                    semiDiameters=self.semiDiameters, 
                    eulerAngles=eulerAnglesIni, 
                    eulerAngularVel=self.eulerAngularVel, 
                    integratorType=self.integratorType, 
                    willFreeze=self.willFreeze, 
                    readFluidVelocity=self.fluidField.velocityField,
                    readFluidGradient=self.fluidField.gradientField,
                    xLimits=self.xLimits, 
                    yLimits=self.yLimits, 
                    rotationalDiff=self.rotationalDiff,
                    randomSeed=self.randomSeed,
                    swimmingSpeed=self.swimmingSpeed
                    )
            elif self.particleType=='ErraticSwimmer':
                pHere=ES.erraticSwimmer(
                    time=self.time, 
                    position=self.position, 
                    velocity=self.velocity, 
                    semiDiameters=self.semiDiameters, 
                    eulerAngles=eulerAnglesIni, 
                    eulerAngularVel=self.eulerAngularVel, 
                    integratorType=self.integratorType, 
                    willFreeze=self.willFreeze, 
                    readFluidVelocity=self.fluidField.velocityField,
                    readFluidGradient=self.fluidField.gradientField,
                    xLimits=self.xLimits, 
                    yLimits=self.yLimits, 
                    rotationalDiff=self.rotationalDiff,
                    randomSeed=self.randomSeed,
                    swimmingSpeed=self.swimmingSpeed, 
                    meanRunTime=self.meanRunTime, 
                    tumbleType=self.tumbleType, 
                    meanReverseAngle=self.meanReverseAngle, 
                    stdDevReverseAngle=self.stdDevReverseAngle, 
                    meanFlickAngle=self.meanFlickAngle, 
                    stdDevFlickAngle=self.stdDevFlickAngle, 
                    reversePct=self.reversePct
                    )
            elif self.particleType=='HelicalSwimmer':
                pHere=HS.helicalSwimmer(
                    time=self.time, 
                    position=self.position, 
                    velocity=self.velocity, 
                    semiDiameters=self.semiDiameters, 
                    eulerAngles=eulerAnglesIni, 
                    eulerAngularVel=self.eulerAngularVel, 
                    integratorType=self.integratorType, 
                    willFreeze=self.willFreeze, 
                    readFluidVelocity=self.fluidField.velocityField,
                    readFluidGradient=self.fluidField.gradientField,
                    xLimits=self.xLimits, 
                    yLimits=self.yLimits, 
                    rotationalDiff=self.rotationalDiff,
                    randomSeed=self.randomSeed,
                    pitch=self.pitch,
                    radius=self.radius,
                    angularFreq=self.angularFreq
                    )
            elif self.particleType=='PassiveBasic':
                pHere=PB.particleBasic(
                    time=self.time, 
                    position=self.position, 
                    velocity=self.velocity, 
                    semiDiameters=self.semiDiameters, 
                    eulerAngles=eulerAnglesIni, 
                    eulerAngularVel=self.eulerAngularVel, 
                    integratorType=self.integratorType, 
                    willFreeze=self.willFreeze, 
                    readFluidVelocity=self.fluidField.velocityField,
                    readFluidGradient=self.fluidField.gradientField, 
                    xLimits=self.xLimits, 
                    yLimits=self.yLimits
                    )
            elif self.particleType=='PassiveBrownian':
                pHere=PB.particleBasicBrownianRotation(
                    time=self.time, 
                    position=self.position, 
                    velocity=self.velocity, 
                    semiDiameters=self.semiDiameters, 
                    eulerAngles=eulerAnglesIni, 
                    eulerAngularVel=self.eulerAngularVel, 
                    integratorType=self.integratorType, 
                    willFreeze=self.willFreeze, 
                    readFluidVelocity=self.fluidField.velocityField,
                    readFluidGradient=self.fluidField.gradientField, 
                    xLimits=self.xLimits, 
                    yLimits=self.yLimits, 
                    rotationalDiff=self.rotationalDiff,
                    randomSeed=self.randomSeed
                    )
            else:
                print('particleType=', self.particleType, 'is wrong.')
                print('The names of the particleType should be:')
                print('SmoothSwimmer,ErraticSwimmer,HelicalSwimmer')
                print('or')
                print('PassiveBasic, PassiveBrownian')
                print('Exiting')
                exit()
            
            #Integrating the trajectory
            pHere.setIntegrator(dt)
            print(self.cloudName,'- Solving the trajectory:', i)
            pHere.integrateTrajectory(tEnd, dt)
            
            #Obtaining the capture percentage
            if (pHere.Frozen):
                self.Nfrozen+=1
            
            
            #Saving the results            
            if i<self.Ntrajectories:
                print(self.cloudName,'- Saving the trajectories')
                self.saveInitialFinalRecord(pHere)
                self.saveTrajectory(pHere, i)
            else:
                print(self.cloudName,'- Just saving initial and final points')
                self.saveInitialFinalRecord(pHere)
        #End of the cycle
        self.saveFrozenPct()    #Saving the percentage of capture
        
#End of cloud class
