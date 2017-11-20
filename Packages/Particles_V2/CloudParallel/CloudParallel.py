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

#Module where the global variables for MPI are defined
import Packages.Particles_V2.CloudParallel.myMPI as myMPI

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
        self.Nfrozen=np.zeros((1, 1), dtype=np.int) #Number of particles in the cloud that became frozen
        self.LocalFrozen=np.zeros((1, 1), dtype=np.int) #Frozen particles for the local rank in parallel solutions
            
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
    def defineDirectoryNames(self):
        self.cloudDir=os.path.join(self.resultsDir, self.cloudName)
        self.initialFinalDir=os.path.join(self.cloudDir,'InitialFinal')
        self.trajectoriesDir=os.path.join(self.cloudDir,'CompleteTrajectories')
        
    def createResultsDirectories(self):
        if (self.saveResults):
            if (not os.path.isdir(self.resultsDir)):
                os.makedirs(self.resultsDir)            
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
            
    def createInitialFinalArray(self):        
        sizeRecord=34 #So far the InitialFinal record has 16 elements (see sendInitialFinalRecord function)
        self.InitialFinalArray=np.ones((self.Nparticles, sizeRecord), dtype=np.dtype('d'))*-1.0
        self.recBuffer=np.ones((1, sizeRecord), dtype=np.dtype('d'))*-1.0
    
    def sendInitialFinalRecord(self, index, part):
        recordIni=np.concatenate((part.time[0, 0:], part.position[0, 0:], part.velocity[0,0: ], part.eulerAngles[0,0: ], part.eulerAngularVel[0,0: ], part.orientation[0, 0:]))
        recordFin=np.concatenate((part.time[-1, 0:], part.position[-1, 0:], part.velocity[-1,0: ], part.eulerAngles[-1,0: ], part.eulerAngularVel[-1,0: ], part.orientation[-1, 0:]))
        recordHere=np.concatenate((recordIni, recordFin))
        if part.Frozen:
            recordHere=np.concatenate((recordHere, [1.0]))
        else:
            recordHere=np.concatenate((recordHere, [0.0]))
        recordHere=np.concatenate(([index], recordHere))
        recordHere=recordHere.reshape(1, np.size(recordHere))
        myMPI.comm.Isend(recordHere, dest=0,  tag=index)
    
    def saveInitialFinalArray(self):        
        fileName=os.path.join(self.initialFinalDir,'InitialFinal.dat')
        with open(fileName, 'ab') as file_handle:        
            np.savetxt(file_handle,self.InitialFinalArray)
            
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
        self.defineDirectoryNames()
        if (myMPI.rank==0):
            print('Creating directories from rank=', myMPI.rank)
            self.createResultsDirectories()
            self.createInitialFinalArray()
            self.status0=myMPI.MPI.Status()
        else:
            print('Waiting for creation of tree directories from rank=', myMPI.rank)
        myMPI.comm.Barrier()
        
        #Defining how which particle trajectories are to be solved by each rank
        #local_N is the number of particle trajectories to be solved by each rank
        local_N0=((self.Nparticles//myMPI.size)*9)//10 #Integer division
        Nleft=self.Nparticles-local_N0 #The rank0 processor will only do a fraction of the operations and the rest of the time will be building the initial final array
        remainder=Nleft%(myMPI.size-1)
        local_NBase=Nleft//(myMPI.size-1) #Integer division
        arrayLocal_N=np.zeros(myMPI.size, dtype=np.int)
        for i in range(0, myMPI.size): #In python, remember to aways mark the end with 1 index above of what you want to get. The beginning is exactly the index you want to start with
            if (i==0):
                local_N=local_N0
            elif (i<=remainder):
                local_N=local_NBase+1
            else:
                local_N=local_NBase
            arrayLocal_N[i]=local_N
            
        #Now we estimate the individual ranges
        print('ARRAY', myMPI.rank, 'The arrayLocalN is:', arrayLocal_N)
        print('ARRAY', myMPI.rank,'The remainder=', remainder, 'local_NBase=', local_NBase, 'local_N', myMPI.rank,'=', local_N)
        if (myMPI.rank==0):    
            suma=0
        else:
            suma=np.sum(arrayLocal_N[0:myMPI.rank]) #In python, remember to aways mark the end with 1 index above of what you want to get. The beginning is exactly the index you want to start with
        print('ARRAY', myMPI.rank, 'suma=', suma)    
        local_Ini=suma   
        local_End=local_Ini+arrayLocal_N[myMPI.rank]
        print('The local ranges are: rank=',myMPI.rank, 'Ini=', local_Ini, 'End=',  local_End, 'N=',  arrayLocal_N[myMPI.rank])
        
        #Parallel cycle for solving the trajectories
        for i in range(local_Ini, local_End): 
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
            print('From rank=', myMPI.rank, ' ', self.cloudName,'- Solving the trajectory:', i)
            pHere.integrateTrajectory(tEnd, dt)
            
            #Obtaining the capture percentage
            if (pHere.Frozen):
                #self.Nfrozen+=1
                self.LocalFrozen[0]=self.LocalFrozen[0]+1
                print('From rank=', myMPI.rank, ',Trajectory: ', i,'Now frozen=', self.LocalFrozen[0])
                        
            #Saving the results            
            if i<self.Ntrajectories:
                print(self.cloudName,'- Saving the trajectories')                
                self.saveTrajectory(pHere, i)
            else:
                print(self.cloudName,'- Just saving initial and final points')
            self.sendInitialFinalRecord(i, pHere)            
            if (myMPI.rank==0):
                tagHere=i #For the self sent message in rank 0
                myMPI.comm.Recv(self.recBuffer, source=0, tag=tagHere)
                self.InitialFinalArray[tagHere, :]=self.recBuffer 
                for j in range(1, myMPI.size):
                    myMPI.comm.Iprobe(source=j, tag=myMPI.ANY_TAG, status=self.status0)
                    tagHere=self.status0.Get_tag()
                    if (tagHere>=0):
                        myMPI.comm.Irecv(self.recBuffer, source=j, tag=tagHere)
                        self.InitialFinalArray[tagHere, :]=self.recBuffer                                                                    
        #End of the cycle
        myMPI.comm.Barrier()
        
        myMPI.comm.Reduce(self.LocalFrozen, self.Nfrozen, op=myMPI.MPI.SUM, root=0) #Summing all the counters of frozen particles
        
        if (myMPI.rank==0):
            for j in range(0, self.Nparticles):
                if (self.InitialFinalArray[j, 0]==-1.0):
                    print('Receiving the leftovers of InitialFinal records')
                    myMPI.comm.Recv(self.recBuffer, source=myMPI.ANY_SOURCE, tag=j)
                    self.InitialFinalArray[j, :]=self.recBuffer
            self.saveInitialFinalArray() #Saving the InitialFinal array            
            self.saveFrozenPct()    #Saving the percentage of capture
        #End if(myMPI.rank==0), used for the final writing of the InitialFinalArray and FrozenPct
        
#End of cloud class
