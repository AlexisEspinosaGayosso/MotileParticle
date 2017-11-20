#Importing necessary 'typical' modules
import os #For interacting with operating system
import numpy as np #Fundamental package for scientific computing
from scipy.integrate import ode #Fundamental package for numerical integration of ordinary differenttial equations
#import matplotlib as mpl #Plotting package emulating matlab plotting tools
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

#Adding the root path for our own modules
from os import path
import sys
sys.path.append(path.abspath('../../../../Python')) #IMPORTANT. Put as many .. to reach the level of the Python directory and then we can use the directory Packages below for importing our modules and packages

#Adding our own modules
import Packages.Particles_V2.Basic.ParticleBasic as PB
import Packages.Particles_V2.SmoothSwimmer.SmoothSwimmer as SS
import Packages.Particles_V2.ErraticSwimmer.ErraticSwimmer as ES
import Packages.Particles_V2.HelicalSwimmer.HelicalSwimmer as HS
import Packages.Particles_V2.Cloud.Cloud as CL
import Packages.FluidVelocity_V2.FluidVelocity as FV

#Defining environment with parallel utilities
#from joblib import Parallel, delayed
import multiprocessing

#For measuring time
import time

print('The number of arguments is:',  np.shape(sys.argv))

#Swimmer and Environment Data
ReynoldsGiven=np.float(sys.argv[1]) #Reading Reynolds from the command line
print('ReynoldsGiven=', ReynoldsGiven)
ReynoldsReady=np.array([                 0.1,                      0.2,                      0.3,                     0.4,                      0.5,                         0.6,                     0.7,                     0.8,                      0.9,                        1.0],  dtype=np.float)
XSeedReady=np.array([                    -5.0,                    -5.0 ,                    -5.0,                    -5.0 ,                    -5.0 ,                      -2.5,                    -2.5,                     -2.5,                     -2.5,                     -2.5],  dtype=np.float)
PassiveEdge=np.array([4.44258989E-05, 4.29691527E-05, 4.24822230E-05, 4.25715586E-05, 4.30719037E-05,  6.58573103E-05, 6.58260442E-05, 6.59528544E-05, 6.61961651E-05, 6.65211154E-05],  dtype=np.float)
#First test factors: FactorEdge=np.array([                     10.0,                    10.0 ,                    9.0,                      8.0 ,                      7.0 ,                      6.0,                      5.0,                      4.0,                       3.0,                      2.0],  dtype=np.float)
FactorEdge=np.array([                     20.0,                    20.0 ,                    20.0,                      20.0 ,                      20.0 ,                  15.0,                      10.0,                   10.0,                       6.0,                 4.0],  dtype=np.float)
NReady=np.size(ReynoldsReady)
iRey=-1
eps=1E-3
for ii in range(0, NReady):
    if (ReynoldsReady[ii]<ReynoldsGiven+eps and ReynoldsReady[ii]>ReynoldsGiven-eps):
        iRey=ii
        ReynoldsHere=ReynoldsReady[ii]
        print('The index of existing Reynolds is iRey=', iRey)
        break
if (iRey==-1):
    print('The given Reynolds=', ReynoldsGiven, 'is not ready for analysis.')
    print('Ready Reynolds are:', ReynoldsReady)
    print('Exiting')
    sys.exit()


SwimmerTypeHere=sys.argv[2] #Reading Reynolds from the command line
if SwimmerTypeHere=='Tumble':
    print('The swimmer is a type=', SwimmerTypeHere)
elif SwimmerTypeHere=='ReverseFlick':
    print('The swimmer is a type=', SwimmerTypeHere)
else:
    print('The swimmer type should be Tumble OR ReverseFlick and program received:', SwimmerTypeHere)
    print('Program is exiting')
    sys.exit()
    
#resultsDirHere='E:\Results' #Change the directory to wherever you will save the results
#resultsDirHere='/media/sf_WIN_K/Blake_Testing/Results_'+SwimmerTypeHere+'_Re'+str(ReynoldsHere)
resultsDirHere='/scratch/pawsey0106/espinosa/Python/Results_Extended200_'+SwimmerTypeHere+'_Re'+str(ReynoldsHere)
    
#Dimensional values
KinematicViscosity= 0.00000105 #[m^2/s]
CollectorDiameter=0.000100 #[m]
FluidVelocity=ReynoldsHere*KinematicViscosity/CollectorDiameter
TimeScale=CollectorDiameter/FluidVelocity
if SwimmerTypeHere=='Tumble':
    RealSwimSpeed=0.000030 #[m/s]
    RealRunDuration=1 #[s]
    reversePctHere=None
    RealFullDiameters=[0.000002,0.000001,0.000001] #[m]
    RealRotationalDiffusion= 0.16 #[rad^2/s]
    meanTumbleAngleHere=68/180*np.pi #[rad]
    stdDevTumbleAngleHere=36/180*np.pi #[rad]
    meanReverseAngleHere=None
    stdDevReverseAngleHere=None
    meanFlickAngleHere=None
    stdDevFlickAngleHere=None    
elif SwimmerTypeHere=='ReverseFlick':
    RealSwimSpeed=0.000045 #[m/s]
    RealRunDuration=0.4 #[s]
    reversePctHere=0.5
    RealFullDiameters=[0.000002,0.000001,0.000001] #[m] #Using the same for comparison, but vibrio values are larger=[4micro,2micro]
    RealRotationalDiffusion= 0.084 #[rad^2/s]
    meanTumbleAngleHere=None
    stdDevTumbleAngleHere=None
    meanReverseAngleHere=180/180*np.pi #[rad]
    stdDevReverseAngleHere=0/180*np.pi  #[rad]
    meanFlickAngleHere=90/180*np.pi  #[rad]
    stdDevFlickAngleHere=30/180*np.pi #[rad]


#Converted values
SemiDiametersHere=(np.array(RealFullDiameters)/2)/CollectorDiameter
SwimSpeedHere=RealSwimSpeed/FluidVelocity
RunDurationHere=RealRunDuration/TimeScale
RotationalDiffusionHere=RealRotationalDiffusion*TimeScale

print('The semidiameters to use are:', SemiDiametersHere)
print('The swimming speed to use is:', SwimSpeedHere)
print('The run duration to use is:', RunDurationHere)
print('The rotational diffusivity to use is:', RotationalDiffusionHere)
print('The swimmer type to use is:', SwimmerTypeHere)
print('The reverse Pct here is:', reversePctHere)

#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#Directory for the OpenFOAM sampled velocity fields
OpenFOAMDir=os.path.join('..','..', 'OpenFOAMFields') #Clearly check where are the OpenFOAMFields with respect to the running directory
print('OpenFOAM Directory is:')
print(OpenFOAMDir)


   

fieldForErratic=FV.creepingCylinderFlow(Reynolds=ReynoldsHere)
'''if (np.size(fieldForErratic.nonFilled)>0): #When using OpenFOAM, check if the reading was done properly
    print('Some sampled points in the mesh were not read properly:')
    print('Take a look into the object.nonFilled array')
    exit()'''
#Defining fluid velocity objects for the different testings
#fieldForErratic=FV.creepingCylinderFlow(Reynolds=0.5) #IMPORTANT, creeping expressions are Skinner1975 functions, which are valid only for Reynolds<1
#ReynoldsHere=0.5
#fieldForErratic=FV.OpenFOAMCylinderFlow(OpenFOAMDir=OpenFOAMDir, Reynolds=ReynoldsHere) #OpenFOAM Reynolds available are 1,10,47,50,100,115,180,239,300,486,600,1000
#if (np.size(fieldForErratic.nonFilled)>0): #When using OpenFOAM, check if the reading was done properly
#    print('Some sampled points in the mesh were not read properly:')
 #   print('Take a look into the object.nonFilled array')
 #   exit()
#fieldForSmooth=FV.OpenFOAMCylinderFlow(OpenFOAMDir=OpenFOAMDir, Reynolds=1) #OpenFOAM Reynolds available are 1,10,47,50,100,115,180,239,300,486,600,1000
#if (np.size(fieldForSmooth.nonFilled)>0): #When using OpenFOAM, check if the reading was done properly
#    print('Some sampled points in the mesh were not read properly:')
#    print('Take a look into the object.nonFilled array')
#    exit()


#Set the conditions on the integrators and solve the problem
tEnd=1000.0
dt=0.01 #I suggest to use 0.005 in production runs

#-------------------------------------------------------------------------------------------------------------------------
#Defining the useful domain for this analysis
xLimitsTest=[-6.0, 1.0]
yLimitsTest=[-3.0, 3.0]



#Testing the Erratic swimmers
fieldHere=fieldForErratic
particleHere='ErraticSwimmer'
passiveHere='PassiveBrownian'
NparticlesPerCloudHere=100 #Should be 100 in production runs if possible
NsavedTrajectories=NparticlesPerCloudHere
#RotationalDiffusionHere=0.16/(1/0.095238095)
randomSeed0=1000
#SemiDiametersHere=[0.0100000000000000, 0.0050000000000000, 0.0050000000000000] #This is related to the radius
meanParticleDiameter=np.mean(SemiDiametersHere)*2
meanunround="%.10f" % meanParticleDiameter
print('Mean particle diameter=', meanunround)
xLimitsHere=xLimitsTest
yLimitsHere=yLimitsTest


#Initial parameters for the search
Dc=1
ExtendedFactorIni=100
ExtendedFactorLast=200
YExtendedSeedLast=PassiveEdge[iRey]*ExtendedFactorLast
YExtendedSeedIni=PassiveEdge[iRey]*ExtendedFactorIni #Starting from the Round2 Final_Run
NPoints=192 #ideally 25 or more for production   
keepGoing=True
dYExtendedSeed=(YExtendedSeedLast-YExtendedSeedIni)/(NPoints-1) 
YExtendedSeed=np.arange(YExtendedSeedIni,YExtendedSeedLast+dYExtendedSeed/2,dYExtendedSeed)
XExtendedSeed=XSeedReady[iRey]
NCloudsExtended=np.size(YExtendedSeed)
print('The number of extended initial positions are:', NCloudsExtended)
print('The passive edge is YedgePassive=', PassiveEdge[iRey])
print('The swimmer edge to try is YSeedLast=', YExtendedSeedLast)
#------------------------------
#Cycle for processing seeding points
#New for parallel. Preparation of the input for the new function calcTheCloud which contains everythin inside the original cycle
inputs = [];
for i in range(0,NCloudsExtended):
    inputs.append((i))
    
#Two New lines for the definition a function that will be operated in parallel instead of the original serial for loop,
def calcTheCloud(iAndOthers):
    i = iAndOthers

#Original line defining the serial for loop,
###for i in range(0, NCloudsExtended): 

#The content of operations to perform
    cloudNameI=fieldHere.fieldType+'Re'+str(fieldHere.Reynolds)+'_'+particleHere+'_'+SwimmerTypeHere+'_'+str(meanunround)+'_'+str(i)+'Extended'+str(ExtendedFactorLast)
    print('Solving For Y=',YExtendedSeed[i], 'cloudName=',  cloudNameI)
    cloudRealI=CL.generalCloud(
                    cloudName=cloudNameI,
                    Nparticles=NparticlesPerCloudHere,
                    particleType=particleHere, 
                    Ntrajectories=NsavedTrajectories,
                    fluidField=fieldHere,
                    resultsDir=resultsDirHere, 
                    saveResults=True, #If =False, then no saving to disk will be performed
                    overwrite=True, #If =False, then the cloud will not perform any execution if results for the same cloud already exist
                    randomSeed=randomSeed0+i, #If =0, then the result will be super random and not repeatible when rerunning
                    time=0, 
                    position=[XExtendedSeed,  YExtendedSeed[i],  0], 
                    velocity=[0,  0,  0], 
                    semiDiameters=SemiDiametersHere, #The ellipsoid characteristics now are given with the semiDiameters in the three axis of the local(body) coordinate system
                    eulerAngles=[0,  0,  0], #Only the first particle in the cloud will have this angles, the rest will have random angles
                    eulerAngularVel=[0,  0 , 0], 
                    integratorType='dopri5', 
                    willFreeze=True,
                    xLimits=xLimitsHere, #Useful limits of the domain in X. Run will stop for any specific particle if it goes out of this limits
                    yLimits=yLimitsHere, #Useful limits of the domain in Y. Run will stop for any specific particle if it goes out of this limits
                    rotationalDiff=RotationalDiffusionHere,
                    swimmingSpeed=SwimSpeedHere, 
                    meanRunTime=RunDurationHere, 
                    tumbleType=SwimmerTypeHere,
                    meanTumbleAngle=meanTumbleAngleHere, 
                    stdDevTumbleAngle=stdDevTumbleAngleHere,
                    meanReverseAngle=meanReverseAngleHere, 
                    stdDevReverseAngle=stdDevReverseAngleHere, 
                    meanFlickAngle=meanFlickAngleHere, 
                    stdDevFlickAngle=stdDevFlickAngleHere, 
                    reversePct=reversePctHere
                    ) 
    time0=time.time()
    cloudRealI.integrateCloud(tEnd, dt)
    timeDone=time.time()-time0
    print('Time for cloud i=', i, 'at YSeed=', YExtendedSeed[i], 'was t=', timeDone)
#End of the for i in range(0,NCloudsExtended) .OR. def calcTheCloud function

#Declaring the number of parallel workers
num_of_workers = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_of_workers)

#Operating the calcTheCloud function in parallel
for data in pool.map(calcTheCloud, inputs):
    i=data
           
#End of script
print("Script done")
