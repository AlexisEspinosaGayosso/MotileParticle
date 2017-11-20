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

#Defining functions
def readFrozenRatio(cloudName, resultsDir):    
    cloudDir=os.path.join(resultsDir, cloudName)
    initialFinalDir=os.path.join(cloudDir,'InitialFinal')
    fileName=os.path.join(initialFinalDir,'InitialFinal.dat')
    M=np.loadtxt(fileName, ndmin=2)
    shape=np.shape(M)
    NParticles=float(np.size(M[:, 0]))
    NFrozen=float(np.sum(M[:, -1])) #The last column in the InitialFinal matrix is the flag for Frozen
    return NFrozen/NParticles

#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#Directory for the OpenFOAM sampled velocity fields
OpenFOAMDir=os.path.join('..','..', 'OpenFOAMFields') #Clearly check where are the OpenFOAMFields with respect to the running directory
print('OpenFOAM Directory is:')
print(OpenFOAMDir)

#resultsDirHere='E:\Results' #Change the directory to wherever you will save the results
#resultsDirHere='/media/sf_WIN_K/Blake_Testing/Results'
resultsDirHere='/scratch/pawsey0106/espinosa/Python/ResultsPassive'

#ReynoldsArray=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ReynoldsArray=[ 0.6, 0.7, 0.8, 0.9, 1.0]
NRey=np.size(ReynoldsArray)
YEdge=5E-5
dYSeed=0.1*YEdge
for jj in range(0, NRey):    
    ReynoldsHere=ReynoldsArray[jj]
    print('Solving for Re=', ReynoldsHere)
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

    #The dimensional stuff


    #Testing the Erratic swimmers
    fieldHere=fieldForErratic
    particleHere='ErraticSwimmer'
    passiveHere='PassiveBrownian'
    NparticlesPerCloudHere=10 #Should be 100 in production runs if possible
    NsavedTrajectories=1
    rotationalDiffHere=0.16/(1/0.095238095)
    swimmingSpeedHere=0.028571429
    meanRunTimeHere=(1/0.095238095)
    randomSeed0=1000
    semiDiametersHere=[0.0100000000000000, 0.0050000000000000, 0.0050000000000000] #This is related to the radius
    meanParticleDiameter=np.mean(semiDiametersHere)*2
    meanunround="%.10f" % meanParticleDiameter
    print('Mean particle diameter=', meanunround)
    xLimitsHere=xLimitsTest
    yLimitsHere=yLimitsTest

                        
    ##Defining the seeding zone
    #PassiveEfficiencyEstimate=0.000044426#It is better to get this estimate from the Moody diagram of Espinosa2012,2013  using Re and rp, where rp=meanParticleDiameter in this case
    #NDivPerEff=5 #Ideally 10 for production runs
    #dYSeed=(PassiveEfficiencyEstimate/2)/NDivPerEff #Separation in Y for the intial seeding
    #YSeedIni=0.0000 #Initial position in Y of the seeding
    #YSeedLast=10*(PassiveEfficiencyEstimate/2) #Ideally should be 10 times Final position in Y for the seeding
    #YSeed=np.arange(YSeedIni, YSeedLast+dYSeed/2, dYSeed) #the dYSeed/2 allows for YSeedLast to be included in the array
    #XSeed=-5 #X position of the initial seeding. Should be at least -10 in production runs'''
    #
    #    #YSeed=YLocalSeed
    #    
    #NClouds=np.size(YSeed)
    #print('The number of initial positions are:', NClouds)
       




    #for i in range(0, NClouds): 
    #    cloudNameI=fieldHere.fieldType+'Re'+str(fieldHere.Reynolds)+'_'+particleHere+str(meanunround)+'_'+str(i)
    #    print('Solving For Y=',YSeed[i], 'cloudName=',  cloudNameI)
    #    cloudRealI=CL.generalCloud(
    #                    cloudName=cloudNameI,
    #                    Nparticles=NparticlesPerCloudHere,
    #                    particleType=particleHere, 
    #                    Ntrajectories=NsavedTrajectories,
    #                    fluidField=fieldHere,
    #                    resultsDir=resultsDirHere, 
    #                    saveResults=True, #If =False, then no saving to disk will be performed
    #                    overwrite=True, #If =False, then the cloud will not perform any execution if results for the same cloud already exist
    #                    randomSeed=randomSeed0+i, #If =0, then the result will be super random and not repeatible when rerunning
    #                    time=0, 
    #                    position=[XSeed,  YSeed[i],  0], 
    #                    velocity=[0,  0,  0], 
    #                    semiDiameters=semiDiametersHere, #The ellipsoid characteristics now are given with the semiDiameters in the three axis of the local(body) coordinate system
    #                    eulerAngles=[0,  0,  0], #Only the first particle in the cloud will have this angles, the rest will have random angles
    #                    eulerAngularVel=[0,  0 , 0], 
    #                    integratorType='dopri5', 
    #                    willFreeze=True,
    #                    xLimits=xLimitsHere, #Useful limits of the domain in X. Run will stop for any specific particle if it goes out of this limits
    #                    yLimits=yLimitsHere, #Useful limits of the domain in Y. Run will stop for any specific particle if it goes out of this limits
    #                    rotationalDiff=rotationalDiffHere,
    #                    swimmingSpeed=0.028571429, #Idealy these parameters and below should be defined above and then input here
    #                    meanRunTime=1/0.095238095,
    #                    tumbleType='Tumble',
    #                    meanTumbleAngle=60/180*np.pi, 
    #                    stdDevTumbleAngle=10/180*np.pi,
    #                    meanReverseAngle=180/180*np.pi, 
    #                    stdDevReverseAngle=10/180*np.pi, 
    #                    meanFlickAngle=90/180*np.pi, 
    #                    stdDevFlickAngle=10/180*np.pi, 
    #                    reversePct=0.33
    #                    )
    #    cloudRealI.integrateCloud(tEnd, dt)
        





    #Initial parameters for the search
    Dc=1
    YSeedLastMax=2*YEdge+dYSeed
    YSeedLast=YSeedLastMax
    YSeedIni=0.0
    NTests=16 #ideally 25 or more for production   
    keepGoing=True
    counter=0
    counterMax=10
    while (keepGoing):
        dYSeed=(YSeedLast-YSeedIni)/(NTests-1) 
        YSeed=np.arange(YSeedIni,YSeedLast+dYSeed/2,dYSeed)
        if (ReynoldsHere<=0.5):
            XSeed=-5
        else:
            XSeed=-2.5
        NClouds=np.size(YSeed)
        print('The number of initial positions are:', NClouds)
        #------------------------------
        #Cycle for processing seeding points
        #New for parallel. Preparation of the input for the new function calcTheCloud which contains everythin inside the original cycle
        inputs = [];
        for i in range(0,NClouds):
            inputs.append((i))
            
        #Two New lines for the definition a function that will be operated in parallel instead of the original serial for loop,
        def calcTheCloud(iAndOthers):
            i = iAndOthers

        #Original line defining the serial for loop,
        ####for i in range(0, NClouds): 

        #The content of operations to perform
            #Solving for the passive particle, just 1 particle per cloud
            passiveNameI=fieldHere.fieldType+'Re'+str(fieldHere.Reynolds)+'_'+passiveHere+str(meanunround)+'_'+str(i)
            print('Solving For Y=',YSeed[i], 'cloudName=',  passiveNameI)
            passiveRealI=CL.generalCloud(
                            cloudName=passiveNameI,
                            Nparticles=1,
                            particleType=passiveHere, 
                            Ntrajectories=NsavedTrajectories,
                            fluidField=fieldHere,
                            resultsDir=resultsDirHere, 
                            saveResults=True, #If =False, then no saving to disk will be performed
                            overwrite=True, #If =False, then the cloud will not perform any execution if results for the same cloud already exist
                            randomSeed=randomSeed0+i, #If =0, then the result will be super random and not repeatible when rerunning
                            time=0, 
                            position=[XSeed,  YSeed[i],  0], 
                            velocity=[0,  0,  0], 
                            semiDiameters=semiDiametersHere, #The ellipsoid characteristics now are given with the semiDiameters in the three axis of the local(body) coordinate system
                            eulerAngles=[0,  0,  0], #Only the first particle in the cloud will have this angles, the rest will have random angles
                            eulerAngularVel=[0,  0 , 0], 
                            integratorType='dopri5', 
                            willFreeze=True,
                            xLimits=xLimitsHere, #Useful limits of the domain in X. Run will stop for any specific particle if it goes out of this limits
                            yLimits=yLimitsHere, #Useful limits of the domain in Y. Run will stop for any specific particle if it goes out of this limits
                            rotationalDiff=rotationalDiffHere)
            passiveRealI.integrateCloud(tEnd, dt)
        #End of the for i in range(0,NClouds) .OR. def calcTheCloud function

        #Declaring the number of parallel workers
        num_of_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_of_workers)

        #Operating the calcTheCloud function in parallel
        time0=time.time()
        for data in pool.map(calcTheCloud, inputs):
            i=data
        timeDone=time.time()-time0
        print('Time for solving all the seeding clouds was:', timeDone, '[s]')
        #Last things to do
        #Reding the FinalPct
        FrozenPctPassive=np.zeros(np.shape(YSeed), dtype=np.dtype('d'))
        edgeSeed=-1
        for seed in range(0, NClouds): 
            passiveNameHere=fieldHere.fieldType+'Re'+str(fieldHere.Reynolds)+'_'+passiveHere+str(meanunround)+'_'+str(seed)
            FrozenPctPassive[seed]=readFrozenRatio(cloudName=passiveNameHere, resultsDir=resultsDirHere)
            if (FrozenPctPassive[seed]>0):
                edgeSeed=seed            
        print(FrozenPctPassive)
        print('Big While loop in counter=', counter)
        if (edgeSeed==NClouds-1):
            print('No clear edge found, Extending the edge further')
            YSeedIni=YSeedLast
            YSeedLast=YSeedLast*2
        elif (edgeSeed==-1):
            print('Nothing was Frozen, stopping the Big While Loop')
            keepGoing=False    
        else:
            print('A clear edge was found at Yedge=', YSeed[edgeSeed],', now trying to refine in the neightborhood')
            YSeedIni=YSeed[edgeSeed]
            YSeedLast=YSeed[edgeSeed+1]            
        counter=counter+1    
        if (counter>=counterMax):
            keepGoing=False
    #End of the Big While Loop
    fileName=os.path.join(resultsDirHere,'PassiveEdge_Re'+str(ReynoldsHere)+'.dat')
    YEdge=YSeed[edgeSeed]
    recordHere=np.array([edgeSeed, YEdge], dtype=np.dtype('d'))
    recordHere=np.reshape(recordHere,(1, 2))
    with open(fileName, 'ab') as file_handle:        
        np.savetxt(file_handle,recordHere)
    print('The edge was found at position i=',edgeSeed,'with is Yedge=', YSeed[edgeSeed])
#End for the superloop        
#End of script
print("Script done")
