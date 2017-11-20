"""
Module for defining the MPI variables as global variables
"""

#Defining environment with parallel utilities
#from joblib import Parallel, delayed
#import multiprocessing
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from mpi4py.MPI import ANY_TAG

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
print('Rank=', rank, 'With size=', size,'Hola Cabrones')
