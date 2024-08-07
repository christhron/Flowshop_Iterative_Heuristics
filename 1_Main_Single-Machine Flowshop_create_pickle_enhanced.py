# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:31:39 2023

This code uses the exact MILP algorithm to generate and store single-machine instances which can then be used to test other single-machine algorithms.  The pickle files in this folder were created using this program.

@author: matth
"""
from Functions.FlowshopProblemClassDefs import *
from Functions.FlowshopProblemCreationRoutines_v1 import *
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import pickle
unif = np.random.uniform
geom = np.random.geometric 


pickleFilename = 'Output/single_machine_6N.pickle'
writeToPickle=True
#Really big number for lateness indicator "on/off"
cBig=10000
# Number of instances generated
nInstances=12000

pctDev = 0.25 # Percent deviation in parameters.
geomProb = 0.6 # Percent of time there is backlog of more than 1 job

N_jobs,M_stage,K_sen,mu_Ar,mu_Ai,mu_Dint=6,1,1,5,1E10,10
N = N_jobs# legacy variable
    #Flat rate:
p=50
    #Rate penalty:
q=1
   
# Parameters for randomly generating a scheduling scenario
Mu_mu_Xm=np.ones((1,M_stage))*5
Mu_sigma_Xm=np.ones((1,M_stage))*0.0001 # Remove uncertainty
Mu_mu_Zm=np.ones((1,M_stage))*5# 
Mu_sigma_Zm=np.ones((1,M_stage))*0.0001 # Remove uncertainty
uncertainFlag=False


def flow_1(A,D,X,P,Q):
  A= A.reshape(D.shape) #arrival times
  X= X.reshape(D.shape)
  D= D #due times
  #X= X.reshape(D.shape) #processing times
  #P= lateness penalty
  #Q= lateness charge rate
  N=len(A) #Number of jobs
  if type(P)!="numpy.ndarray":
    P=P+0*A
    Q=Q+0*A
  # id=np.identity(N)

  # U_j=np.tile(id,(1,N)) #This corresponds to the first column ones and the rest zeroes and so forth (column n for row n); N*N^2
  # U_k=np.repeat(id,N).reshape((N,N*N)) #This corresponds to the first row ones and the rest zeroes and so forth (row n for row n); N*N^2

  #Linear Programming Portion
# Objective function: (p_1,...,p_N,q_1,...,q_n,0,...) since lateness is all that affects the cost
# In this order: N*N: s_jk N*N: u_jk 2N: v_k & w_k    WHERE:
# s_k is the start time of job in slot k, u_nk is 1 if job n is in slot k and 0 otherwise, and v_n is 1 & w_n is how late if job n is late and 0 if not
  numvar=N*N + N
  c=np.zeros(numvar+2*N)
  c[N*N + N:N*N + 2*N]=p
  c[N*N + 2*N:]=q
  
#-----Equality Constraints-----
#Constraint 1: Matching sums are 1
  A_eq=np.zeros([2*N,numvar+2*N])
  for j in range(N):
     A_eq[j, N + j:numvar:N]=1  #Sum over n for slot j (matching variables)
     A_eq[N+j, N + j*N:N + (j+1)*N]=1  #Sum over j for job n (matching variables)
  b_eq=np.ones(2*N)
  
#-----Inequality Constraints-----
  A_up=np.zeros([5*N-1,numvar]) #add w & v in final array       ################### 2 & 3 accounted for
  b_up=np.zeros(5*N-1)                                                              ###################
  lates=np.zeros([5*N-1,2*N])                                     

  for j in range(N):
    #Constraint 2: Lateness value (N total: 2*N:3*N)
    lates[j, N + j] = -1
    A_up[j, j] = 1
    A_up[j, j+N:numvar:N] = X-D
    
    #Constraint 3: Lateness greater than or equal to 0 (N total: 0:N)
    lates[N+j,j]=-1
    
    #Constraint 4: Lateness indicator (N total: N:2*N)
    lates[2*N+j,N+j]=1
    lates[2*N+j,j]=-1*cBig
    
    #Constraint 5: Start is greater than or equal to arrival (N total: 3*N:4*N)
    A_up[3*N+j, j] = -1  
    A_up[3*N+j, N+j:numvar:N] = A
    
    #Constraint 7: start k < start k+1 (N-1 total: 4*N:5*N-1)
    if j<N-1:
        A_up[4*N+j, j]=1
        A_up[4*N+j, j+1]=-1
        A_up[4*N+j, N+j:numvar:N] = X

  A_up=np.append(A_up, lates, axis=1)


  integ=np.zeros(numvar+2*N)
  integ[N:numvar+N]=1

  res=linprog(c, A_ub=A_up, b_ub=b_up, A_eq=A_eq, b_eq=b_eq, integrality=integ, options={"disp":False})
  # print(res)
  return res


# Set up classes for parameters
params=Params(M_stage,N_jobs,K_sen,mu_Ar,mu_Ai,mu_Dint,Mu_mu_Xm,Mu_sigma_Xm,Mu_mu_Zm,Mu_sigma_Zm)
# Temporary params
paramsTmp=Params(M_stage,N_jobs,K_sen,mu_Ar,mu_Ai,mu_Dint,Mu_mu_Xm,Mu_sigma_Xm,Mu_mu_Zm,Mu_sigma_Zm)


# Save to existing pickle file
# Note only works for 1 scenario
with open(pickleFilename, 'wb') as f:
    for i in range(nInstances): 
        if i>-1:
            print(i)
        #print("Instance no. ",i+1)
        paramsTmp.Mu_mu_Xm = params.Mu_mu_Xm*(1 + unif(-pctDev,pctDev))
        paramsTmp.Mu_mu_Zm = params.Mu_mu_Zm*(1 + unif(-pctDev,pctDev))
        paramsTmp.mu_Ar = params.mu_Ar*(1 + unif(-pctDev,pctDev))
        paramsTmp.mu_Dint = params.mu_Dint*(1 + unif(-pctDev,pctDev))               
        
        [Vec,indVec,mu_X,sigma_X,mu_Z,sigma_Z]=Create_inst(paramsTmp, uncertainFlag)    
        
        # Push back arrival so that more than one is ready at time 0.
        Vec.A[:,0] = Vec.A[:,0] - Vec.A[min(geom(geomProb),N_jobs)-1]
 
        A = Vec.A[:,0]; D=Vec.D[:,0]; X=Vec.X[:,:,0]; 
        if (A[0]<0):
            A=A-A[0]
            D=D-A[0]
        res=flow_1(A,D,X,p,q)
        
        # Job Orders (Integers, Sorted)
        matchMx = res['x'][-N*N-2*N:-2*N].reshape(N,N)
        matchMx[matchMx > .9] = 1
        matchMx[matchMx <.9] = 0
        orders= np.arange(N) @ matchMx 
        # print(orders)
    
        instance={'A':A,'X':X,'D':D,'order':orders, 'p':p, 'q':q}
   
        if writeToPickle:
            pickle.dump(instance, f)