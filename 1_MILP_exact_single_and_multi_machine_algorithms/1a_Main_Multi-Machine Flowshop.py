# -*- coding: utf-8 -*-
"""
This code finds the exact MILP solution for the multi-machine problem.  It runs very slowly.
In the current version, the actual solution is not saved.  Only the objective function is saved.


@author: matth
"""
from Functions.FlowshopProblemClassDefs import *
from Functions.FlowshopProblemCreationRoutines_v1 import *
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

output_filename="MILPs_4N3M_start.pkl"
save_this_scenario=True
nInstances= 150

N_jobs,M_stage,K_sen,mu_Ar,mu_Ai,mu_Dint=4,3,1,5,1E10,10
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

# Initialize arrays
Asave=[]
Dsave=[]
Xsave=[]
Zsave=[]
Tsave=[]
Ressave=[]

# with open(output_filename, 'rb') as f:
#     Asave=pickle.load(f)
#     Dsave=pickle.load(f)
#     Xsave=pickle.load(f)
#     Zsave=pickle.load(f)
#     Tsave=pickle.load(f)
#     Ressave=pickle.load(f)

############ Gantt Chart ######################
#A arrival times,Z=waiting times,S=starting times,Y = finishing times, D = due times
def plotSched(A,Z,S,Y,D):
    # Number of jobs, machines
    N,M=Z.shape
    W = np.zeros(Z.shape)
    ig,ax=plt.subplots()
    # Define a specific color for each job
    color = plt.cm.rainbow(np.linspace(0, 1, N))    
    
    # Plot initial arrival and waiting times
    for jn in range(N):
        yVal = M+1 - 2*(jn+1)/(3*N)-1/(3*N)
        W[jn,0] = A[jn]+Z[jn,0]
        esp=(int)(W[jn,0]-A[jn])+1
        ax.scatter(np.linspace(A[jn],W[jn,0],esp),\
         np.linspace(yVal,yVal,esp),marker='.',color=color[jn],\
             linewidth=0.1)
    # Plot lines for machines, machine by machine.
    ind=0
    labl=[]
    for jm in range(M):
        for jn in range(N):
            # Job interval on machine
            if (ind<N):
                ax.plot([S[jn,jm],Y[jn,jm]],[M-jm,M-jm],color=color[jn],linewidth=5,\
                         label='job'+str(ind))
                ind+=1
            else:
                ax.plot([S[jn,jm],Y[jn,jm]],[M-jm,M-jm],color=color[jn],linewidth=5)
            # If start time = prev. availability time, then draw dotted line
            if S[jn,jm] == W[jn,jm]:
                yVal = M-jm+1  - 2*(jn+1)/(3*N)-1/(3*N)  # heights are staggered so different weighting times are visible.
                ax.plot([S[jn,jm],S[jn,jm]],[yVal,M-jm],'c:')

            # Plot waiting time
            if jm<M-1:
                yVal = M-jm  - 2*(jn+1)/(3*N)-1/(3*N)   # heights are staggered so different weighting times are visible.
                ax.plot([Y[jn,jm],Y[jn,jm]],[M-jm,yVal],'k:')
                W[jn,jm+1] = Y[jn,jm]+Z[jn,jm+1]
                esp=(int)(W[jn,jm+1]-Y[jn,jm])
                ax.plot(np.linspace(Y[jn,jm],W[jn,jm+1],esp),\
                      np.linspace(yVal,yVal,esp),color=color[jn],solid_capstyle = 'round',\
                          linewidth=2)
        labl.append(M-jm)
    for jn in range(N):
        ax.scatter(D[jn],1/(3*N),marker='x',color=color[jn])
    plt.xlabel('time')
    ax.set_yticks(labl)
    ax.set_yticklabels(range(len(labl)))
    plt.ylabel('machine')
    plt.title('gantt chart')
    plt.legend()



def flow_2(A,D,X,Z,P,Q):
  A= A.reshape(D.shape) #arrival times
  D= D #due times
 #------------------------------------------- X= X.reshape(D.shape) #processing times
  #P= lateness penalty
  #Q= lateness charge rate
  N=len(A) #Number of jobs
  M=X.shape[1] #Number of machines
  if type(P)!="numpy.ndarray":
    P=P+0*A
    Q=Q+0*A

  #Linear Programming Portion

#Objective function: (p_1,...,p_N,q_1,...,q_n,0,...) since lateness is all that affects the cost
#In this order: N*N*M: s_ijk N*N*M: u_jk 2N: v_k & w_k    WHERE:
#s_ijk is the start time of job i in spot j of machine k, u_ijk is 1 if job i is in position j on machine k and 0 otherwise, and v_k is 1 & w_k is how late if job k is late and 0 if not
  numvar=2*N*N*M
  c=np.zeros(numvar+2*N)
  c[2*N*N*M:2*N*N*M+N]=p
  c[2*N*N*M+N:]=q

#-----Equality Constraints-----
#Constraint 1: Matching sums are 1
  A_eq=np.zeros([2*N*M,numvar+2*N])
  for i in range(M):
    for j in range(N):
      A_eq[M*N+i*N+j, N*N*M+i*N*N+j:N*N*M+(i+1)*N*N:N]=1  #Sum over n for slot j on machine i (matching variables)
      A_eq[i*N+j, N*N*M+i*N*N+j*N:N*N*M+i*N*N+(j+1)*N]=1  #Sum over j for job n on machine i (matching variables)
    b_eq=np.ones(2*N*M)

#-----Inequality Constraints-----
  A_up=np.zeros([N*((2+2*N)*M+3+M*N),numvar]) #add w & v in final array
  b_up=np.zeros(N*((2+2*N)*M+3+M*N))  #Accounts for constraints 2-8
  lates=np.zeros([N*((2+2*N)*M+3+M*N),2*N])

  for j in range(N):
    #Constraint 2: Lateness greater than or equal to 0 (N total: 0:N)
    lates[j,N+j]=-1
    #Constraint 3: Lateness indicator (N total: N:2*N)
    lates[N+j,N+j]=1
    lates[N+j,j]=-1000
    #Constraint 4: Lateness value (N total: 2*N:3*N)
    lates[2*N+j,N+j]=-1
    for i in range(M):
      A_up[2*N+ j, (M-1)*N*N+j:M*N*N:N]=1
      A_up[2*N+ j, M*N*N+ (M-1)*N*N+j:2*M*N*N:N]=X[:,M-1]-D  #Sum over n for slot j on machine i (matching variables)
      #Constraint 8: Start is greater than or equal to 0 (M*N*N total: N*(2*M+3)-M:N*((2+N)*M+3)-M)
      for k in range(N):
        A_up[N*(2*M+3)-M+i*N*N+j*N+k,i*N*N+j*N+k]=-1
        #Constraint 9: s_nki>0 if u_nki>0 (M*N*N total: N*((2+N)*M+3)-M:N*((2+2*N)*M+3+M*N))
        A_up[N*((2+N)*M+3)-M+i*N*N+j*N+k,i*N*N+j*N+k]=1
        A_up[N*((2+N)*M+3)-M+i*N*N+j*N+k,M*N*N+i*N*N+j*N+k]=-10000

    #Constraint 5: First start time; arrival + wait 1 <= first start (N total: 3*N:4*N)
    A_up[3*N+ j, j*N:(j+1)*N]=-1
    b_up[3*N+j]=-1*(A[j]+Z[j,0])

  #Constraint 6: Other starts; (N*(M-1) total: 4*N:N*(M+3))
  for i in range(1,M):
      startT=X[:,i-1] + Z[:,i]
      for j in range(N):
        A_up[3*N+ i*N + j, (i-1)*N*N+j*N:(i-1)*N*N+(j+1)*N]=1                        #The previous start time for job j is included: (i-1)*N*N ensures we begin on machine i, j*N ensures job j is the one changed
        A_up[3*N+ i*N + j, i*N*N+j*N:i*N*N+(j+1)*N]=-1               #The next start time for job j is subtracted: i*N*N ensures the next machine
        b_up[3*N+ i*N + j]=-1*startT[j]    #The matching variable is used to add the work & wait times: M*N*N skips the start times, i*N*N gets the correct machine, j*N gives the job row

  #Constraint 7: start k + work k + wait k+1 < start k+1 (M*(N-1) total: N*(M+3):N*(2*M+3)-M)
  for i in range(M):
      for j in range(N-1):
        A_up[N*(M+3)+ i*(N-1)+ j, j+i*N*N:(i+1)*N*N:N]=1
        A_up[N*(M+3)+ i*(N-1)+ j, j+i*N*N+1:(i+1)*N*N:N]=-1
        A_up[N*(M+3)+ i*(N-1)+ j, M*N*N+i*N*N+j:M*N*N+(i+1)*N*N:N]=X[:,i]
        
        

  A_up=np.append(A_up, lates, axis=1)
  integ=np.zeros(numvar+2*N)
  integ[N*N*M:numvar+N]=1 # bounds=(0,1),

  res=linprog(c, A_ub=A_up, b_ub=b_up, A_eq=A_eq, b_eq=b_eq, integrality=integ, options={"disp":False})
  print(res)
  return res

for count in range(len(Asave), nInstances):
    start=time.time()
    # Set up classes for parameters
    params=Params(M_stage,N_jobs,K_sen,mu_Ar,mu_Ai,mu_Dint,Mu_mu_Xm,Mu_sigma_Xm,Mu_mu_Zm,Mu_sigma_Zm)
    
    
    [Vec,indVec,mu_X,sigma_X,mu_Z,sigma_Z]=Create_inst(params, uncertainFlag)    
    As = Vec.A[:,0]; Ds=Vec.D[:,0]; Xs=Vec.X[:,:,0]; Zs=Vec.Z[:,:,0] 
    
    # res=flow_2(As,Ds,Xs,Zs,p,q)
    # total=time.time()-start
    # print("This took: ", total/60, " minutes")
    
    Asave.append(As)
    Dsave.append(Ds)
    Xsave.append(Xs)
    Zsave.append(Zs)
    # Tsave.append(total)
    # Ressave.append(res["fun"])

    if save_this_scenario:
            with open(output_filename, 'wb') as f:
                pickle.dump(Asave, f)
                pickle.dump(Dsave, f)
                pickle.dump(Xsave, f)
                pickle.dump(Zsave, f)
                # pickle.dump(Tsave, f)
                # pickle.dump(Ressave, f)

# answers=np.zeros((2*N_jobs*M_stage+2,N_jobs))
# for i in range(2*N_jobs*M_stage+2):
#   answers[i]=res['x'][i*N_jobs:(i+1)*N_jobs]
#  print(answers[i])

#print(Ds-res['x'][2*N_jobs*N_jobs*M_stage+N_jobs:]-np.sum(answers[N_jobs*(M_stage-1):N_jobs*M_stage], axis=0)-Xs[M_stage-1])

#In this order: N*N*M: s_ijk N*N*M: u_jk 2N: v_k & w_k    WHERE:
#s_ijk is the start time of job i in spot j of machine k, u_ijk is 1 if job i is in position j on machine k and 0 otherwise, and v_k is 1 & w_k is how late if job k is late and 0 if not
# def getS(Z,x,M,N):
#     S=np.zeros_like(Z)
#     for i in range(M):
#         for j in range(N):
#             row=x[i*N*N+j*N:i*N*N+(j+1)*N]
#             S[i,j]=np.max(row)
#     return S

# def getY(S,X,M,N):
#     Y=S+X
#     return Y
    
# S=getS(Zs,res['x'],M_stage,N_jobs)
# Y=getY(S,Xs,M_stage,N_jobs)
# plotSched(As,Zs.T,S.T,Y.T,Ds)
