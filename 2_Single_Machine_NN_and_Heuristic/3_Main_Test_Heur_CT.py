# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 08:51:09 2023
Compares single-machine algorithms'  performance


@author: matth
"""

import pickle
import numpy as np
import statistics as st
from matplotlib import pyplot as plt
from Functions.FlowshopBaseSchedRoutines import Obj_pq
from Functions.InsertionHeuristicFunctions import insertHeurFn
from Functions.SelectionHeuristicFunctions import selectHeurFn

 #plt.style.use('seaborn-deep')


# Paths to training/validation data and testing data respectively
dataFilePath = "Data/single_machine_8N.pickle"


# Number of jobs
N=8


# Lateness for objective function: penalty & rate
p=50
q=1

        
### Compute the start and finish times based on system parameters and the given job order (simplified to only do 1 machine)
## @@@ should move this to base sched routines.
def CalcYS_single(m0,job,Am0,X,Z):
    N,M=X.shape
    #Starting time(S) and finishing time(Y)
    S=np.empty((N,M))
    Y=np.empty((N,M))
    Idle =np.empty((N,M))

    j=int(job[0,m0])
    # First Stage:
    S[j,m0]=Am0[j]+Z[j,m0]
    Y[j,m0]=S[j,m0]+X[j,m0]
    Idle[j,m0] = 0

    for jn in range(1,N):
        j=int(job[jn,m0])
        S[j,m0]=max(Am0[j]+Z[j,m0],Y[int(job[jn-1,m0]),m0])
        Y[j,m0]=S[j,m0]+X[j,m0]
        Idle[j,m0] = S[j,m0]-Y[int(job[jn-1,m0]),m0]
        
    return Y,S,Idle

all_arrival=[]
all_work=[]
all_due=[]
all_labels=[]
# Save scale factors(to compute objective function)
all_scale = []
with open(dataFilePath, 'rb') as f:
    while True:
        try:
            data = pickle.load(f)
            #scale per instance so max due  time =1
            scale = 1#np.max(data['D'])
            all_scale.append(scale)
            all_arrival.append(data['A']/scale) 
            all_work.append(data['X']/scale)
            all_due.append(data['D']/scale)
            label=np.zeros([1,N])
            label[0]=data['order']
           # label[1]=scale
            all_labels.append(label)
        except EOFError:
            # Reached the end of the file
            break
all_labels=np.squeeze(all_labels, axis=1)


# Count of accurate objective predictions and arrays for objective values of NN, MILP, and random for comparison
count=0
n_test = len(all_labels)
#obj=np.zeros(n_test)  #NN
objr=np.zeros(n_test) #MILP
objRand=np.zeros(n_test) #Random Order
objOrder=np.zeros(n_test) #In Order
objHeur=np.zeros(n_test) #Heuristic
for i in range(n_test):
        print("Test case",i)
        scale=1 #y_test[i][1,0]
        A=all_arrival[i].reshape([1,N])#(scale*x_test[i][0]).reshape([N,1])
        X=all_work[i].reshape([1,N])
        D=all_due[i].reshape([1,N])#(scale*x_test[i][2])#.reshape([N,1])
        
        
        samp=np.append(np.append(A, X, axis=0), D, axis=0)
        D=D.flatten()
        
        

        # Prediction for greedy (in arrival order)
        # This can be used as baseline
        ordY, ordS, ordIdle=CalcYS_single(0, np.arange(N).reshape([N,1]), A.T, X.T, np.zeros([N,1]))

        # Prediction using heuristic
        HeurOrders, Heur_late_totals, Heur_fin_totals=selectHeurFn(A,D,X,p,q)
        heurY, heurS, heurIdle=CalcYS_single(0, HeurOrders.reshape([N,1]), A.T, X.T, np.zeros([N,1]))

        # obj[i]=Obj_pq(D,predY,p,q)
        # objr[i]=Obj_pq(D,realY,p,q)
        # objRand[i]=Obj_pq(D,randY,p,q)
        # objOrder[i]=Obj_pq(D,ordY,p,q)        
        objHeur[i]=Obj_pq(D.flatten(),heurY,p,q)

        
        # if (obj[i]==objr[i]):
        #     count+=1

fig, (ax1, ax2) = plt.subplots(nrows=2)
binw=25
Tmax = max(max(objr),max(objOrder),max(objHeur))
binsize=np.arange(0,Tmax+binw,binw)
#histtype='step', fill=True)
colors=["blue","purple",'green']
names=["MILP", "Heuristic","Greedy"]
ax1.hist([objr,objHeur,objOrder], bins=binsize, color=colors, label=names)#, stacked=False, density=True)
ax1.set_title('Penalty Distributions for ' + dataFilePath) 
ax1.legend(loc='best')
ax1.set_xlabel("Penalty")
ax1.set_ylabel("Num. Observations")
#ytick=np.linspace(Tmin,Tmax,int(len(objr)/10))
#ax1.set_yticks(ytick)
xtick=binsize[::2]
ax1.set_xticks(xtick)

colors2=colors[1:]
names=["Heuristic minus MILP", "Greedy minus MILP"]
binsize=np.arange(0,max(objOrder-objr)+binw,binw)
ax2.hist([objHeur-objr,objOrder-objr], bins=binsize, color=colors2, histtype='bar', stacked=False, fill=True, label=names)
ax2.set_title('Penalty differences for ' + dataFilePath) 
ax2.set_xlabel("Penalty Difference ")
ax2.set_ylabel("Num. Observations")
ax2.legend(loc='best')
xtick=binsize[::2]
ax2.set_xticks(xtick)
plt.subplots_adjust(left=0.2,
                    bottom=1,
                    right=0.9,
                    top=2.4,
                    wspace=0.4,
                    hspace=0.4)
# bin_centers = (binsize[:-1] + binsize[1:]) / 2
# ax2.bar(bin_centers - binw/4, np.histogram(obj - objr, bins=binsize)[0], width=binw/2, color="purple", alpha=0.5, label='obj-objr')
# ax2.bar(bin_centers + binw/4, np.histogram(objHeur - objr, bins=binsize)[0], width=binw/2, color="green", alpha=0.5, label='objHeur-objr')
plt.show()

print("Objective Accuracy for heuristic is: ", sum(objHeur<=objr)/n_test*100,"%")
print("Max Objective Difference is: ", max(objHeur-objr))
print("Average Objective Difference: ", np.mean(objHeur-objr), " Compared to Random's: ", np.mean(objRand-objr))
print("Standard deviation of Objective Difference is: ", st.stdev(objHeur-objr), " Compared to Random's: ", st.stdev(objRand-objr))


# In order versus NN
# Vec=objOrder-obj
# pltVec=np.sort(Vec)
# plt.plot(pltVec)

#Eq=count(np.where(objOrder-obj==0))

# Sorted: exact
pltObj=np.sort(objr)
y_values = np.arange(len(pltObj)) / len(pltObj)
plt.plot(pltObj, y_values, label="MILP",color=colors[0])

# Sorted: Heuristic
pltObj=np.sort(objHeur)
y_values = np.arange(len(pltObj)) / len(pltObj)
plt.plot(pltObj, y_values, label="Heur.",color=colors[1])

# Sorted: Greedy
pltObj=np.sort(objOrder)
y_values = np.arange(len(pltObj)) / len(pltObj)
plt.plot(pltObj, y_values, label="Greedy",color=colors[2])


plt.title("Distribution of objective functions")
plt.legend()
    #ax2.set_xlim(xmin=-5,xmax=5)
    #ax2.set_ylim(ymin=0,ymax=10)
    #plt.xlabel("Percent Difference (Bin Size=" + str(binw) + ")")
plt.ylabel("Proportion of Jobs")
plt.xlabel("Objective Value")

plt.show()
