# -*- coding: utf-8 -*-


import pickle
import numpy as np
import statistics as st
from matplotlib import pyplot as plt
from Functions.FlowshopBaseSchedRoutines import Obj_pq
import Functions.InsertionHeuristicFunctions as hv4
import Functions.Heuristics.Pareto_Heuristics as hSelect
import time
from math import *

# Tell the main which algorithm to use
algorithm = "Select" #Select or Insert
# True if importing the other algorithm for comparison
readOnly = False
# evalFn: Insert or Select based on algorithm
if algorithm=='Insert':
    evalFn=hv4.insertHeurLimInsFn
    saveFile='Output/timeVsNKEEP_50N_Insertion.pkl'
    numIns=np.array([10,15,20,30,35]) 
if algorithm=='Select':
    evalFn=hSelect.insertHeurLimIns_revisedFn
    saveFile='Output/timeVsNKEEP_50N_Selection.pkl'
    numIns=np.array([3,4,5,6,7]) 

# Path to data
dataFilePaths = ["Data/single_machine_50N_100int.pickle"] 


# Number of jobs; if importing multiple data files, there should be 1 for each
Ns=[50]
# Number of kept permutations to try
Ks=np.array([30,40,50,60,70,80])
# Lateness for objective function: penalty & rate
p=50
q=1

# Number of instances from each data file to use
nb_instances = [100]

###### Variable definition to allow for storage
#Kept permutations, heuristic parameter, instance
objHeur=np.zeros([len(Ks),len(numIns),nb_instances[0]])
total_runtimes=np.zeros([len(Ks),len(numIns),nb_instances[0]])

### Compute the start and finish times based on system parameters and the given job order (copied from MILP code, and simplified to only do 1 machine)
def CalcYS(m0,job,Am0,X,Z):
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
        
    return Y.flatten(),S,Idle


# Prepare each instance for use
for file in  range(len(dataFilePaths)):
    all_arrival=[]
    all_work=[]
    all_due=[]
    all_labels=[]
    # Save scale factors (to compute objective function)
    all_scale = []
    dataFilePath = dataFilePaths[file]
    N = Ns[file]
    with open(dataFilePath, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                scale = 1
                all_scale.append(scale)
                all_arrival.append(data['A']/scale) 
                all_work.append(data['X']/scale)
                all_due.append(data['D']/scale)
                label=np.zeros([1,N])
                label[0]=data['order']
                all_labels.append(label)
            except EOFError:
                break
    all_scale = np.array(all_scale)
    all_labels=np.squeeze(all_labels, axis=1)
    
    instance=np.zeros([3,N])
    inputs=np.zeros([len(all_arrival),3,N])
    for i in range(len(all_arrival)):
        instance[0]=all_arrival[i]
        instance[1]=all_work[i].flatten()
        instance[2]=all_due[i]
        
        inputs[i]=instance
    x_test,y_test = inputs, all_labels

# Main: generates algorithm statistics
for j in range(len(Ks)):
    N=Ns[0]
    # Change the number of kept permutations
    hv4.NKEEP=Ks[j]
    hSelect.NKEEP=Ks[j]
    # Count of accurate objective predictions and arrays for objective values of NN, MILP, and random for comparison
    count=0
    n_test = nb_instances[file]
    objr=np.zeros(n_test)

    # Best obj value
    objHeurMin = np.zeros(n_test)
    # Save the orders
    heurOrders=np.zeros([n_test,len(numIns),N])
    # Finishing times, used for objective calculation
    heurY=np.zeros([len(numIns),N])

    # Run each algorithm for each instance
    for i in range(nb_instances[0]):
            scale=1
            A=np.array(x_test[i,0]).reshape([1,N])
            X=np.array(scale*x_test[i][1]).reshape([1,N])
            D=np.array(x_test[i,2]).reshape([1,N])
            
            label= np.array(y_test[i]) 
            
            samp=np.append(np.append(A, X, axis=0), D, axis=0)
            D=D.flatten()
            
    
            # Prediction from saved order (in order or MILP)
            realY, realS, realIdle=CalcYS(0, label.reshape([N,1]), A.T, X.T, np.zeros([N,1]))

            # # Prediction from Heuristic
            for k in range(len(numIns)):
                    start_time = time.time()
                    heurOrders[i,k], Heur_late_totals10, Heur_fin_totals10 = evalFn(A,D,X,p,q,numIns[k])
                    end_time = time.time()
                    total_runtimes[j,k,i] = end_time - start_time
            
            # Get finishing times, start times, and idle times
            for k in range(len(numIns)):
                heurY[k], heurS, heurIdle=CalcYS(0, heurOrders[i,k].reshape([N,1]), A.T, X.T, np.zeros([N,1]))
                objHeur[j,k,i]=Obj_pq(D,heurY[k].reshape([N,1]),p,q)
            
    print(j)

# Plot each Kept to compare Obj vs Instance
# for run in range(5):
#     plt.figure(figsize=(20, 8))
#     for alg in range(len(numIns)):
#         instance_numbers = np.arange(nb_instances[0])
#         # Assuming `objective_values[alg]` has multiple runs, we will index it with `run`
#         plt.scatter(4*instance_numbers+alg*0.6, difference_vectors[run,alg], label=methods[alg])
#     plt.title("Performance by Instance for "+str(Ks[run])+" Kept Permutations")
#     plt.xlabel('Instance Number')
#     plt.ylabel('Objective Value')
#     plt.legend()
#     plt.show()

# # Calculate standard deviations

# std_devs = (((accuracys_vectors/100) * (1 - (accuracys_vectors/100))) / nb_instances[0]) *100

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 6))

# for i in range(accuracys_vectors.shape[0]):
#     # # Convert number of jobs to integers for proper spacing
#     # x = np.arange(1, len(names) + 1)

#     # Plot each line with error bars
#     ax.errorbar(10, accuracys_vectors[i], yerr=2*std_devs[i], fmt='o', color=colors[i], label=methods[i])

#     ax.plot(10, accuracys_vectors[i], marker='o', color=colors[i])

# ax.set_title('Objective Accuracy (%) for The best of dispatch rules using Pareto as exact algorithm   for ' + str(nb_instances[0]) + ' instances')
# ax.set_xlabel('Number of Jobs')
# ax.set_ylabel('Accuracy')
# #ax.set_xticks(x)
# #ax.set_xticklabels(names)
# ax.legend()
# # Set y-axis scale to logarithmic
# #ax.set_yscale('log')
# plt.tight_layout()
# plt.show()




# fig, ax = plt.subplots(figsize=(10, 6))

# for i in range(std_difference_vectors.shape[0] ):
#     # # Convert number of jobs to integers for proper spacing
#     # x = np.arange(1, len(names) + 1)

#     # Plot each line
#     ax.plot(10, std_difference_vectors[i], marker='o', color=colors[i], label=methods[i])

# ax.set_title('Standard deviation of Objective Difference for ' + str(nb_instances[0]) + ' instances')
# ax.set_xlabel('Number of Jobs')
# ax.set_ylabel('Standard deviation')
# #ax.set_xticks(x)
# #ax.set_xticklabels(names)
# ax.legend()
# # Set y-axis scale to logarithmic

# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(10, 6))

# for i in range(mean_difference_vectors.shape[0]  ):
#     # # Convert number of jobs to integers for proper spacing
#     # x = np.arange(1, len(names) + 1)

#     # Plot each line
#     ax.plot(10, mean_difference_vectors[i], marker='o', color=colors[i], label=methods[i])

# ax.set_title('Average Objective Difference (%) for ' + str(nb_instances[0]) + ' instances')
# ax.set_xlabel('Number of Jobs')
# ax.set_ylabel('Average Objective Difference (%)')
# #ax.set_xticks(x)
# #ax.set_xticklabels(names)
# ax.legend()
# #ax.set_yscale('log')
# plt.tight_layout()
# plt.show()




# # percentiles = [90, 95, 99]
# # linestyles = ['-', '--', ':']
# # fig, ax = plt.subplots(figsize=(10, 6))
# # for method_idx in range(len(methods) ):
# #     data_method = difference_vectors[:,method_idx,:]
# #     data_method = np.sort(data_method, axis=1)
    

# #     nom_methode = methods[method_idx]
# #     couleur = colors[method_idx]

# #     for percentile, style_ligne in zip(percentiles, linestyles):
# #         valeur_percentile = np.percentile(data_method, percentile, axis=1)


# #         ax.plot(range(1, len(names) + 1), valeur_percentile, marker='o', linestyle=style_ligne, color=couleur,
# #                 label=f"{nom_methode} - {percentile}th percentile")

# # ax.set_title('Percentiles Objective Difference for ' + str(nb_instances[0]) + ' instances')
# # ax.set_xlabel('Number of Jobs')
# # ax.set_ylabel('Difference')
# # ax.set_xticks(np.arange(1, len(names) + 1))
# # ax.set_xticklabels(names)
# # ax.legend()

# # plt.tight_layout()
# # plt.show()



# data_runtime = pd.DataFrame(data = runtimes_vectors, columns = names, index = methods )
# data_runtime.index.name = "runtimes"

# data_accuracy = pd.DataFrame(data = accuracys_vectors, columns = names, index = methods )
# data_accuracy.index.name = "accuracy"

# data_std = pd.DataFrame(data = std_difference_vectors, columns = names, index = methods)
# data_std.index.name = "std"

# data_mean = pd.DataFrame(data = mean_difference_vectors, columns = names, index = methods)
# data_mean.index.name = "mean"

# with open('Output/runtimesBest.pkl', 'wb') as file:
#   pickle.dump(data_runtime, file)

# with open('Output/accuracyBest.pkl', 'wb') as file:
#   pickle.dump(data_accuracy, file)

# with open('Output/stdBest.pkl', 'wb') as file:
#   pickle.dump(data_std, file)
  
# with open('Output/meanBest.pkl', 'wb') as file:
#   pickle.dump(data_mean, file)
  

# print('\nExecution Time per instances')
# display(data_runtime)


##### Save statistics
with open(saveFile,'wb') as file:
        pickle.dump(numIns, file)
        pickle.dump(Ks, file)
        pickle.dump(total_runtimes, file)
        pickle.dump(objHeur, file)
