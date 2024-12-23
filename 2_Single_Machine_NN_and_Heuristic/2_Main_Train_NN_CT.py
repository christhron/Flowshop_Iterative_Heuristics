# -*- coding: utf-8 -*-
"""

Either: 
    
     Train NN using pre-solved instances, and compare performance with heuristic solution

or  

     compare a previously trained NN with heuristic 

NOTE:  NN performance is VERY bad.  NN approach is not recommended.


@author: matth
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statistics as st
from matplotlib import pyplot as plt
from Functions.FlowshopBaseSchedRoutines import Obj_pq
from Functions.InsertionHeuristicFunctions import insertHeurFn

#plt.style.use('seaborn-deep')


'''
######################################################
Specify file names for data, weights, and model here
#####################################################
'''

# Paths to training/validation data and testing data respectively
dataFilePath = "Data/single_machine_8N.pickle"

# If weights are already created, load them here
weightsFilePath = "stuff_for_Main_Train_NN/NN_Weights/NN_weights_8N.pickle"
loadWeights = True
storeWeights = False

modelname= 'stuff_for_Main_Train_NN/NN_Models/test'
saveModel = True

# Number of jobs
N=8
# Lateness for objective function: penalty & rate
p=50
q=1
scaler=MinMaxScaler()
nEpoch = 250
nInBatch = 75




##### Custom callback to compute metric during training. Currently not used.
# class CustomMetricCallback(tf.keras.callbacks.Callback):
#     def __init__(self, x_input, validation_data):
#         super().__init__()
#         self.x_input = x_input
#         self.validation_data = validation_data
    
#     def on_epoch_end(self, epoch, logs=None):
#         # Get predicted outputs
#         y_pred = self.model.predict(self.validation_data[0])
        
#         # Calculate the custom metric
#         metric_value = objective_metric(self.validation_data[1], y_pred, self.x_input)
#         print("Objective Difference:", metric_value)
        
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
        
    return Y,S,Idle

### Objective metric to compare the difference in objective function values between the MILP & NN; 
### currently not used--using the built-in metric instead
# def objective_metric(y_true, y_pred, x_input):
#     # y_true: True labels
#     # y_pred: Predicted output from the neural network
#     totalsamples=0
#     total_obj_diff=0
#     Z=np.zeros([N,1])
    
#     for true_batch, pred_batch, input_batch in zip(y_true, y_pred, x_input):
#             # Get finish times on true_sample and pred_sample. 
#             # true_batch is the MILP job orders and pred_batch is the predicted orders.
#             # argsort of argsort [j] gives the position of the original j'th entry
#             # in the ordered list.
#             true_sample=np.argsort( np.argsort(true_batch)).reshape([N,1])
#             pred_sample=np.argsort( np.array(pred_batch)).reshape([N,1])
#             realF, realS, realI=CalcYS(0,true_sample,instance[0],instance[1].reshape([N,1]),Z)
#             predF, predS, predI=CalcYS(0,pred_sample,instance[0],instance[1].reshape([N,1]),Z)
            
#             # Calculate objective values
#             valR=Obj_pq(instance[2],realF,p,q)
#             valP=Obj_pq(instance[2],predF,p,q)
  
#             # Calculate the absolute difference in time
#             obj_diff = valP-valR
#             total_obj_diff+=obj_diff
#             totalsamples+=1

#     return total_obj_diff/totalsamples

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
            label[0]=data['order']*1.0/(N-1)
           # label[1]=scale
            all_labels.append(label)
        except EOFError:
            # Reached the end of the file
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

# If NN has already been trained and performance comparison is desired
if loadWeights:
    with open(weightsFilePath, 'rb') as f:
        [weights,biases,x_test,y_test] = pickle.load(f)
    
else:    
    # Train/test split    
    x_train, x_test, y_train, y_test = train_test_split(inputs, all_labels, test_size=2000)
    # Train/validate split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    
    # For unpacking scales from labels
    # n_train=len(y_trains)
    # n_val=len(y_vals)
    # testscale=np.zeros([n_train])
    # y_train=np.zeros([n_train,N])
    # valscale=np.zeros([n_val])
    # y_val=np.zeros([n_val,N])
    # for i in range(n_train):
    #     testscale[i]=y_trains[i][1,0]
    #     y_train[i]=y_trains[i][0]
    #     if i<len(y_vals):
    #         valscale[i]=y_vals[i][1,0]
    #         y_val[i]=y_vals[i][0]
    
    # store_input_data_callback = CustomMetricCallback(x_train, validation_data=(x_val, y_val))
    
        
    # Definition of model structure
    singleMachineModel = tf.keras.Sequential([
            Input(shape=(3,N)), 
            Flatten(),
            Dense(4 * N),
            # Dense(20 * N),# activation='selu'),
            # Dense(12 * N),
            Dense(N, activation="sigmoid"),
    ])
        
    # Setting up how the model trains
    singleMachineModel.compile(loss='mae', optimizer='SGD', metrics="mse")#run_eagerly=True) #metrics=[objective_metric])#"mse"])
       
    # Model training 
    history = singleMachineModel.fit(x_train, y_train, verbose='0', validation_data=(x_val, y_val), epochs=nEpoch, batch_size=nInBatch)#, callbacks=[store_input_data_callback])
    plt.title('Learning Curves for ' + modelname)
    plt.ylabel('Mean Absolute Error')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')    #bars side-by-side or something for readability
    plt.legend()
    plt.show()
    
    # Extracting weights
    weights=[]
    biases=[]
    for i, layer in enumerate(singleMachineModel.layers):
        a=0
        while a<=i:
            try:
               weights.append(layer.get_weights()[0])
               biases.append(layer.get_weights()[1])
               break
            except:
               a+=1
               continue

    if storeWeights:
        with open(weightsFilePath, 'wb') as f:
            pickle.dump([weights,biases,x_test,y_test],f)
    if saveModel:
        singleMachineModel.save(modelname)

        

# Count of accurate objective predictions and arrays for objective values of NN, MILP, and random for comparison
count=0
n_test = len(y_test)
obj=np.zeros(n_test)  #NN
objr=np.zeros(n_test) #MILP
objRand=np.zeros(n_test) #Random Order
objOrder=np.zeros(n_test) #In Order
objHeur=np.zeros(n_test) #Heuristic
for i in range(n_test):
        print("Test case",i)
        scale=1 #y_test[i][1,0]
        A=np.array(x_test[i,0]).reshape([1,N])#(scale*x_test[i][0]).reshape([N,1])
        X=np.array(scale*x_test[i][1]).reshape([1,N])
        D=np.array(x_test[i,2]).reshape([1,N])#(scale*x_test[i][2])#.reshape([N,1])
        
        label=np.rint(np.array(y_test[i])*(N-1) ).astype(int) #Ensure the orders are in a numpy array. rint rounds to the nearest integer but leaves the type as float
        
        samp=np.append(np.append(A, X, axis=0), D, axis=0)
        D=D.flatten()
        
        #Current prediction
        pred=x_test[i].flatten()
        for j in range(len(weights)):
            pred=(pred@weights[j]) + biases[j]
        pred=np.argsort(pred)

        # Predictions of Y (job finish times),S (start times), and idle times for different algorithms
        
        # Prediction from NN
        predY, predS, predIdle=CalcYS(0, pred.reshape([N,1]), A.T, X.T, np.zeros([N,1]))
        # Prediction from MILP
        realY, realS, realIdle=CalcYS(0, label.reshape([N,1]), A.T, X.T, np.zeros([N,1]))
        # Predicion for random order (not currently used)
        randomS=np.random.choice(N, size=N, replace=False)
        randY, randS, randIdle=CalcYS(0, randomS.reshape(N,1), A.T, X.T, np.zeros([N,1]))

        # Prediction for greedy (in arrival order)
        # This can be used as baseline
        ordY, ordS, ordIdle=CalcYS(0, np.arange(N).reshape([N,1]), A.T, X.T, np.zeros([N,1]))

        # Prediction using heuristic
        HeurOrders, Heur_late_totals, Heur_fin_totals=insertHeurFn(A,D,X,p,q)
        heurY, heurS, heurIdle=CalcYS(0, HeurOrders.reshape([N,1]), A.T, X.T, np.zeros([N,1]))

        # obj[i]=Obj_pq(D,predY,p,q)
        # objr[i]=Obj_pq(D,realY,p,q)
        # objRand[i]=Obj_pq(D,randY,p,q)
        # objOrder[i]=Obj_pq(D,ordY,p,q)        
        objHeur[i]=Obj_pq(D.flatten(),heurY,p,q)

        
        if (obj[i]==objr[i]):
            count+=1

fig, (ax1, ax2) = plt.subplots(nrows=2)
binw=25
Tmax = max(max(obj),max(objr),max(objOrder),max(objHeur))
binsize=np.arange(0,Tmax+binw,binw)
#histtype='step', fill=True)
colors=["blue","purple",'green','red']
names=["MILP", "Heuristic","Greedy","NN"]
ax1.hist([objr,objHeur,objOrder,obj], bins=binsize, color=colors, label=names)#, stacked=False, density=True)
ax1.set_title('Penalty Distributions for ' + modelname) 
ax1.legend(loc='best')
ax1.set_xlabel("Penalty")
ax1.set_ylabel("Num. Observations")
#ytick=np.linspace(Tmin,Tmax,int(len(objr)/10))
#ax1.set_yticks(ytick)
xtick=binsize[::2]
ax1.set_xticks(xtick)

colors2=colors[1:]
names=["Heuristic minus MILP", "Greedy minus MILP", "NN minus MILP" ]
binsize=np.arange(0,max(obj-objr)+binw,binw)
ax2.hist([objHeur-objr,objOrder-objr,obj-objr], bins=binsize, color=colors2, histtype='bar', stacked=False, fill=True, label=names)
ax2.set_title('Penalty differences for ' + modelname) 
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

print("Objective Accuracy for heuristic is: ", sum(objHeur<=objr)/len(y_test)*100,"%")
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

# Sorted: NN
pltObj=np.sort(obj)
y_values = np.arange(len(pltObj)) / len(pltObj)
plt.plot(pltObj, y_values, label="NN",color=colors[3])

plt.title("Distribution of objective functions")
plt.legend()
    #ax2.set_xlim(xmin=-5,xmax=5)
    #ax2.set_ylim(ymin=0,ymax=10)
    #plt.xlabel("Percent Difference (Bin Size=" + str(binw) + ")")
plt.ylabel("Proportion of Jobs")
plt.xlabel("Objective Value")

plt.show()
