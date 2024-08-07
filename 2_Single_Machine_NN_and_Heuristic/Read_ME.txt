Folders: 
  Archive: old code
  Data.zip: pickle file single machine instances for comparison  
  Functions.zip: header functions
  Models.zip: neural network models 
  NN_Weights.zip:weights from the models 
  Output.zip: pickle files of runtimes and performance data
  Training_Instances.zip: training data for the neural network models

Main: 
  1_Single-Machine Flowshop_create_single_machine_pickle creates instances for comparison. 
  2_Main_Train_NN_CT can either train a new neural network or bring in weights from an existing neural network and print performance. 
  3_Main_Test_Heur_CT compares the insertion heuristic with the neural network on the same instances compared to in arrival order. 
  4_Compare_time_vs_NKEEP uses either the insertion or selection algorithm, as selected at the start of the file, over a varying number of kept permutations and either insertion 
    slots from the end or length of permutation window (the parameters changed according to the research paper). 
  5_Comparison_Graphs creates graphs for both insertion and selection, including pairs plots.
