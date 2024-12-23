import numpy as np
import itertools

NKEEP = 60

def select_nkeep_items(tmp_lateness, tmp_finish, tmp_perms):
    # Calculate the step to select items with as much diversity as possible
    step = len(tmp_lateness) // NKEEP
    
    # Select indices based on the step
    indices = [i * step for i in range(NKEEP)]
    
    indices[-1] = len(tmp_lateness) - 1
    
    # Select items based on the calculated indices
    selected_lateness = [tmp_lateness[i] for i in indices]
    selected_finish = [tmp_finish[i] for i in indices]
    selected_perms = [tmp_perms[i] for i in indices]

    return selected_lateness, selected_finish, selected_perms

def nkeep_fn(tmp_lateness, tmp_finish,tmp_perms):
    indices = np.lexsort((tmp_finish, tmp_lateness))
    indices = indices.tolist()
    # Sort each array using the sorted indices
    tmp_lateness_sorted = [tmp_lateness[i] for i in indices]
    tmp_finish_sorted = [tmp_finish[i] for i in indices]
    tmp_perms_sorted = [tmp_perms[i] for i in indices]
    
    tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted = select_nkeep_items(tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted)

    # Output the sorted arrays
    return tmp_lateness_sorted, tmp_finish_sorted, tmp_perms_sorted


def count_unique_vectors_2d(array):
    # Convertir le tableau 2D en une liste de tuples
    tuples_list = [tuple(row) for row in array]
    
    # Compter les occurrences de chaque tuple
    unique_values, counts = np.unique(tuples_list, return_counts=True, axis=0)
    
    solutions_index = np.argsort(counts)[::-1]
 
    sorted_vector = unique_values[solutions_index]
    nb_solutions = counts[solutions_index]

    
    return sorted_vector, nb_solutions
    
#remove_duplicate
def remove_duplicates(arr_list):
    unique_tuples = set()

    result_list = []
    for arr in arr_list:
        arr_tuple = tuple(sorted(arr))
        if arr_tuple not in unique_tuples:
            unique_tuples.add(arr_tuple)
            result_list.append(tuple(arr))

    return result_list



# Function to evaluate the lateness and finish time for a given permutation of jobs

def eval_perm_ct_fn(atmp, xtmp, dtmp, ptmp, qtmp):
    ntmp = len(atmp)
    ytmp = np.zeros(ntmp)  # Initialize the list of finish times
    ytmp[0] = atmp[0] + xtmp[0]  # Finish time for the first task
    late = max(ytmp[0] - dtmp[0], 0)  # Calculate the lateness for the first task
    late = qtmp[0] * late + ptmp[0] * (late > 0)  # Calculate the lateness penalty

    # Calculate finish times and lateness penalties for each task
    for k in range(1, ntmp):
        ytmp[k] = max(ytmp[k - 1], atmp[k]) + xtmp[k]  # Finish time of task k
        late_add = max(0, ytmp[k] - dtmp[k])  # Calculate lateness of task k
        late += qtmp[k] * late_add + ptmp[k] * (late_add > 0)  # Add the lateness penalty

    return late, float(ytmp[-1])
    
    
# Function to evaluate the lateness and finish time for all  permutation of jobs
def eval_perm_ct_fn_all(partial_list, A, X, D, P, Q):
    result = []
    new_lateness = []
    new_finish = []

    for perm in partial_list:
        penalties, finish_time = eval_perm_ct_fn(A[list(perm)], X[list(perm)], D[list(perm)], P[list(perm)], Q[list(perm)])
        result.append(perm)
        new_lateness.append(penalties)
        new_finish.append(finish_time)

    return np.array(result), np.array(new_lateness), np.array(new_finish)

def eval_perm_ct_fn_all_Ins(partial_list, A, X, D, P, Q):
    result = []
    new_lateness = []
    new_finish = []

    for perm in partial_list:
        penalties, finish_time = eval_perm_ct_fn(A[list(perm)], X[list(perm)], D[list(perm)], P[list(perm)], Q[list(perm)])
        result.append(perm)
        new_lateness.append(penalties)
        new_finish.append(finish_time)

    if len(result)>NKEEP:
        new_lateness, new_finish, result = nkeep_fn(new_lateness,new_finish,result)

    return np.array(result), np.array(new_lateness), np.array(new_finish)

# Function to generate  permutations of jobs
def generate_permutations(partial_list1, J, initial):
    masque = np.isin(partial_list1, list(initial))
    preliminary_order = partial_list1[~masque][:J]
    initial_permutations = []

    for job in preliminary_order:
        remaining_tasks = [task for task in preliminary_order if task != job]
        permutations = itertools.permutations(remaining_tasks, J - 1)

        initial_permutations.extend([initial + (job,) + perm for perm in permutations])
    return initial_permutations

def generate_permutations_Ins(partial_list1, J, initial):
    masque = np.isin(partial_list1, list(initial))
    preliminary_order = partial_list1[~masque][:J]
    initial_permutations = []

    for job in preliminary_order:
        remaining_tasks = [task for task in preliminary_order if task != job]
        permutations = itertools.permutations(remaining_tasks, J - 1)

        initial_permutations.extend([initial + (job,) + perm for perm in permutations])
    return initial_permutations
 

#pareto optimal solutions 
def pareto_solution(result_permutations, tmp_lateness, tmp_finish, taille,J):
    late_compare_gt = (tmp_lateness.reshape((1,-1))>tmp_lateness.reshape((-1,1))).astype(int)
    finish_compare_gt = (tmp_finish.reshape((1,-1))>tmp_finish.reshape((-1,1))).astype(int)
    late_compare_ge = (tmp_lateness.reshape((1,-1))>=tmp_lateness.reshape((-1,1))).astype(int)
    finish_compare_ge = (tmp_finish.reshape((1,-1))>=tmp_finish.reshape((-1,1))).astype(int)
    
    index_remove_mx = (late_compare_gt * finish_compare_ge + late_compare_ge * finish_compare_gt)  
    ix = (np.sum(index_remove_mx,axis=0)==0)
    
    # II Identify pareto optimal solutions   
    pareto_list = result_permutations[ix]
    
    #III Iv classification and order the sequences

    pareto_list_jobs = pareto_list[:, 0:taille]
    unique_pareto,  nb_solutions = count_unique_vectors_2d(pareto_list_jobs)
    if unique_pareto.shape[0] > J:
    
    # V Remove duplicate sequences
        pareto_no_dupl = remove_duplicates(unique_pareto) 
    else:
        pareto_no_dupl = unique_pareto

    if len(pareto_no_dupl) > J : 
        candidates = pareto_no_dupl[0:J]
    else : 
        candidates = pareto_no_dupl

   
    return candidates


# Iteration function
def sequential_Ins(A, X, D, P, Q, partial_list1, J):
    initial = ()
    partial_list = []
    for i in range(len(partial_list1) - (J - 1)):
        if i == 0:
            partial_list = generate_permutations_Ins(partial_list1, J, initial)
        else:
            result_permutations, lateness, finish = eval_perm_ct_fn_all_Ins(partial_list, A, X, D, P, Q)
            initials = pareto_solution(result_permutations, lateness, finish, i, J)
            partial_list = []
            for init in initials:
                initial = tuple(init)
                partial_list += generate_permutations_Ins(partial_list1, J, initial)
    return partial_list

   
def insertHeurLimIns_revisedFn(A, D, X, P, Q,J):
    A = A.reshape(D.shape)  # Arrival times
    X = X.reshape(D.shape)
    if type(P) != "numpy.ndarray":
        P = P + 0 * A
        Q = Q + 0 * A
    N = D.shape[0]
    initial_orders = [D,D - X, A]
    J=int(J)
    if N <= 20 : 

        result_permutations_list, lateness_list, finish_list = [],[],[]
        Due = initial_orders[0]        
        partial_list1 = np.argsort(Due)
        last_list = sequential_Ins(A, X, D, P, Q, partial_list1, J)
        result_permutations, lateness, finish = eval_perm_ct_fn_all_Ins(last_list, A, X, D, P, Q)
        
        ix0 = np.where(lateness == min(lateness))[0]
        finish0 = finish[ix0]
        ix1 = np.where(finish0 == min(finish0))[0][0]
        iSelect = ix0[ix1]
            
        result_permutations_list.append(result_permutations[iSelect])
        lateness_list.append(lateness[iSelect])
        finish_list.append(finish[iSelect])
        if lateness[iSelect] == 0 : 
            return result_permutations_list[0], lateness_list[0], finish_list[0]
            
            
        Due = initial_orders[1]        
        partial_list1 = np.argsort(Due)

        last_list = sequential_Ins(A, X, D, P, Q, partial_list1, J)
        result_permutations, lateness, finish = eval_perm_ct_fn_all_Ins(last_list, A, X, D, P, Q)
 
        ix0 = np.where(lateness == min(lateness))[0]
        finish0 = finish[ix0]
        ix1 = np.where(finish0 == min(finish0))[0][0]
        iSelect = ix0[ix1]
            
        result_permutations_list.append(result_permutations[iSelect])
        lateness_list.append(lateness[iSelect])
        finish_list.append(finish[iSelect])
        if lateness[iSelect] == 0 : 
            return result_permutations_list[1], lateness_list[1], finish_list[1]
        

        Due = initial_orders[2]        
        partial_list1 = np.argsort(Due)
        last_list = sequential_Ins(A, X, D, P, Q, partial_list1, J)
        result_permutations, lateness, finish = eval_perm_ct_fn_all_Ins(last_list, A, X, D, P, Q)
 
        ix0 = np.where(lateness == min(lateness))[0]
        finish0 = finish[ix0]
        ix1 = np.where(finish0 == min(finish0))[0][0]
        iSelect = ix0[ix1]
            
        result_permutations_list.append(result_permutations[iSelect])
        lateness_list.append(lateness[iSelect])
        finish_list.append(finish[iSelect])
        best_index =np.argsort(np.array(lateness_list))[0]   
                
        return result_permutations_list[best_index], lateness_list[best_index], finish_list[best_index]
          
    partial_list1 = np.argsort(D)

    last_list = sequential_Ins(A, X, D, P, Q, partial_list1, J)
    result_permutations, lateness, finish = eval_perm_ct_fn_all_Ins(last_list, A, X, D, P, Q)
    
    ix0 = np.where(lateness == min(lateness))[0]
    finish0 = finish[ix0]
    ix1 = np.where(finish0 == min(finish0))[0][0]
    iSelect = ix0[ix1]
        
    return result_permutations[iSelect], lateness[iSelect], finish[iSelect]
    
