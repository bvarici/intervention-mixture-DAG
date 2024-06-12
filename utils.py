"""
Utils and helpers for simulations of Interventional Causal Discovery in a Mixture of DAGs 
"""

import numpy as np
import causaldag as cd
import itertools

def confusion_mat_graph(true_g,hat_g):
    """Takes true graph adjacency matrix (true_g) and estimated graph adjacency matrix (hat_g).
    Returns confusion matrix
    """
    edge_cm = [
        [
            ( true_g &  hat_g).sum(dtype=int),
            (~true_g &  hat_g).sum(dtype=int),
        ], [
            ( true_g & ~hat_g).sum(dtype=int),
            (~true_g & ~hat_g).sum(dtype=int),
        ]
    ]
    return edge_cm

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def setminus(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

def allsubsets(s):
    """Return the power set (as a list) of a set 
    """
    N = len(s)
    temp = []
    for n in range(0,N+1):
        temp.append(list(itertools.combinations(s, n)))

    ss = []
    for n in range(0,N+1):
        ss += temp[n]

    ss = [list(a) for a in ss]
    return ss

def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def find_non_invariant_coordinates(*matrices):
    # Create an array to store the differences
    diffs = [matrices[i] != matrices[j] for i in range(len(matrices)) for j in range(i + 1, len(matrices))]
    
    # Combine the differences using logical OR
    non_invariant_coords = np.logical_or.reduce(diffs)
    
    # Get the coordinates where differences exist
    coords = np.transpose(np.nonzero(non_invariant_coords))
    
    return coords


def generate_mixture_model(n_nodes,K,avg_neighbor=2,
                           max_weight=2,min_weight=0.25,max_mean_noise=1,min_mean_noise=-1,max_var_noise=1.5,min_var_noise=0.5):

    A_all = np.zeros((K,n_nodes,n_nodes))
    for i in range(K):
        Ai = np.triu(np.random.uniform(size=[n_nodes,n_nodes]) < (avg_neighbor/n_nodes),k=1)   
        # randomize topological order
        idx = np.random.permutation(n_nodes)
        Ai = Ai[idx][:,idx]
        A_all[i] = Ai

    # create graph objects
    G_all = [cd.DAG.from_amat(A_all[i]) for i in range(K)]
    top_order_all = [G_all[i].topological_sort() for i in range(K)]

    # all true edges
    mixture_parents_mat = A_all.sum(0)
    mixture_parents_mat[mixture_parents_mat != 0] = 1

    # save mixture ancestors: i,j = 1 if i is mixture ancestor of j 
    mixture_ancestors_mat = np.zeros((n_nodes,n_nodes))

    for i in range(n_nodes):
        anm_i = set()
        for graph_idx in range(K):
            anm_i.update(G_all[graph_idx].ancestors_of(i))

        mixture_ancestors_mat[list(anm_i),i] = 1

    # sample edge weights
    rand_weights = np.random.uniform(min_weight,max_weight,[n_nodes,n_nodes])* np.random.choice([-1,1],size=[n_nodes,n_nodes])

    W_all = [A_all[i]*rand_weights for i in range(K)]
    changing_coords = find_non_invariant_coordinates(*A_all)
    delta = list(set(changing_coords[:,1]))
    #non_delta = [i for i in range(n_nodes) if i not in delta]

    # sample parameters of the exogenous noise terms
    mean_epsilon = np.random.uniform(min_mean_noise,max_mean_noise,size=n_nodes)
    var_epsilon = np.random.uniform(min_var_noise,max_var_noise,size=n_nodes)

    mixture_model={'n_nodes':n_nodes,
                'G_all': G_all, 
                'A_all': A_all,
                'mixture_parents_mat':mixture_parents_mat, 
                'mixture_ancestors_mat':mixture_ancestors_mat,
                'W_all': W_all,
                'top_order_all':top_order_all,
                'delta': delta,
                'mean':mean_epsilon,
                'var':var_epsilon}

    return mixture_model

def generate_intervention_model(mixture_model,I, 
                                int_reduce_mean_rate=2,int_reduce_var_rate=2):
    G_all = mixture_model['G_all']
    K = len(G_all)
    n_nodes = mixture_model['n_nodes']
    A_all = mixture_model['A_all']
    W_all = mixture_model['W_all']
    delta = mixture_model['delta']
    mean_epsilon = mixture_model['mean']
    var_epsilon = mixture_model['var']
    top_order_all = mixture_model['top_order_all']


    A_int_all = A_all.copy()
    W_int_all = [W_all[i].copy() for i in range(K)]
    delta_int = [i for i in delta if i not in I]
    for i in range(K):
        A_int_all[i][:,I] = 0
        W_int_all[i][:,I] = 0

    G_int_all = [cd.DAG.from_amat(A_int_all[i]) for i in range(K)]

    # save mixture ancestors: i,j = 1 if i is mixture ancestor of j 
    mixture_ancestors_mat = np.zeros((n_nodes,n_nodes))

    for i in range(n_nodes):
        anm_i = set()
        for graph_idx in range(K):
            anm_i.update(G_int_all[graph_idx].ancestors_of(i))

        mixture_ancestors_mat[list(anm_i),i] = 1

    mean_int_epsilon = mean_epsilon.copy()
    var_int_epsilon = var_epsilon.copy()
    mean_int_epsilon[I] /=  int_reduce_mean_rate
    var_int_epsilon[I] /= int_reduce_var_rate

    int_mixture_model={'n_nodes':n_nodes,
                'G_all': G_int_all, 
                'A_all': A_int_all, 
                'W_all': W_int_all,
                'mixture_ancestors_mat':mixture_ancestors_mat,
                'top_order_all':top_order_all,
                'delta': delta_int,
                'mean':mean_int_epsilon,
                'var':var_int_epsilon}

    return int_mixture_model

def generate_samples_mixture(mixture_model,n_samples):
    W_all = mixture_model['W_all']
    mean_epsilon = mixture_model['mean']
    var_epsilon = mixture_model['var']
    top_order_all = mixture_model['top_order_all']
    K = len(W_all)
    n_nodes = len(W_all[0])

    samples = np.zeros((K,n_samples,n_nodes))
    noise = np.zeros((K,n_samples,n_nodes))

    for idx_graph in range(K):
        for ix, (mean,var) in enumerate(zip(mean_epsilon,var_epsilon)):
            noise[idx_graph,:,ix] = np.random.normal(loc=mean,scale=var ** .5, size=n_samples)
        
        for node in top_order_all[idx_graph]:
            parents_node = np.where(W_all[idx_graph][:,node])[0]
            if len(parents_node)!=0:
                parents_vals = samples[idx_graph,:,parents_node]
                samples[idx_graph,:,node] = np.sum(parents_vals.T * W_all[idx_graph][parents_node,node],axis=1) + noise[idx_graph,:,node]
            else:
                samples[idx_graph,:,node] = noise[idx_graph,:,node]            

    return samples


def find_breaking_set(cycles_anm_i):
    C_i_nodes = set([i for cycle in cycles_anm_i for i in cycle])
    tau = 1
    while True:
        tau_sized_subsets = findsubsets(C_i_nodes,tau)
        for candi_B in tau_sized_subsets:
            valid_B = True
            for cycle in cycles_anm_i:
                if len(intersection(candi_B,cycle)) == 0:
                    valid_B = False
                    break

            if valid_B == True:
                final_B = candi_B
                break
            else:
                continue

        if valid_B == True:
            break
        else:
            tau += 1

    return final_B

 