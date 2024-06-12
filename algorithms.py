"""
Algorithms for Interventional Causal Discovery in a Mixture of DAGs 
"""

import warnings
warnings.simplefilter('ignore')
import numpy as np
import os
from causaldag import partial_correlation_suffstat, partial_correlation_test
from utils import generate_intervention_model, generate_samples_mixture, find_breaking_set, intersection, setminus, allsubsets
import networkx as nx


if not os.path.exists('./results'):
    os.makedirs('./results')

def oracle_CADIM(mixture_model):
    """Oracle version of CADIM algorithm: Causal Discovery from Interventions on Mixture Models 

    Args:
        mixture_model (dict): mixture model

    Returns:
        oracle_parent_mat (np array): 0-1 2d array, j-i entry is 1 if j \in pam(i)
        oracle_tau_list (list): list of cyclic complexity numbers for all nodes
    """
    n_nodes = mixture_model['n_nodes']
    ### Step 1: Identify Mixture Ancestors
    anm_mat = mixture_model['mixture_ancestors_mat']
    anm_list = {}
    dem_list = {}
    for i in range(n_nodes):
        anm_list[i] = list(np.where(anm_mat[:,i])[0])
        dem_list[i] = list(np.where(anm_mat[i,:])[0])

    oracle_parent_mat = np.zeros((n_nodes,n_nodes))
    oracle_tau_list = np.zeros(n_nodes)

    # repeat Steps 2,3,4 for each node to find mixture parents
    for idx_node in range(n_nodes):
        anm_i_mat = np.zeros((n_nodes,n_nodes))
        anm_i = anm_list[idx_node]

        for j in anm_i:
            for k in anm_i:
                if anm_mat[j,k] == 1:
                    anm_i_mat[j,k] = 1
        
        ### Step 2: Obtain cycle-free descendats
        # find all cycles that consist of nodes in anm_i
        G_anm_i = nx.from_numpy_array(anm_i_mat, create_using=nx.DiGraph)
        cycles_anm_i = list(nx.simple_cycles(G_anm_i))
        
        de_i_list = {}
        if len(cycles_anm_i) == 0:
            B_i = []
            tau_i = 0
            for j in anm_i:
                de_i_list[j] = intersection(dem_list[j],anm_i)
                de_i_list[j].append(idx_node)
        else:
            B_i = list(find_breaking_set(cycles_anm_i))
            tau_i = len(B_i)
            for j in anm_i:
                if j in B_i:
                    I = B_i
                    I_mixture_model = generate_intervention_model(mixture_model,I)
                    de_i_list[j] = list(set(np.where(I_mixture_model['mixture_ancestors_mat'][j])[0]) & set(anm_i + [idx_node]))             
                else:
                    I = B_i + [j]
                    I_mixture_model = generate_intervention_model(mixture_model,I)
                    de_i_list[j] = list(set(np.where(I_mixture_model['mixture_ancestors_mat'][j])[0]) & set(anm_i + [idx_node]))                        


        # update ancestors of node i after interventions
        anm_i_refined = [j for j in anm_i if idx_node in de_i_list[j]]
        de_i_refined_list = {}
        for j in anm_i_refined:
            de_i_refined_list[j] = [k for k in de_i_list[j] if k in anm_i_refined]

        ### Step 3: Topological Layering
        A = anm_i_refined.copy()
        t = 0
        S_all = {}

        while len(A) > 0:
            t = t+1
            S_all[t] = []
            for j in A:
                if len(intersection(de_i_refined_list[j],A)) == 0:
                    S_all[t].append(j)

            for j in S_all[t]:
                A.remove(j) 

        ### Step 4: Identify Mixture Parents
        pam_i = []
        for u in range(1,t+1):
            for j in S_all[u]:
                I = pam_i + B_i + [j]
                I_mixture_model = generate_intervention_model(mixture_model,I)
                if I_mixture_model['mixture_ancestors_mat'][j,idx_node] == 1:
                    pam_i += [j]

        oracle_parent_mat[pam_i,idx_node] = 1
        oracle_tau_list[idx_node] = tau_i

    return oracle_parent_mat, oracle_tau_list


def find_mixture_ancestors(mixture_model,n_samples=1000,alpha=0.1):
    """Find all mixture ancestors, sample version.

    Args:
        mixture_model (dict): mixture model
        n_samples (int): Number of samples to use from each component DAG. Defaults to 1000.
        alpha (float):  Significance level for CI tests. Defaults to 0.1.

    Returns:
        anm_hat_mat (np array): 0-1 2d array for mixture ancestors. j-i entry is 1 if j \in anm(i)
    """
    n_nodes = mixture_model['n_nodes']
    anm_hat_list = {}
    dem_hat_list = {}
    p_values_mat = np.zeros((n_nodes,n_nodes))
    for i in range(n_nodes):
        I_mixture_model = generate_intervention_model(mixture_model,I=[i])
        X_int = generate_samples_mixture(I_mixture_model,n_samples=n_samples)
        X_int = np.vstack([X_int[graph_idx] for graph_idx in range(len(X_int))])
        I_suffstat = partial_correlation_suffstat(X_int)
        p_values_mat[i] = [partial_correlation_test(I_suffstat,i,j)['p_value'] for j in range(n_nodes)]
        np.fill_diagonal(p_values_mat,1)

    anm_hat_mat = p_values_mat < alpha

    return anm_hat_mat

def process_node(idx_node,mixture_model,anm_mat,n_samples,alpha):
    """Find mixture parents of a node.

    Args:
        idx_node (int): index for node i
        mixture_model (dict): mixture model
        anm_mat (np array): 2d 0-1 array for mixture ancestors
        n_samples (int): Number of samples to use from each component DAG.
        alpha (float):  Significance level for CI tests. Defaults to 0.1.

    Returns:
        hat_pa_i (list): mixture parents of node i
        tau_i (int): cyclic complexity of node i 
    """
    n_nodes = mixture_model['n_nodes']

    dem_list = {}
    for i in range(n_nodes):
        dem_list[i] = list(np.where(anm_mat[i,:])[0])

    anm_i_mat = np.zeros((n_nodes,n_nodes))
    anm_i = list(np.where(anm_mat[:,idx_node])[0])

    for j in anm_i:
        for k in anm_i:
            if anm_mat[j,k] == 1:
                anm_i_mat[j,k] = 1
    
    ### Step 2: Obtain cycle-free descendants
    # find all cycles that consist of nodes in anm_i
    G_anm_i = nx.from_numpy_array(anm_i_mat, create_using=nx.DiGraph)
    cycles_anm_i = list(nx.simple_cycles(G_anm_i))

    de_i_list = {}
    if len(cycles_anm_i) == 0:
        B_i = []
        tau_i = 0
        for j in anm_i:
            de_i_list[j] = intersection(dem_list[j],anm_i)
            de_i_list[j].append(idx_node)       
    else:
        B_i = list(find_breaking_set(cycles_anm_i))
        tau_i = len(B_i)
        for j in anm_i:
            I = list(set(B_i + [j])) 
            I_mixture_model = generate_intervention_model(mixture_model,I)
            X_int = generate_samples_mixture(I_mixture_model,n_samples=n_samples)
            X_int = np.vstack([X_int[graph_idx] for graph_idx in range(len(X_int))])
            I_suffstat = partial_correlation_suffstat(X_int)

            de_i_list[j] = []
            for k in intersection(setminus(dem_list[j],B_i), anm_i + [idx_node]):
                if partial_correlation_test(I_suffstat,j,k)['p_value'] < alpha and j != k:
                    de_i_list[j] = de_i_list[j] + [k]

    
    # update ancestors of node i after interventions
    anm_i_refined = [j for j in anm_i if idx_node in de_i_list[j]]
    de_i_refined_list = {}
    for j in anm_i_refined:
        de_i_refined_list[j] = [k for k in de_i_list[j] if k in anm_i_refined]

    # Step 3: Topological Layering
    A = anm_i_refined.copy()
    t = 0
    S_all = {}

    while len(A) > 0:
        t = t+1
        S_all[t] = []
        for j in A:
            if len(intersection(de_i_refined_list[j],A)) == 0:
                S_all[t].append(j)

        for j in S_all[t]:
            A.remove(j) 

    ### Step 4: Identify Mixture Parents
    hat_pa_i = []
    for u in range(1,t+1):
        for j in S_all[u]:
            I = list(set(hat_pa_i + B_i + [j]))
            I_mixture_model = generate_intervention_model(mixture_model,I)
            X_int = generate_samples_mixture(I_mixture_model,n_samples=n_samples)
            X_int = np.vstack([X_int[graph_idx] for graph_idx in range(len(X_int))])
            I_suffstat = partial_correlation_suffstat(X_int)

            if partial_correlation_test(I_suffstat,idx_node,j)['p_value'] < alpha:
                hat_pa_i += [j]

    return hat_pa_i, tau_i


def CADIM(mixture_model,n_samples,alpha):
    """Sample version of CADIM algorithm: Causal Discovery from Interventions on Mixture Models 

    Args:
        mixture_model (dict): mixture model
        n_samples (int): Number of samples to use from each component DAG. Defaults to 1000.
        alpha (float):  Significance level for CI tests. Defaults to 0.1.

    Returns:
        sample_parent_mat (np array): 0-1 2d array for mixture parents. j-i entry is 1 if j \in pam(i)
        oracle_tau_list (list): list of cyclic complexity numbers for all nodes
    """
    n_nodes = mixture_model['n_nodes']
    sample_anm_mat = find_mixture_ancestors(mixture_model=mixture_model,
    n_samples=n_samples,alpha=alpha)

    sample_parent_mat = np.zeros((n_nodes,n_nodes))
    sample_tau_list = np.zeros(n_nodes)

    for idx_node in range(n_nodes):
        sample_pa_i, sample_tau_i = process_node(idx_node=idx_node,mixture_model=mixture_model,anm_mat=sample_anm_mat,n_samples=n_samples,alpha=alpha)

        sample_parent_mat[sample_pa_i,idx_node] = 1
        sample_tau_list[idx_node] = sample_tau_i

    return sample_parent_mat, sample_tau_list


def CI_skeleton(mixture_model,n_samples,alpha):
    """Finds the inseparable node pairs in the mixture model.

    Args:
        mixture_model (dict): mixture_model
        n_samples (int): Number of samples to use from each component DAG. 
        alpha (float):  Significance level for CI tests.

    Returns:
        adj_mat (np array): symmetric adjacency matrix for inseparable node pairs.
    """
    n_nodes = mixture_model['n_nodes']
    X = generate_samples_mixture(mixture_model,n_samples=n_samples)
    X = np.vstack([X[graph_idx] for graph_idx in range(len(X))])
    suffstat = partial_correlation_suffstat(X)

    sepsets = {}
    all_pairs = [(i,j) for i in range(n_nodes) for j in range(n_nodes) if i < j]
    all_edges = [(i,j) for i in range(n_nodes) for j in range(n_nodes) if i < j]
    adj_mat = np.ones((n_nodes,n_nodes))
    np.fill_diagonal(adj_mat,0)

    for (i,j) in all_pairs:
        rest_nodes = [k for k in range(n_nodes) if k not in (i,j)]
        all_S = allsubsets(rest_nodes)
        flag_edge = True
        for S in all_S:
            if len(S) > 0:
                p_val = partial_correlation_test(suffstat,i,j,cond_set=S,alpha=alpha)['p_value']  
            else:
                p_val = partial_correlation_test(suffstat,i,j,cond_set=(),alpha=alpha)['p_value']
           
            if p_val > alpha:
                sepsets[(i,j)] = S   
                all_edges.remove((i,j))
                adj_mat[i,j] = 0
                adj_mat[j,i] = 0
                flag_edge = False
                break

            if flag_edge is False:
                break

    return adj_mat



