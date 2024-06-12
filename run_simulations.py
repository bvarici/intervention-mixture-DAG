"""
Run simulations for Interventional Causal Discovery in a Mixture of DAGs 
"""

import warnings
warnings.simplefilter('ignore')
import numpy as np
import os
from utils import generate_mixture_model, confusion_mat_graph
import pickle as pkl

if not os.path.exists('./results'):
    os.makedirs('./results')

run_dir = './results/'

from algorithms import CADIM, oracle_CADIM, CI_skeleton



def run_sample_repeat(n_repeat=10,n_nodes=5,K=3,avg_neighbor=2,n_samples=1000,alpha=0.1,run_dir='./results/',save_results=False,run_CI_skeleton=False):
    """Repeats the simulation for different realizations.

    Args:
        n_repeat (int, optional): Number of different realizations. Defaults to 10.
        n_nodes (int, optional): Number of nodes in each DAG. Defaults to 5.
        K (int, optional): Number of component DAGs in the mixture. Defaults to 3.
        avg_neighbor (int, optional): Density p for Erdös-Renyi Random graphs G(n,p). Defaults to 2.
        n_samples (int, optional): Number of samples from each component DAG. Defaults to 1000.
        alpha (float, optional): Significance level for CI tests. Defaults to 0.1.
        run_dir (str, optional): Run directory. Defaults to './results/'.
        save_results (bool, optional): Whether to save the results. Defaults to False.
        run_CI_skeleton (bool, optional): Whether to learn the inseparable node pairs from observational data. Defaults to False.

    Returns:
        all_cm (np array): (n_repeat x 3) matrix, recording TP, FP, FN
        all_sample_tau(np array): (n_repeat x n_nodes) matrix, recording cyclic complexity values in sample version of CADIM
        all_sample_tau(np array): (n_repeat x n_nodes) matrix, recording true cyclic complexity values using oracle version of CADIM
        res(dict): recording everything
    """
    all_cm = np.zeros((n_repeat,3))
    all_oracle_tau = np.zeros((n_repeat,n_nodes))
    all_sample_tau = np.zeros((n_repeat,n_nodes))
    all_ci_cm = np.zeros((n_repeat,3))
    all_skeleton_cm = np.zeros((n_repeat,3))

    for run_idx in range(n_repeat):
        print(f"On it: n={n_nodes}, K={K}, n_s={n_samples}, a={alpha}, d={avg_neighbor}, run_idx={run_idx}")

        # generate the mixture model
        mixture_model = generate_mixture_model(n_nodes=n_nodes,K=K,avg_neighbor=avg_neighbor)

        ### ground truths - run the oracle version of CADIM algorithm
        true_parent_mat = mixture_model['mixture_parents_mat']
        oracle_parent_mat, oracle_tau_list = oracle_CADIM(mixture_model=mixture_model)
        all_oracle_tau[run_idx] = oracle_tau_list

        ### finite sample - run the sample version of CADIM algorithm
        sample_parent_mat, sample_tau_list = CADIM(mixture_model=mixture_model,n_samples=n_samples,alpha=alpha)       

        n_tp = int(np.sum(true_parent_mat * sample_parent_mat))
        n_fp = int(np.sum((sample_parent_mat - true_parent_mat) > 0))
        n_fn = int(np.sum((true_parent_mat - sample_parent_mat) > 0))
        all_cm[run_idx] = [n_tp, n_fp, n_fn]
        all_sample_tau[run_idx] = sample_tau_list

        # also let's do a skeleton comparison with CI tests.
        true_skeleton = (true_parent_mat + true_parent_mat.T) > 0 
        sample_skeleton = (sample_parent_mat + sample_parent_mat.T) > 0
        skeleton_sample_cm = confusion_mat_graph(true_skeleton,sample_skeleton)
        all_skeleton_cm[run_idx] = [skeleton_sample_cm[0][0], skeleton_sample_cm[0][1], skeleton_sample_cm[1][0]]

        if run_CI_skeleton is True:
            ci_skeleton = CI_skeleton(mixture_model,n_samples=n_samples,alpha=alpha).astype(bool)
            skeleton_ci_cm =  confusion_mat_graph(true_skeleton,ci_skeleton)
            all_ci_cm[run_idx] = [skeleton_ci_cm[0][0], skeleton_ci_cm[0][1], skeleton_ci_cm[1][0]]


        res = {'n_nodes': n_nodes,
            'K': K,
            'n_samples': n_samples,
            'alpha': alpha,
            'n_repeat': n_repeat,
            'sample_cm': all_cm,
            'sample_tau': all_sample_tau,
            'oracle_tau': all_oracle_tau,
            'skeleton_ci_cm': all_ci_cm,
            'skeleton_cm': all_skeleton_cm}
        
    if save_results is True:
        with open(
            os.path.join(run_dir, f"{n_nodes}_{K}_{n_samples}_{avg_neighbor}_{alpha}.pkl"), "wb"
        ) as f:
            pkl.dump(res,f)
                    
    return all_cm, all_sample_tau, all_oracle_tau, res

#%%

## using run_sample_repeat, simulate for desired realizations of n_nodes, K, avg_neighbor, n_samples,alpha, and n_repeat.

n_repeat = 1

for density in [3]:
    for n_nodes in [5,6,7,8,9,10]:
        for K in [2,3,4]:
            for n_samples in [1000,2000,5000,10000]:
                for alpha in [0.05]:
                    run_sample_repeat(n_repeat=n_repeat,n_nodes=n_nodes,K=K,avg_neighbor=density,n_samples=n_samples,alpha=alpha,save_results=True,run_CI_skeleton=True)


