"""
Generate plots for simulations of Interventional Causal Discovery in a Mixture of DAGs 
"""
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter


if not os.path.exists('./plots'):
    os.makedirs('./plots')

xticks_size = 16
yticks_size = 16
xlabel_size = 18
ylabel_size = 18
legend_size = 16
legend_loc = 'lower right'
linewidth = 3
linestyle = '--'
markersize = 10

def load_res(n_nodes,K,n_samples,avg_neighbor,alpha,run_dir='./results/'):
    path = os.path.join(run_dir, f"{n_nodes}_{K}_{n_samples}_{avg_neighbor}_{alpha}.pkl")
    f = open(path,'rb')
    res = pkl.load(f)
    f.close
    return res

#%%
n_nodes_list = [5,6,7,8,9,10]
K_list = [2,3,4]
n_samples_list = [1000,2000,5000,10000]
avg_neighbor = 3
alpha = 0.05

#%% 
## Vary n_nodes and K for fixed n_sample = 5000
# 7 dimensions for tp, fp, fn, precision, recall, oracle_tau and sample_tau
mat_n_K = np.zeros((len(n_nodes_list),len(K_list),7))

for idx_n_nodes in range(len(n_nodes_list)):
    for idx_K in range(len(K_list)):
        res = load_res(n_nodes=n_nodes_list[idx_n_nodes],K=K_list[idx_K],n_samples=5000,avg_neighbor=3,alpha=0.05)

        tp, fp, fn = np.sum(res['sample_cm'],0)
        precision = tp / (tp+fp)
        recall = tp / (tp + fn)
        mean_oracle_tau = np.mean(res['oracle_tau'])
        mean_sample_tau = np.mean(res['sample_tau'])
        
        mat_n_K[idx_n_nodes,idx_K] = [tp, fp, fn, precision, recall, mean_oracle_tau, mean_sample_tau]


## Vary n_nodes and n_samples for fixed K = 3  

mat_n_s = np.zeros((len(n_nodes_list),len(n_samples_list),7))

for idx_n_nodes in range(len(n_nodes_list)):
    for idx_s in range(len(n_samples_list)):
        res = load_res(n_nodes=n_nodes_list[idx_n_nodes],K=3,n_samples=n_samples_list[idx_s],avg_neighbor=3,alpha=0.05)

        tp, fp, fn = np.sum(res['sample_cm'],0)
        precision = tp / (tp+fp)
        recall = tp / (tp + fn)
        mean_oracle_tau = np.mean(res['oracle_tau'])
        mean_sample_tau = np.mean(res['sample_tau'])
        
        mat_n_s[idx_n_nodes,idx_s] = [tp, fp, fn, precision, recall, mean_oracle_tau, mean_sample_tau]


#%%  PLOT FOR VARYING n_nodes and n_samples for fixed K
# Figure 1a

mat_n_s_precision = mat_n_s[:,:,3]
mat_n_s_recall = mat_n_s[:,:,4]
mat_n_s_oracle_tau = mat_n_s[:,:,5]
mat_n_s_sample_tau = mat_n_s[:,:,6]

plt.figure()
plt.plot(n_samples_list,mat_n_s_precision[0],'-gD',markersize=markersize,label='n=5 - precision',linewidth=linewidth,linestyle=linestyle)
#plt.plot(n_samples_list,mat_n_s_precision[3],'-mD',markersize=markersize,label='n=8 - precision',linewidth=linewidth,linestyle=linestyle)
plt.plot(n_samples_list,mat_n_s_precision[5],'-mD',markersize=markersize,label='n=10 - precision',linewidth=linewidth,linestyle=linestyle)

plt.plot(n_samples_list,mat_n_s_recall[0],'-bX',markersize=markersize,label='n=5 - recall',linewidth=linewidth,linestyle=linestyle)
#plt.plot(n_samples_list,mat_n_s_recall[3],'-mX',markersize=markersize,label='n=8 - recall',linewidth=linewidth,linestyle=linestyle)
plt.plot(n_samples_list,mat_n_s_recall[5],'-rX',markersize=markersize,label='n=10 - recall',linewidth=linewidth,linestyle=linestyle)

plt.xscale('log')
plt.ylim([0.8,1])
plt.ylabel('Precision & recall of true edges',size=ylabel_size)
plt.xlabel('Number of samples',size=xlabel_size)
plt.yticks([0.80,0.85,0.90,0.95,1.0],fontsize=yticks_size)
plt.xticks(n_samples_list,fontsize=xticks_size)
plt.legend(fontsize=legend_size,loc='lower right')
plt.tight_layout()
plt.grid()
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
plt.savefig('./plots/n_s_precision_recall.pdf')


#%%  PLOT FOR VARYING n_nodes and K for fixed sample size
# Figure 4

mat_n_K_precision = mat_n_K[:,:,3]
mat_n_K_recall = mat_n_K[:,:,4]
mat_n_K_oracle_tau = mat_n_K[:,:,5]
mat_n_K_sample_tau = mat_n_K[:,:,6]

plt.figure()

plt.plot(n_nodes_list,mat_n_K_recall[:,0],'-gX',markersize=markersize,label='K=2',linewidth=linewidth,linestyle=linestyle)
plt.plot(n_nodes_list,mat_n_K_recall[:,1],'-mX',markersize=markersize,label='K=3',linewidth=linewidth,linestyle=linestyle)
plt.plot(n_nodes_list,mat_n_K_recall[:,2],'-bX',markersize=markersize,label='K=4',linewidth=linewidth,linestyle=linestyle)

plt.ylim([0.9,1])
plt.ylabel('Recall of true edges',size=ylabel_size)
plt.xlabel('Number of nodes',size=xlabel_size)
plt.yticks(fontsize=yticks_size)
plt.xticks(fontsize=xticks_size)
plt.legend(fontsize=legend_size,loc='lower left')
plt.tight_layout()
plt.grid()
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
plt.savefig('./plots/n_K_precision_recall.pdf')

#%% PLOT FOR cyclic complexity (true and estimated ones) for varying n_nodes and K
# Figure 2c

plt.figure()
plt.plot(n_nodes_list,mat_n_K_oracle_tau[:,0],'-bD',markersize=markersize,label='K=2 - true')
plt.plot(n_nodes_list,mat_n_K_oracle_tau[:,1],'-bo',markersize=markersize,label='K=3 - true')
plt.plot(n_nodes_list,mat_n_K_oracle_tau[:,2],'-bX',markersize=markersize,label='K=4 - true')

plt.plot(n_nodes_list,mat_n_K_sample_tau[:,0],'-rD',markersize=markersize,label='K=2 - est.')
plt.plot(n_nodes_list,mat_n_K_sample_tau[:,1],'-ro',markersize=markersize,label='K=3 - est.')
plt.plot(n_nodes_list,mat_n_K_sample_tau[:,2],'-rX',markersize=markersize,label='K=4 - est.')


plt.ylim([0,4])
plt.ylabel('Cyclic complexity',size=ylabel_size)
plt.xlabel('Number of nodes',size=xlabel_size)
plt.yticks(fontsize=yticks_size)
plt.xticks(fontsize=xticks_size)
plt.legend(fontsize=legend_size,loc='upper left',ncol=2)
plt.tight_layout()
plt.grid()
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
plt.savefig('./plots/n_K_tau.pdf')

#%%
#%% Figure 2b.
# CI test based skeleton recovery comparison
n_nodes_list = [5,6,7,8,9,10]
K = 2
n_samples = 10000
avg_neighbor = 3
alpha = 0.05

mat_skel = np.zeros((len(n_nodes_list),6))

for idx_n_nodes in range(len(n_nodes_list)):
    res = load_res(n_nodes=n_nodes_list[idx_n_nodes],K=K,n_samples=n_samples,avg_neighbor=avg_neighbor,alpha=alpha,run_dir='./results/')

    skel_tp, skel_fp, skel_fn = np.sum(res['skeleton_cm'],0)
    skel_precision = skel_tp / (skel_tp + skel_fp)
    skel_recall = skel_tp / (skel_tp + skel_fn)

    ci_skel_tp, ci_skel_fp, ci_skel_fn = np.sum(res['skeleton_ci_cm'],0)
    ci_skel_precision = ci_skel_tp / (ci_skel_tp + ci_skel_fp)
    ci_skel_recall = ci_skel_tp / (ci_skel_tp + ci_skel_fn)
    
    mat_skel[idx_n_nodes] = [skel_tp, skel_fp, skel_fn, ci_skel_tp, ci_skel_fp, ci_skel_fn]


skel_f1 = 2*mat_skel[:,0] / (2*mat_skel[:,0] + mat_skel[:,1] + mat_skel[:,2])
skel_ci_f1 = 2*mat_skel[:,3] / (2*mat_skel[:,3] + mat_skel[:,4] + mat_skel[:,5])

plt.figure()
plt.plot(n_nodes_list,skel_f1,'-bD',markersize=markersize,label="Ours (intervention-based)", linewidth=linewidth,linestyle=linestyle)
plt.plot(n_nodes_list,skel_ci_f1,'-rX',markersize=markersize,label="Observation-based", linewidth=linewidth,linestyle=linestyle)

#plt.ylim([0.8,1])
plt.ylabel('F1 score',size=ylabel_size)
plt.xlabel('Number of nodes',size=xlabel_size)
plt.yticks(fontsize=yticks_size)
plt.xticks(fontsize=xticks_size)
plt.legend(fontsize=legend_size,loc='lower left')
plt.tight_layout()
plt.grid()
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
plt.savefig('./plots/skel_comparison.pdf')
