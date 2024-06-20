# intervention-mixture-DAG
Codebase for the paper "[Interventional Causal Discovery in a Mixture of DAGs](https://arxiv.org/abs/2406.08666)"

### Requirements
For generating and manipulating graphs, it uses networkx and causaldag packages
```
$ pip3 install networkx
$ pip3 install causaldag
```
### Reproducing the results
To reproduce the simulation results reported in the paper, run `run_simulations.py`.
To generate the plots from the simulations, run `plots.py`.
