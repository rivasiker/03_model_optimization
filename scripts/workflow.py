from gwf import Workflow
import sys
import numpy as np
from os.path import exists

gwf = Workflow()

t_1 = 2e5
t_2 = 3e4
t_3 = 5e5 
N_AB = 50000
N_ABC = 40000
r = 2e-9
mu = 2e-8

for n_int in [1, 3, 5, 7, 9, 11, 13]:
    for c in [2, 4, 6, 8, 12, 16]:
        gwf.target('benchmark_{}_{}'.format(n_int, c), 
                inputs=[], 
                outputs=['../results/bench_{}_{}.csv'.format(n_int, c)],  
                cores=c,
                memory='10g',
                walltime= '1:00:00',
                account='Primategenomes') << """
        python optimize.py {} {} {} {} {} {} {} {} {} {} {}
        """.format(0, t_1, t_2, t_3, N_AB, N_ABC, r, mu, n_int, n_int, c)

for n_int in [1, 3, 5]:
    for seed in range(10):
        gwf.target('simulate_{}_{}'.format(n_int, seed), 
                inputs=[], 
                outputs=['../results/{}_{}_{}.csv'.format(x, n_int, seed) for x in ['dat', 'sim']],  
                cores=n_int,
                memory='{}g'.format(n_int*2),
                walltime= '{}:00:00'.format(n_int*2),
                account='Primategenomes') << """
        python optimize_2.py {} {} {} {} {} {} {} {} {} {}
        """.format(seed, t_1, t_2, t_3, N_AB, N_ABC, r, mu, n_int, n_int)
