import sys
from ilsmc.optimizer import trans_emiss_calc
from ilsmc.cutpoints import cutpoints_ABC
from numba import njit
import numpy as np
from ilsmc.optimizer import forward_loglik, write_list, optimizer_no_mu
import pandas as pd
import time
import multiprocessing as mp
import os

try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
    ncpus = mp.cpu_count()

print(ncpus)

####################### Model parameters #######################

seed = int(sys.argv[1])
t_1 = float(sys.argv[2])
t_2 = float(sys.argv[3])
t_3 = float(sys.argv[4])
N_AB = float(sys.argv[5])
N_ABC = float(sys.argv[6])
r = float(sys.argv[7])
mu = float(sys.argv[8])
n_int_AB = int(sys.argv[9])
n_int_ABC = int(sys.argv[10])

coal_ABC = N_ABC/N_AB
t_upper = t_3-cutpoints_ABC(n_int_ABC, coal_ABC)[-2]*N_AB

transitions, emissions, starting, hidden_states, observed_states = trans_emiss_calc(
    t_1, t_2, t_upper, 
    N_AB, N_ABC, 
    r, mu, n_int_AB, n_int_ABC)


####################### Simulation #######################

@njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@njit
def simulate(n_sim, transitions, emissions, starting, hid, obs, seed):
    np.random.seed(seed)
    H = np.zeros(n_sim, dtype = np.int16)
    E = np.zeros(n_sim, dtype = np.int16)
    h = rand_choice_nb(
        list(range(hid)),
        starting
    )
    H[0] = h
    e = rand_choice_nb(
        list(range(obs)),
        emissions[H[0]]
    )
    E[0] = e
    for i in range(1, n_sim):
        h = rand_choice_nb(
            list(range(hid)),
            transitions[H[i-1]]
        )
        e = rand_choice_nb(
            list(range(obs)),
            emissions[h]
        )
        E[i] = e
        H[i] = h
    return E, H

start = time.time()
E, H = simulate(10000000, transitions, emissions, starting, len(hidden_states), len(observed_states), seed)
end = time.time()
print("Simulation = %s" % (end - start))

pd.DataFrame({'E':E,'H':H}).to_csv('../results/dat_{}_{}.csv'.format(n_int_AB, seed), index = False)

####################### Optimization #######################

start = time.time()
forward_loglik(transitions, emissions, starting, E)
end = time.time()
print("Likelihood = %s" % (end - start))

write_list([-1, t_1, t_2, t_upper, N_AB, N_ABC, r, forward_loglik(transitions, emissions, starting, E)], '../results/sim_{}_{}'.format(n_int_AB, seed))

np.random.seed(seed)

t_init_1 = np.random.normal(t_1, t_1/5)
t_init_2 = np.random.normal(t_2, t_2/5)
t_init_upper = np.random.normal(t_upper, t_upper/5)
N_init = np.random.normal(np.mean([N_AB, N_ABC]), np.mean([N_AB, N_ABC])/5)
r_init = np.random.normal(r, r/5)
mu_init = mu

res = optimizer_no_mu(
    t_init_1, t_init_2, t_init_upper, 
    N_init, N_init, 
    r_init, mu_init, 
    n_int_AB, n_int_ABC, E, '../results/sim_{}_{}'.format(n_int_ABC, seed), False
)
