import tfbm

import numpy as np
import sys
import json
from multiprocessing import Pool

T = 1

TRAJECTORIES_PER_PARAMETERS = int(sys.argv[1])
TFBM_TYPE = int(sys.argv[3])
OUTPUT_DIRETORY = sys.argv[2]
PARAMETERS_CONFIG_PATH = sys.argv[4]
WORKERS = int(sys.argv[5])

with open(PARAMETERS_CONFIG_PATH) as json_file:
    parameters = json.load(json_file)

N_H = parameters['N_H']
MIN_H = parameters['MIN_H']
MAX_H = parameters['MAX_H']
N_LAMBDA = parameters['N_LAMBDA']
MIN_LAMBDA = parameters['MIN_LAMBDA']
MAX_LAMBDA = parameters['MAX_LAMBDA']

if TFBM_TYPE == 1:
    TFBM = tfbm.TFBM1
elif TFBM_TYPE == 2:
    TFBM = tfbm.TFBM2
elif TFBM_TYPE == 3:
    TFBM = tfbm.TFBM3

def generate_trajectories_for_parameter(H, lambd, n_trajectories):
    generator = TFBM(N=1000, H=H, lambd=lambd, T=T)
    try:
        trajectories =  generator.generate_samples(n_trajectories)
    except np.linalg.LinAlgError as e:
        print("LinAlgError for H: {H} and lambda: {lambd}")
        print(f"Error: {e}")

    np.save(f'{OUTPUT_DIRETORY}/TYPE_{TFBM_TYPE}_T_{T}_H_{H}_lambda_{lambd}.npy', trajectories)

def worker(params):
    H, lambd = params
    generate_trajectories_for_parameter(H, lambd, TRAJECTORIES_PER_PARAMETERS)

if __name__ == "__main__":
    params_list = [(H, lambd) for H in np.linspace(MIN_H, MAX_H, N_H) for lambd in np.linspace(MIN_LAMBDA, MAX_LAMBDA, N_LAMBDA)]
    with Pool(WORKERS) as pool:
        pool.map(worker, params_list)
