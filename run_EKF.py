import torch as tc
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

from data import get_data
from kalman import ExtendedKalman
from models import obs_jac_x, obs, SIR, SIR_jac_x

LATENT_DIM = 3
OBS_DIM = 1

def run():
    print("Program Running...")
    cases, N = get_data()

    #Initial conditions
    I0 = tc.tensor([1.])
    R0 = tc.tensor([0.])
    S0 = N - I0 - R0
    x = tc.stack([S0, I0, R0]) #initial latent state vector
    beta = tc.tensor([0.2])
    gamma = tc.tensor([0.1])
    params = tc.stack([beta, gamma, N]) #fixed parameters of the ODE


    P = tc.diag(tc.tensor([1., 1., 1.])).float() #initial covariance of the system


    h = partial(obs, t = None, params = params)
    H = partial(obs_jac_x, t = None, params = params)

    f = partial(SIR, t = None, params = params)
    F = partial(SIR_jac_x, t = None, params = params)


    k = ExtendedKalman(latent_dim = 3, obs_dim = 1, state_func = f, state_jac = F, obs_func = h, obs_jac = H)


    posterior = []
    cov = []
    likelihoods = []

    for y in tqdm(cases):
        x, P = k.predict(x, P)
        x, P, _, y_mean, y_cov, likelihood = k.update(x, P, tc.tensor([y]))
        k.Q = tc.eye(LATENT_DIM) * (y+1)
        k.R = tc.eye(OBS_DIM) * (y+1)
        posterior.append(y_mean)
        cov.append(y_cov)
        likelihoods.append(likelihood)

    posterior = tc.stack(posterior).flatten()
    cov = tc.stack(cov).flatten()

    f, ax = plt.subplots(2, 1, figsize=(16,10), gridspec_kw={'height_ratios': [3,1]})
    ax[0].plot(cases, '.', markersize=3, color = 'blue')
    ax[0].plot(posterior, color = 'orange')
    ax[0].fill_between(tc.arange(len(posterior)), posterior - cov.sqrt(), posterior + cov.sqrt(), color='orange', alpha=0.2)
    ax[1].plot(cov.sqrt(), '.', markersize=2)
    f.savefig("./result.png")

if __name__ == "__main__":
    run()