import torch as tc
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial


LATENT_DIM = 3
OBS_DIM = 1

def SIR(x, t, u, params):
    S, I, R = x
    beta, gamma, N = params

    dS = - beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I

    return tc.stack([dS, dI, dR])

def SIR_jac(x, t, u, params):
    S, I, R = x
    beta, gamma, N = params
    return tc.tensor([[-beta * I / N, -beta * S / N, tc.tensor(0.)],
                      [beta * I / N, beta * S / N - gamma, tc.tensor(0.)],
                      [tc.tensor(0.), gamma, tc.tensor(0.)]])

def meas(x, t, u, params):
    S, I, R, dS, dI, dR = x
    return tc.stack([dS, dI, dR]) - SIR(tc.stack([S, I, R]), t, u, params)

def meas_jac(x, t, u, params):
    S, I, R = x
    return tc.eye(len(x)).float() - SIR_jac(tc.stack([S, I, R]), t, u, params)

def run():
    print("Program Running...")

    #Initial conditions
    N = tc.tensor([1e6]).float()
    I0 = tc.tensor([1.])
    R0 = tc.tensor([0.])
    S0 = N - I0 - R0
    x0 = tc.stack([S0, I0, R0]) #initial latent state vector

    #ODE parameters
    beta = tc.tensor([0.2])
    gamma = tc.tensor([0.1])
    params = tc.stack([beta, gamma, N]) #fixed parameters of the ODE

    P = tc.diag(tc.tensor([0.1, 0.1, 0.1])).float() #initial covariance of the system

    h = partial(meas, t = None, params = params)
    H = partial(meas_jac, t = None, params = params)

    f = partial(SIR, t = None, params = params)
    F = partial(SIR_jac, t = None, params = params)

    Q = tc.eye(3).float()
    R = tc.eye(3).float()

    n_steps = 200
    x = [x0] + (n_steps-1) * [0.0]
    for k in tqdm(range(1, n_steps)):
        #prior prediction
        x_prior = f(x=x[k-1], u=None)
        P = F(x=x[k-1], u=None) @ P @ F(x=x[k-1], u=None).T + Q 

        meas_res = h(tc.cat([x[k-1], x_prior]), u=None) # residuals of x prior minus ODE of x_{k-1}
        K = P @ H(x=x_prior, u=None).T @ tc.inverse(R + H(x=x_prior, u=None) @ P @ H(x=x_prior, u=None).T)

        x[k] = x_prior + K @ meas_res
        P = P - K @ H(x=x_prior, u=None) @ P

    x = tc.stack(x)
    plt.plot(x[:, 0])
    plt.show()
if __name__ == "__main__":
    run()