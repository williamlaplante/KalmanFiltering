import torch as tc

def SIR(x, t, u, params):
    S, I, R = x
    beta, gamma, N = params

    dS = - beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I

    return tc.stack([dS, dI, dR])

def obs(x, t, params):
    S, I, R = x
    beta, gamma, N = params

    return tc.stack([beta * S * I / N])

def SIR_jac_x(x, t, u, params):
    S, I, R = x
    beta, gamma, N = params

    return tc.tensor([[-beta * I / N, -beta * S / N, tc.tensor(0.)],
               [beta * I / N, beta * S / N - gamma, tc.tensor(0.)],
               [tc.tensor(0.), gamma, tc.tensor(0.)]])


def obs_jac_x(x, t, params):
    """
    counts y are modeled as follows :
    E[y] = h(S, I, R ; beta, gamma, N) = beta * S * I / N

    Hence,
    jac = [beta * I / N, beta * S / N, 0.]
    """
    S, I, R = x
    beta, gamma, N = params
    return tc.stack([beta * I / N, beta * S / N, tc.tensor([0.])]).T