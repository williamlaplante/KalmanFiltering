import torch as tc


#https://arxiv.org/pdf/1204.0375.pdf 

#https://www.intechopen.com/chapters/63164 

class Kalman():
    """
    Kalman Filter : 

    x_t = F * x_{t-1} + B * u_{t-1}  + w_{t-1} (latent equation)

    y_t = H * x_t + v_t (observation equation)

    where w_t ~ N(0, Q), v_t ~ N(0, R)


    F : state transition matrix
    B : control-input matrix applied to control vector

    H : measurement matrix

    """
    def __init__(self, latent_dim, obs_dim, H) -> None:
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        #Noise matrices
        self.Q = tc.eye(latent_dim) #Latent process noise
        self.R = tc.eye(obs_dim) #Observation process noise

        #Transition matrices
        self.F = tc.eye(latent_dim) #State transition matrix
        self.B = tc.eye(latent_dim) #Control-input matrix
        self.H = H #Observation/Measurement matrix

        if self.H.size() != tc.Size((obs_dim, latent_dim)):
            raise Exception("Observation matrix H must be of size (%d, %d)"%(obs_dim, latent_dim))
        return

        
    
    def predict(self, x : tc.tensor, P : tc.tensor, u = None):

        if u==None: u = tc.zeros_like(x)

        #compute prior estimate of state
        x_p = tc.matmul(self.F, x) + tc.matmul(self.B, u) #x_t = F * x_{t-1} + B * u_{t-1}

        #compute prior estimate of state covariance
        P_p = tc.matmul(self.F, tc.matmul(P, self.F.T)) + self.Q # P_t = F_{t-1} * P_{t-1} * F^T_{t-1} + Q_{t-1}

        return x_p, P_p

    def update(self, x, P, y):
        y_mean = tc.matmul(self.H, x) #predicted y based on predicted x (prior, not posterior) at time t
        y_cov = tc.matmul(self.H, tc.matmul(P, self.H.T)) + self.R #covariance based on prior error covariance
        likelihood = tc.distributions.MultivariateNormal(y_mean, y_cov).log_prob(y) #compute the likelihood of y

        residuals = y - y_mean #measurement residuals
        K = tc.matmul(P, tc.matmul(self.H.T, tc.inverse(y_cov))) #kalman gain

        x = x + tc.matmul(K, residuals) #posterior estimate of state

        P = P - tc.matmul(K, tc.matmul(self.H, P)) #posterior estimate of covariance

        return x, P, K, y_mean, y_cov, likelihood


class DiscreteExtendedKalman():
    """
    Discrete/Discrete EKF : 

    x_t = f(x_{t-1}, u_{t-1}) + w_{k-1}

    y_t = h(x_t) + v_t

    where f() and h() are non-linear functions of the state x. This forces us to take a linear approximation via jacobians.
    
    """
    def __init__(self, latent_dim, obs_dim, state_func, state_jac, obs_func, obs_jac) -> None:
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
    
        #noise matrices for the observation and latent equations
        self.Q = tc.eye(latent_dim)
        self.R = tc.eye(obs_dim)

        #state transition function and its jacobian. This function should take (x, u) as its argument.
        self.f = state_func
        self.F = state_jac

        #control matrix
        self.B = tc.eye(latent_dim)

        #observation function and its jacobian. This function should take (x) as its argument.
        self.h = obs_func
        self.H = obs_jac
    
    def predict(self, x, P, u = None):
        if u==None: u = tc.zeros_like(x)
        #compute prior estimate of state
        x_p = self.f(x=x, u=u)

        #compute prior estimate of state covariance
        P = tc.matmul(self.F(x=x, u=u), tc.matmul(P, self.F(x=x, u=u).T)) + self.Q # P_t = A_{t-1} * P_{t-1} * A^T_{t-1} + Q_{t-1}

        return x_p, P

    def update(self, x, P, y):

        y_mean = self.h(x=x) #predicted y based on prior predicted x at time t
        y_cov = tc.matmul(self.H(x=x), tc.matmul(P, self.H(x=x).T)) + self.R #covariance based on prior error covariance
        likelihood = tc.distributions.MultivariateNormal(y_mean, y_cov).log_prob(y)

        residuals = y - y_mean #measurement residuals
        K = tc.matmul(P, tc.matmul(self.H(x=x).T, tc.inverse(y_cov))) #kalman gain

        x = x + tc.matmul(K, residuals) #posterior estimate of state

        P = P - tc.matmul(K, tc.matmul(self.H(x=x), P)) #posterior estimate of covariance

        return x, P, K, y_mean, y_cov, likelihood


class ContinuousExtendedKalman():
    def __init__(self, latent_dim, obs_dim, state_func, state_jac, obs_func, obs_jac) -> None:
        
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        self.delta_y = 1.0
        self.dt = 0.01
        
        #noise matrices for the observation and latent equations
        self.Q = tc.eye(latent_dim)
        self.R = tc.eye(obs_dim)

        #state transition function and its jacobian. This function should take (x, u) as its argument.
        self.f = state_func
        self.F = state_jac

        #control matrix
        self.B = tc.eye(latent_dim)

        #observation function and its jacobian. This function should take (x) as its argument.
        self.h = obs_func
        self.H = obs_jac
    
    def predict(self, x, P, u = None):
        if u==None: u = tc.zeros_like(x)
        #compute prior estimate of state
        x_p = self.f(x=x, u=u)

        #compute prior estimate of state covariance
        P = tc.matmul(self.F(x=x, u=u), tc.matmul(P, self.F(x=x, u=u).T)) + self.Q # P_t = A_{t-1} * P_{t-1} * A^T_{t-1} + Q_{t-1}

        return x_p, P

    def update(self, x, P, y):

        y_mean = self.h(x=x) #predicted y based on prior predicted x at time t
        y_cov = tc.matmul(self.H(x=x), tc.matmul(P, self.H(x=x).T)) + self.R #covariance based on prior error covariance
        likelihood = tc.distributions.MultivariateNormal(y_mean, y_cov).log_prob(y)

        residuals = y - y_mean #measurement residuals
        K = tc.matmul(P, tc.matmul(self.H(x=x).T, tc.inverse(y_cov))) #kalman gain

        x = x + tc.matmul(K, residuals) #posterior estimate of state

        P = P - tc.matmul(K, tc.matmul(self.H(x=x), P)) #posterior estimate of covariance

        return x, P, K, y_mean, y_cov, likelihood
