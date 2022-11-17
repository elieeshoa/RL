import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost. 
        Parameter:
            s (1D numpy array) with shape (4,) 
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation


    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        assert s_star.shape == (4,) # state has 4 dimensions
        assert a_star.shape == (1,) # action has 1 dimension
        assert T > 0
        
        A = jacobian(lambda s: self.f(s, a_star), s_star)
        B = jacobian(lambda a: self.f(s_star, a), a_star)

        r = gradient(lambda a: self.c(s_star, a), a_star)
        q = gradient(lambda s: self.c(s, a_star), s_star)

        n_s = len(s_star)

        H = hessian(lambda x: self.c(x[:n_s], np.array([x[n_s]])), np.append(s_star, a_star))
        evals, evecs = np.linalg.eig(H)
        evals[evals<0] = 0
        H_new = evecs * evals @ np.linalg.inv(evecs) + (1e-7) * np.identity(n_s + 1)

        Q = H_new[:n_s, :n_s]
        R = np.array([H_new[n_s, n_s]])
        M = np.array([H_new[:n_s, -1]]).T

        Q_2 = Q/2
        R_2 = np.array([R/2])

        q_2 = np.array([q.T - s_star.T @ Q - a_star.T @ M.T]).T
        r_2 = np.array([r.T - a_star.T @ R - s_star.T @ M]).T

        b = np.array([self.c(s_star, a_star)\
            + (1/2) * s_star.T @ (Q/2) @ s_star \
            + (1/2) * np.array([a_star]).T @ R @ a_star \
            + s_star.T @ M @ a_star \
            - q.T @ s_star \
            - r.T @ a_star])

        m = np.array([self.f(s_star, a_star) - A @ s_star - B @ a_star]).T

        Ks = lqr(A, B, m, Q_2, R_2, M, q_2, r_2, b, T)

        return Ks


class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a
