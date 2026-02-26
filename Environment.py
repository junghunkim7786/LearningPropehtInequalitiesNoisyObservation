import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli
from scipy import stats


class ProphetInequalityEnv:
    """
    Environment for the prophet inequality with linear rewards and noisy feedback.

    Args:
        d (int): Dimension of the preference vector and items.
        n (int): Total number of rounds/items.
        m (int): Number of feedback rounds allowed (m <= n).
        theta (np.ndarray): User's true preference vector (optional, random if None).
        item_dist (callable): Function to sample items from D_t (optional, standard normal if None).
        noise_std (float): Standard deviation of sub-Gaussian noise (default 1.0).
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, seed, d, n,noise_std):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.d = d
        self.n = n
        # self.m = m
        self.norm=1/np.sqrt(d)
        # self.norm=0.5
        self.noise_std = noise_std
        self.theta = np.random.uniform(0,self.norm, size=d)
        # def _normalize_to_radius(x):
        #     nrm = np.linalg.norm(x)
        #     if nrm == 0:
        #         return x
        #     return (x / nrm) * self.norm

        # --- NEW: small non-negative samplers (match your non-neg reward assumption) ---
        # def _sample_uniform():
        #     return _normalize_to_radius(np.random.uniform(0.0, self.norm, size=d))

        # def _sample_abs_gaussian():
        #     return _normalize_to_radius(np.abs(np.random.normal(0.0, 1.0, size=d)))

        # def _sample_abs_student_t():
        #     return _normalize_to_radius(np.abs(np.random.standard_t(df=student_t_df, size=d)))

        # def _sample_exponential():
        #     return _normalize_to_radius(np.random.exponential(scale=1.0, size=d))




        self.item_dist = (lambda: np.random.uniform(0, self.norm, size=d))
        # self.item_dist = (lambda: _normalize_to_radius(np.abs(np.random.normal(0.0, 1.0, size=d))))


        self.items = [self.item_dist() for _ in range(n)]
        self.feedback_indices = set()
        self.feedback = {}
        self.stopped = False
        self.tau = None

    def get_item(self, t):
        """Return the t-th item (0-based)."""
        return self.items[t]

    def get_optimal_reward(self):
        """Return the optimal expected reward (oracle)."""
        return np.max([x.dot(self.theta) for x in self.items])

    def recommend_and_feedback(self, t):
        # """
        # Recommend item at round t and get noisy feedback.
        # Only allowed up to m times.
        # """
        # if len(self.feedback_indices) >= self.m:
        #     raise Exception("Feedback budget exceeded")
        # if t in self.feedback_indices:
        #     raise Exception("Already queried feedback for this item")
        x = self.items[t]
        noise = np.random.normal(0, self.noise_std)
        feedback = x.dot(self.theta) + noise
        self.feedback_indices.add(t)
        self.feedback[t] = feedback
        return feedback

    def stop_and_choose(self, t):
        """
        Stop at round t and select item t as final choice.
        """
        if self.stopped:
            raise Exception("Already stopped")
        self.stopped = True
        self.tau = t
        return self.items[t].dot(self.theta)

    # def get_regret(self):
    #     """Return the regret of the final choice."""
    #     if not self.stopped:
    #         raise Exception("Haven't stopped yet")
    #     return self.get_optimal_reward() - self.items[self.tau].dot(self.theta)

    def get_ratio(self):
        """Return the ratio of achieved to optimal reward."""
        # if not self.stopped:
        #     raise Exception("Haven't stopped yet")
        return self.items[self.tau].dot(self.theta) / self.get_optimal_reward()


        
class Noniid_ProphetInequalityEnv:
    """
    Environment for the prophet inequality with linear rewards and noisy feedback.

    Args:
        d (int): Dimension of the preference vector and items.
        n (int): Total number of rounds/items.
        m (int): Number of feedback rounds allowed (m <= n).
        theta (np.ndarray): User's true preference vector (optional, random if None).
        item_dist (callable): Function to sample items from D_t (optional, standard normal if None).
        noise_std (float): Standard deviation of sub-Gaussian noise (default 1.0).
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, seed, d, n,noise_std):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.d = d
        self.n = n
        # self.m = m
        self.noise_std = noise_std
        self.norm=1/np.sqrt(d)
        # self.norm=1
        # self.theta_dist = (lambda: np.random.uniform(0, 1/np.sqrt(d), size=d))
        self.theta = np.random.uniform(0, self.norm, size=d)
        self.lows= np.zeros((n, d))
        self.highs= np.zeros((n, d))

   
        # mus = np.zeros((n, d))
        # for i in range(n):
        #     scale = (n - i) / n   
        #     mus[i] = np.random.uniform(0.9*scale / np.sqrt(d), scale / np.sqrt(d), size=d)
           
        mus = np.zeros((n, d))
        for i in range(n):
        #     scale = (n - i) / n   
            mus[i] = np.random.uniform(0,self.norm, size=d)
        # widths= np.random.uniform(0, 1/np.sqrt(d), size=1)

        widths = np.random.uniform(0, self.norm,  size=(n, d))   
        self.lows  = np.clip(mus - widths/2, 0, self.norm)
        self.highs = np.clip(mus + widths/2, 0, self.norm)

        # self.lows  = np.clip(mus, 0, 1/np.sqrt(d))
        # self.highs = np.clip(mus, 0, 1/np.sqrt(d))
        self.items = [ np.random.uniform(self.lows[i], self.highs[i], size=d) for i in range(n)]
        # for i in range(n):
        #     print(self.lows[i],self.highs[i])
        # self.item_dist = (lambda: np.random.uniform(0, 1/np.sqrt(d), size=d))
        # self.items = [self.item_dist() for _ in range(n)]
        self.feedback_indices = set()
        self.feedback = {}
        self.stopped = False
        self.tau = None

    def get_item(self, t):
        """Return the t-th item (0-based)."""
        return self.items[t]
    def get_inform_dis(self):
        return self.lows, self.highs

    def get_optimal_reward(self):
        """Return the optimal expected reward (oracle)."""
        return np.max([x.dot(self.theta) for x in self.items])

    def recommend_and_feedback(self, t):
        # """
        # Recommend item at round t and get noisy feedback.
        # Only allowed up to m times.
        # """
        # if len(self.feedback_indices) >= self.m:
        #     raise Exception("Feedback budget exceeded")
        # if t in self.feedback_indices:
        #     raise Exception("Already queried feedback for this item")
        x = self.items[t]
        noise = np.random.normal(0, self.noise_std)
        feedback = x.dot(self.theta) + noise
        self.feedback_indices.add(t)
        self.feedback[t] = feedback
        return feedback

    def stop_and_choose(self, t):
        """
        Stop at round t and select item t as final choice.
        """
        if self.stopped:
            raise Exception("Already stopped")
        self.stopped = True
        self.tau = t
        return self.items[t].dot(self.theta)

    # def get_regret(self):
    #     """Return the regret of the final choice."""
    #     if not self.stopped:
    #         raise Exception("Haven't stopped yet")
    #     return self.get_optimal_reward() - self.items[self.tau].dot(self.theta)

    def get_ratio(self):
        """Return the ratio of achieved to optimal reward."""
        # if not self.stopped:
        #     raise Exception("Haven't stopped yet")
        return self.items[self.tau].dot(self.theta) / self.get_optimal_reward()
