import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environment import *
from tqdm import tqdm



class ETD_LCBT:
    def __init__(self, seed, d, n):
        print('LCB-Threshold Algorithm (Explore-then-Decide)')
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.n = int(n)
        self.d = int(d)
        self.lambd = 1
        # self.beta = np.sqrt(self.d * np.log(self.n))  # Confidence parameter

        # Exploration length (can be overridden via setter)
        self.l_n = int(np.sqrt(self.n))
        self.l_n = int((self.n)**(2/3))

        # Running (exploration) accumulators: V = lambd*I + sum x x^T, b = sum y x
        self.V = self.lambd * np.eye(self.d)
        self.b = np.zeros(self.d)

        # Frozen (decision-phase) params computed once at t == l_n
        self.Vinv= None
        self.theta_hat= None
    
        # Bookkeeping
        self.stopped = False
        self.tau = None
        self.lcb_values = np.array([np.nan])
        self.rewards = np.array([0.0])

        # Placeholder: before freezing, alpha is not known
        # Keep the param but the *actual* alpha for ETD will be computed from theta_hat at t == l_n
        self.alpha = 100

    # ---------- optional: let user change l_n while keeping the original constructor ----------
    #     self.l_n = int(l_n)
    def _xi(self, x, Vinv):
        self.S=1
        self.L=1
        # ξ(x) := sqrt(x^T V^{-1} x) * ( sqrt(d * log(n + n^2 L^2 / λ)) + S * sqrt(λ) )
        # print('x',x)
        # print('Vinv',Vinv)
        rad = np.sqrt(float(x @ Vinv @ x ))
        # term = np.sqrt(self.d * np.log(self.n + (self.n**2) * (self.L**2) / self.lambd)) + self.S * np.sqrt(self.lambd)
        # term = 1
        term = np.sqrt(self.d * np.log(self.n))
        return rad * term
        
    def _compute_alpha_from_empirical_cdf(self):

        
        self.Z_dist = (lambda: np.random.uniform(0, 1/np.sqrt(self.d), size=self.d))
        sample=10000
        self.Z = [self.Z_dist() for _ in range(sample)]


        # Compute Z^{LCB} = z^T θ̂ - ξ(z)
        z_lcbs = []
        for z in self.Z:
            xi_z = self._xi(z, self.Vinv)
            z_lcbs.append(float(z @ self.theta_hat - xi_z))
        z_lcbs = np.array(z_lcbs)
        q = 1.0 - 1.0 / self.n
        q = np.clip(q, 0.0, 1.0)
        self.alpha = float(np.quantile(z_lcbs, q))
        print('alpha',self.alpha)
    def run(self, t, x, y):

        if t<=self.l_n:    
            self.V += np.outer(x, x)
            self.b += x * y
            # self._x_hist.append(x)
        if t ==self.l_n:
            self.theta_hat = np.linalg.solve(self.V, self.b)
            self.Vinv = np.linalg.inv(self.V)
            self._compute_alpha_from_empirical_cdf()

        if t>self.l_n:
            xi_x = self._xi(x, self.Vinv)
            lcb=float(x @ self.theta_hat - xi_x)

            if not self.stopped and lcb >= self.alpha:
                print(lcb, self.alpha)
                print('self.theta',self.theta_hat)
                self.tau = t
                self.stopped = True
                self.rewards = y
            elif t == self.n - 1:
                # If not stopped by threshold, stop at last round
                self.tau = self.n
                self.stopped = True
                self.rewards = y

    def get_stopping_time(self):
        return self.tau

    def get_lcb_values(self):
        return self.lcb_values

    def get_rewards(self):
        return self.rewards

    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def name(self):
        return 'ETD-LCBT(iid)'
    


class Secretary:
    def __init__(self, seed, d, n):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.y_his=[]
        self.n = int(n)
        self.d = int(d)
        self.lambd = 1

        # Exploration length (can be overridden via setter)
        self.l_n = int(np.sqrt(self.n))
    
        # Running (exploration) accumulators: V = lambd*I + sum x x^T, b = sum y x
        self.V = self.lambd * np.eye(self.d)
        self.b = np.zeros(self.d)

        # Frozen (decision-phase) params computed once at t == l_n
        self.Vinv= None
        self.theta_hat= None
    
        # Bookkeeping
        self.stopped = False
        self.tau = None
        self.rewards = np.array([0.0])
        self.y_his=[]

        self.alpha = 100

        
    def run(self, t, x, y):

        if t < math.ceil(self.n / math.e):
            self.y_his.append(y)
        elif t == math.ceil(self.n / math.e):
            self.alpha=max(self.y_his)

        else:

            if not self.stopped and y >= self.alpha:
                print(y, self.alpha)
                self.tau = t
                self.stopped = True
                self.rewards = y
            elif t == self.n - 1:
                print(y, self.alpha)
                self.tau = self.n
                self.stopped = True
                self.rewards = y

    def get_stopping_time(self):
        return self.tau

    def get_lcb_values(self):
        return self.lcb_values

    def get_rewards(self):
        return self.rewards

    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def name(self):
        return 'Gusein-Zade'

class greedy:
    def __init__(self, seed, d, n):
        print('ε-Greedy-LCBT')
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)


        self.n = int(n)
        self.d = int(d)
        self.lambd = 1.0
        self.beta = np.sqrt(self.lambd) + np.sqrt(self.d * np.log(self.n))  # Confidence parameter

        # Exploration length (can be overridden via setter)
        self.l_n = max(1, int(np.sqrt(self.n)))
        self.l_n = int((self.n)**(2/3))

        # Running (exploration) accumulators: V = lambd*I + sum x x^T, b = sum y x
        self.V = self.lambd * np.eye(self.d)
        self.b = np.zeros(self.d)

        # Frozen (decision-phase) params computed once at t == l_n
        self.Vinv= None
        self.theta_hat= None
    
        # Bookkeeping
        self.stopped = False
        self.tau = None
        self.lcb_values = np.array([np.nan])
        self.rewards = np.array([0.0])
        self.epsilon=np.sqrt(self.l_n/self.n)
        self.bool=True
        # Placeholder: before freezing, alpha is not known
        # Keep the param but the *actual* alpha for ETD will be computed from theta_hat at t == l_n
        self.alpha = 100

    # ---------- optional: let user change l_n while keeping the original constructor ----------
    #     self.l_n = int(l_n)
    def _xi(self, x, Vinv):
        self.S=1
        self.L=1
        # ξ(x) := sqrt(x^T V^{-1} x) * ( sqrt(d * log(n + n^2 L^2 / λ)) + S * sqrt(λ) )
        rad = np.sqrt(float(x @ Vinv @ x ))
        term = np.sqrt(self.d * np.log(self.n + (self.n**2) * (self.L**2) / self.lambd)) + self.S * np.sqrt(self.lambd)
        term = np.sqrt(self.d * np.log(self.n))
        return rad * term
        
    def _compute_alpha_from_empirical_cdf(self):

        
        self.Z_dist = (lambda: np.random.uniform(0, 1/np.sqrt(self.d), size=self.d))
        sample=10000
        self.Z = [self.Z_dist() for _ in range(sample)]
        z_lcbs = []
        for z in self.Z:
            xi_z = self._xi(z, self.Vinv)
            z_lcbs.append(float(z @ self.theta_hat - xi_z))
        z_lcbs = np.array(z_lcbs)
        q = 1.0 - (1.0 / self.n)
        q = np.clip(q, 0.0, 1.0)
        self.alpha = float(np.quantile(z_lcbs, q))

    def run(self, t, x, y):
        ber=np.random.binomial(1,self.epsilon)
        if ber==1 and t!=self.n-1:
            self.V += np.outer(x, x)
            self.b += x * y
            self.bool=True
        else:
            self.theta_hat = np.linalg.solve(self.V, self.b)
            self.Vinv = np.linalg.inv(self.V)
            if self.bool==True:
                self._compute_alpha_from_empirical_cdf()
                self.bool=False
            xi_x = self._xi(x, self.Vinv)
            lcb=float(x @ self.theta_hat - xi_x)

            if not self.stopped and lcb >= self.alpha:
                print(lcb, self.alpha)
                print('self.theta',self.theta_hat)
                print('x',x)
                self.tau = t
                self.stopped = True
                self.rewards = y
            elif t == self.n - 1:
                # If not stopped by threshold, stop at last round
                self.tau = self.n
                self.stopped = True
                self.rewards = y

    def get_stopping_time(self):
        return self.tau

    def get_lcb_values(self):
        return self.lcb_values

    def get_rewards(self):
        return self.rewards

    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def name(self):
        return 'ε-Greedy-LCBT'



class ETD_LCBT_NonIID:
    def __init__(self, seed, d, n):
        print('LCB-Threshold Algorithm (Explore-then-Decide)')
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.highs=np.zeros((n, d))
        self.lows=np.zeros((n, d))  
        self.n = int(n)
        self.d = int(d)
        self.lambd = 1
        # self.beta = np.sqrt(self.d * np.log(self.n))  # Confidence parameter

        # Exploration length (can be overridden via setter)
        # self.l_n = int(np.sqrt(self.n))
        self.l_n = int(self.n**(2/3))
        # self.l_n = int(np.sqrt(self.n))

        # Running (exploration) accumulators: V = lambd*I + sum x x^T, b = sum y x
        self.V = self.lambd * np.eye(self.d)
        self.b = np.zeros(self.d)

        # Frozen (decision-phase) params computed once at t == l_n
        self.Vinv= None
        self.theta_hat= None

        # Bookkeeping
        self.stopped = False
        self.tau = None
        self.lcb_values = np.array([np.nan])
        self.rewards = np.array([0.0])

        # Placeholder: before freezing, alpha is not known
        # Keep the param but the *actual* alpha for ETD will be computed from theta_hat at t == l_n
        self.alpha = 100

    # ---------- optional: let user change l_n while keeping the original constructor ----------
    #     self.l_n = int(l_n)
    def _xi(self, x, Vinv):
        self.S=1
        self.L=1
        rad = np.sqrt(float(x @ Vinv @ x ))
        # term = np.sqrt(self.d * np.log(self.n + (self.n**2) * (self.L**2) / self.lambd)) + self.S * np.sqrt(self.lambd)
        # term = 1
        term = np.sqrt(self.d * np.log(self.n))
        # term = 2
        return rad * term
        
    def _compute_alpha_from_empirical_cdf(self):
        sum=0
        sample=1000
        # sample=10000
        for k in range(sample):
            self.items = [np.random.uniform(self.lows[i], self.highs[i], size=self.d) for i in range(self.l_n,self.n)]
            sum+=(1/2)*max([x.dot(self.theta_hat) for x in self.items])
        self.alpha=(1/sample)*sum
    def run(self, t, x, y,l,h):
        self.lows=l
        self.highs=h

        if t<=self.l_n:    
            self.V += np.outer(x, x)
            self.b += x * y
            # self._x_hist.append(x)
        if t ==self.l_n:
            self.theta_hat = np.linalg.solve(self.V, self.b)
            self.Vinv = np.linalg.inv(self.V)
            self._compute_alpha_from_empirical_cdf()

        
        if t>self.l_n:
            xi_x = self._xi(x, self.Vinv)
            lcb=float(x @ self.theta_hat - xi_x)
            # print(lcb, self.alpha)

            if not self.stopped and lcb >= self.alpha:
                # print('self.theta',self.theta_hat)
                self.tau = t
                self.stopped = True
                self.rewards = y
            elif t == self.n - 1:
                # If not stopped by threshold, stop at last round
                self.tau = self.n
                self.stopped = True
                self.rewards = y


    def get_stopping_time(self):
        return self.tau

    def get_lcb_values(self):
        return self.lcb_values

    def get_rewards(self):
        return self.rewards

    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def name(self):
        return 'ETD-LCBT(non-iid)'
    



class ETD_LCBT_NonIID_Window:
    def __init__(self, seed, d, n):
        print('LCB-Threshold Algorithm (Explore-then-Decide)')
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.n = int(n)
        self.d = int(d)
        self.lambd = 1
        self.highs=np.zeros((n, d))
        self.lows=np.zeros((n, d))  
        self.l_n = int(self.n**(2/3))

        # Running (exploration) accumulators: V = lambd*I + sum x x^T, b = sum y x
        self.V = self.lambd * np.eye(self.d)
        self.b = np.zeros(self.d)
        self.x_hist=[]
        # Frozen (decision-phase) params computed once at t == l_n
        self.Vinv= None
        self.theta_hat= None

        # Bookkeeping
        self.stopped = False
        self.tau = None
        self.lcb_values = np.array([np.nan])
        self.rewards = np.array([0.0])
        self.alpha = 100

    # ---------- optional: let user change l_n while keeping the original constructor ----------
    #     self.l_n = int(l_n)
    def _xi(self, x, Vinv):
        self.S=1
        self.L=1
        rad = np.sqrt(float(x @ Vinv @ x ))
        term = np.sqrt(self.d * np.log(self.n))
        return rad * term
        
    def _compute_alpha_from_empirical_cdf(self):
        sum=0
        sample=50
        sample=10000

        for k in range(sample):
            self.items = [np.random.uniform(self.lows[i], self.highs[i], size=self.d) for i in range(self.n)]
            sum+=(1/2)*max([x.dot(self.theta_hat) for x in self.items])
        self.alpha=(1/sample)*sum
        print('alpha',self.alpha)
    def run(self, t, x, y,l,h):
        self.lows=l
        self.highs=h
        self.x_hist.append(x)
        if t<=self.l_n:    
            self.V += np.outer(x, x)
            self.b += x * y
        if t ==self.l_n+1:
            self.theta_hat = np.linalg.solve(self.V, self.b)
            self.Vinv = np.linalg.inv(self.V)
            self._compute_alpha_from_empirical_cdf()
            
            lcb=max(float(x @ self.theta_hat - self._xi(x, self.Vinv)) for x in self.x_hist)
            ind = np.argmax([float(x @ self.theta_hat - self._xi(x, self.Vinv)) for x in self.x_hist])
            if not self.stopped and lcb >= self.alpha:
                print(lcb, self.alpha)
                print('self.theta',self.theta_hat)
                self.tau = ind
                self.stopped = True
                self.rewards = y
        if t>self.l_n+1:
            xi_x = self._xi(x, self.Vinv)
            lcb=float(x @ self.theta_hat - xi_x)

            if not self.stopped and lcb >= self.alpha:
                print(lcb, self.alpha)
                print('self.theta',self.theta_hat)
                self.tau = t
                self.stopped = True
                self.rewards = y
            elif t == self.n - 1:
                # If not stopped by threshold, stop at last round
                self.tau = self.n
                self.stopped = True
                self.rewards = y

    def get_stopping_time(self):
        return self.tau

    def get_lcb_values(self):
        return self.lcb_values

    def get_rewards(self):
        return self.rewards

    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def name(self):
        return 'ETD-LCBT-WA'
    
