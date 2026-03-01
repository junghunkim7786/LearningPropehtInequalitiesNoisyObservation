import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environment import *
from tqdm import tqdm
from scipy.optimize import fmin_tnc
from scipy.optimize import minimize
from scipy.linalg import cholesky
from itertools import product
import traceback
import copy
#################    
    
    
class UCB:
    def loss(self, theta, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
        obj=self.g@(theta-theta_prev)+(1/2)*(theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev)    
        return obj 
    
    def prob(self,i,S,theta):

        means = np.dot(self.z[S], theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(self.r)
        prob = np.exp(np.dot(self.z[i], theta)) / SumExp
        
        return prob

    def G(self,S,theta):
        gram=np.zeros((self.r,self.r))
        for i in S:
            gram+=self.prob(i,S,theta)*np.outer(self.z[i],self.z[i])
            for j in S:
                gram-=self.prob(i,S,theta)*self.prob(j,S,theta)*np.outer(self.z[i],self.z[j])
        return gram
    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta[0:self.r])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.r] = theta[0:self.r] / norm_theta
            
        return theta
        
    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        # print('S',S)
        # print('theta_prev',theta_prev)
        # print('x',x)
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(self.r)
        for i,n in enumerate(S):
            # print('theta_prev',theta_prev)
            # print('x[n]',x[n])
            # print('y[i]',y[i])
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
        if len(S)>0:
            iter = 0
            theta=copy.deepcopy(theta_prev)
            while True:
                theta_=copy.deepcopy(theta)
                grad=self.g+1/self.eta*np.linalg.inv(V)@(theta-theta_prev)
                theta = theta - 0.01*grad
                iter += 1
                if np.linalg.norm(theta-theta_) < 1e-5 or iter > 15:
                    break
            theta=self.project_to_unit_ball(theta)
        else:
            theta=theta_prev

        return theta       
    def g(self,theta,S_k, lambd):
        result = 0.0
        for S in S_k:
            x=self.z[S]
            means = np.dot(x, theta)
            u = np.exp(means)
            SumExp = np.sum(u)+1
            for n in S:
                p = np.exp(np.dot(self.z[n], theta)) / SumExp
                result += p
        result += lambd * np.linalg.norm(theta)
        return result

    def distance_function(self,theta, S_k, lambd, theta_hat):
        g_theta = self.g(theta,S_k, lambd)
        g_theta_hat = self.g(theta_hat, S_k, lambd)
        return np.linalg.norm(g_theta - g_theta_hat)
    
    
    # def projection(self,theta,S_k,sim=True):
    #     if sim==True:
    #         optimized_theta=theta/np.sqrt(np.sum(theta**2))
    #     else:
    #         lambd=1
    #         initial_theta_guess = np.zeros((self.r,))  # Replace r with the actual dimension of theta

    #         # Optimization using scipy minimize
    #         result = minimize(
    #             lambda theta: self.distance_function(theta, S_k, lambd, initial_theta_guess),
    #             initial_theta_guess,
    #             constraints={'type': 'eq', 'fun': lambda theta: np.linalg.norm(theta) - 1},
    #             bounds=[(-1, 1) for _ in range(self.r)]  # Replace r with the actual dimension of theta
    #         )

    #         # The optimized parameter vector
    #         optimized_theta = result.x
    #     return optimized_theta 

    
    
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
                A_[n].append(k)           
        A_=[a if a else [None] for a in A_]

        combinations=list(product(*A_))
        for combination in combinations:
            S=[[] for _ in range(self.K)]
            for n in range(len(combination)):
                if combination[n]!=None: 
                    S[combination[n]].append(n)
            if all(len(sublist) <= self.L for sublist in S):
                M.append(S)
        return M              

    def compute_prob_loss(self, theta, x, y):
        means = np.dot(x, theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        if 1 in y:
            prob = u / SumExp
        else: 
            prob = 1 / SumExp
        return prob    
    
    def compute_prob_grad(self, theta, x, y):
        means = np.dot(x, theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        prob = u / SumExp

        return prob        
    def cost_function(self, theta, *args):
        x, y, S= args[0], args[1], args[2]
        loss=0
        for t in range(len(S)):
            x_=x[S[t]]
            y_=y[t]
            prob = self.compute_prob_loss(theta, x_, y_)
            loss+=-np.sum(np.multiply(y_, np.log(prob)))
        return loss + (1/2)*self.lamb* np.linalg.norm(theta)**2

    def gradient(self, theta, *args):
        x, y, S = args[0], args[1], args[2]
        m = 1
        grad=0
        for t in range(len(S)):
            if len(S[t])!=0:
                x_=x[S[t]]
                y_=y[t]
                prob = self.compute_prob_grad(theta, x_, y_)
                eps = (prob - y_)
                
                prod = eps[:, np.newaxis] * x_
                grad+=(1/m)*prod.sum(axis=0)
        grad=grad+self.lamb*theta
        return grad

    # def fit(self, theta, *args):
    #     opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=args, ftol=1e-6, disp=False)
    #     w = opt_weights[0]
    #     return w
    
    def divide_into_groups(self, K, N):
        # Generate the input array [0, 1, ..., N]
        input_array = np.arange(N)
        # Split the shuffled array into K groups
        groups = np.array_split(input_array, K)

        return groups



    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T,rev):
        print('UCB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.y_hist=[]
        self.x=x
        self.S=[]
        self.S_hist=[]
        self.r=0
        self.alpha=1
        self.kappa=0.1
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.eta=1/2*np.log(K+1)+3
        self.h=np.zeros((N,K))
        self.V=[None]*self.K
        self.V_til=[None]*self.K
        self.theta=[None]*self.K
        self.z=[]
        self.Ur=[]
        self.lamb=0
        self.rev=rev
        self.M=self.construct_M()
        self.number=0
    def run(self,t,index):   
    
        if t==1:
            U, Sigma, Vt = np.linalg.svd(self.x.T)
            self.r=np.count_nonzero(Sigma)
            self.Ur=U[:,:self.r]
            self.z=(self.Ur.T@self.x.T)
            self.z=self.z.T
            self.M=self.construct_M()
            self.S=random.choice(self.M)
            self.lamb=max(self.r*self.eta,self.eta)

            # for k in range(self.K):
            #     self.V.append(self.lamb*np.identity(self.r))
            for k in range(self.K):
                self.V[k]=self.lamb*np.identity(self.r)
                self.V_til[k]=self.lamb*np.identity(self.r)
                self.theta[k]=np.zeros(self.r)


        else:
            # theta_0=np.ones(self.r)
            y=self.match_elements(self.S, index) 
            # self.y_hist.append(y)
            # self.S_hist.append(self.S)
            # theta_prev=copy.deepcopy(self.theta)
            # self.theta=self.fit(theta_prev,self.z,y,self.S,self.V_til)
            theta_prev=copy.deepcopy(self.theta)

            for k in range(self.K):
                # y_k=[sublist[k] for sublist in self.y_hist]
                # S_k=[sublist[k] for sublist in self.S_hist]
                self.theta[k]=self.fit(theta_prev[k],self.z,y[k],self.S[k],self.V_til[k])
                self.V_til[k]=self.V[k]+self.eta*self.G(self.S[k],self.theta[k]) 
                self.V[k]+= self.G(self.S[k],self.theta[k]) 
                # for n in self.S[k]:
                #     self.V[k]+=np.outer(self.z[n],self.z[n])
            self.alpha=2*np.sqrt(self.r)*np.log(self.L)*(np.log(t)+np.sqrt(np.log(t)*np.log(self.K*self.T)))

            for k in range(self.K):
                for n in range(self.N):
                    self.h[n,k]=self.z[n]@self.theta[k]+self.alpha*np.sqrt(self.z[n]@np.linalg.inv(self.V[k])@self.z[n])
                    
            self.M=self.construct_M()
            R=0
            tmp_R=0
            self.number+=1
            for partition in self.M:
                for k in range(self.K):
                    if len(partition[k])>0:
                        # print('self.rev',self.rev)
                        # print('self.h',self.h)
                        tmp_R+=np.sum(self.rev[k][partition[k]]*np.exp(self.h[partition[k],k]))/(1+np.sum(np.exp(self.h[partition[k],k])))
                if R<tmp_R:
                    R=tmp_R
                    self.S=partition
                tmp_R=0
        
    def offer(self):
        return self.S   
    def M_len(self):
        return len(self.M)
    def number_result(self):
        return self.number
    def name(self):
        return 'UCB'    
        
########################


    
    
class Elimination:
    
    def compute_prob_loss(self, theta, x, y):
        means = np.dot(x, theta)
        u = np.exp(means.astype(float))
        SumExp = np.sum(u)+1
        if 1 in y:
            prob = u / SumExp
        else: 
            prob = 1 / SumExp
             
        return prob    
    def compute_prob_grad(self, theta, x):
        means = np.dot(x, theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        prob = u / SumExp
   

        return prob        
    def cost_function(self, theta, *args):
        x, y, S= args[0], args[1], args[2]
        loss=0
        for t in range(len(S)):
            if len(S[t])!=0:    
                x_=[x[i] for i in S[t]]
                y_=y[t]
                prob = self.compute_prob_loss(theta, x_, y_)
                loss+=-np.sum(np.multiply(y_, np.log(prob)))
        return loss + (1/2)*self.lamb* np.linalg.norm(theta)**2

    def gradient(self, theta, *args):
        x, y, S = args[0], args[1], args[2]
        m = 1
        grad=0
        for t in range(len(S)):
            if len(S[t])!=0:
                x_=x[S[t]]
                y_=y[t]
                prob = self.compute_prob_grad(theta, x_)
                eps = (prob - y_)
                prod = eps[:, np.newaxis] * x_
                grad+=(1/m)*prod.sum(axis=0)
        grad=grad+self.lamb*theta
        return grad

    # def fit(self, theta, *args):
    #     opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=args, ftol=1e-6, disp=False,approx_grad=True)
    #     w = opt_weights[0]
    #     return w
    
    
    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta[0:self.r])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.r] = theta[0:self.r] / norm_theta
            
        return theta
        
    def fit(self, theta_prev, *args):
        x, y, S = args[0], args[1], args[2]
        # print('S',S)
        # print('theta_prev',theta_prev)
        # print('x',x)
        self.g=np.zeros(self.r)
        x=np.array(x)
        for t in range(len(S)):
            if len(S[t])!=0:
                # print('S[t]',S[t])
                x_=x[S[t]]
                y_=y[t]
        
                means = np.dot(x_, theta_prev)
                u = np.exp(means)
                SumExp = np.sum(u)+1
                # print('S[t]',S[t])
                for i,n in enumerate(S[t]):
                    # print('theta_prev',theta_prev)
                    p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y_[i])*x[n]
                    self.g += p
        iter = 0
        theta=copy.deepcopy(theta_prev)
        while True:
            theta_=copy.deepcopy(theta)
            grad=self.g+1/2*np.linalg.norm(theta)
            theta = theta - 0.01*grad
            iter += 1
            if np.linalg.norm(theta-theta_) < 1e-5 or iter > 15:
                break

        return theta   
    
    
    
    
    def divide_into_groups(self, t,k):
        groups=[[] for _ in range(self.K)]
        l=min(self.L,len(self.A_original[k]))
        a=(l*(t-1))%len(self.A_original[k])
        b=(l*t)%len(self.A_original[k])
        if a<b:
            groups[k]=self.A_original[k][a:b]
        elif a==b:
            groups[k]=self.A_original[k]
        else:
            groups[k]=self.A_original[k][:b]+self.A_original[k][a:]

        # Split the shuffled array into K groups
        input_array=[]
        for k_ in range(self.K):
            input_array+=self.A_original[k]
        input_array = set(input_array)
        input_array=np.array(list(input_array.difference(set(groups[k]))))
        # print('self.M_original',self.M_original)
        M=[]
        for S in self.M_original:
            # print('S[k]',S[k])
            # print('groups[k]',groups[k])
            # print(set(S[k])==set(groups[k]))
            if set(S[k])==set(groups[k]):
                M.append(S)
        # print('M',M)
        if len(input_array)!=0:
            groups=random.choice(M)
   
        return groups

    def divide_into_groups_max(self,k):

        groups=[[] for _ in range(self.K)]
        groups[k]=self.A[k]
        input_array = np.arange(self.N)
        input_array=np.array(list(set(input_array).difference(set(self.A[k]))))

        R_sum=0
        tmp=0
        for S in self.M:
            if groups[k]==S[k]:
                tmp= self.upper_R(S)
                if tmp>R_sum:
                    R_sum=tmp
                    S_max=S
        return S_max
          
    def divide_into_groups_random(self):
        groups=[[] for _ in range(self.K)]
        # Split the shuffled array into K groups
        input_array = np.arange(self.N)
        A_=[[] for _ in range(self.N)]
        for k in range(self.K):
            for n in self.A_original[k]:
                A_[n].append(k)
        for n in input_array:
            groups[random.choice(A_[n])].append(n)


        return groups



    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in self.A[k]:
                A_[n].append(k)           
        A_=[a if a else [None] for a in A_]

        combinations=list(product(*A_))
        for combination in combinations:
            S=[[] for _ in range(self.K)]
            for n in range(len(combination)):
                if combination[n]!=None: 
                    S[combination[n]].append(n)
                    # and sum(len(sublist) for sublist in S)>=min(self.L*self.K,self.N)
            if all(len(sublist) <= self.L for sublist in S):
                M.append(S)
        return M              

            
    def upper_R(self,S):
        R_sum=0
        for k in range(self.K):
            if len(S[k])!=0:
                u = np.exp(self.p[k][S[k]])
                u_sum=np.sum(u)
                r_sum=np.sum(self.rev[k][S[k]]*u)
                SumExp =  u_sum+1
                prob_sum =  (r_sum / SumExp)  +self.beta*np.max([np.sqrt(self.z[n]@np.linalg.inv(self.V[k])@self.z[n]) for n in S[k]])
                R_sum+=prob_sum
        return R_sum
    def lower_R(self,S):
        R_sum=0
        for k in range(self.K):
            if len(S[k])!=0:
                u = np.exp(self.p[k][S[k]])
                u_sum=np.sum(u)
                r_sum=np.sum(self.rev[k][S[k]]*u)
                SumExp =  u_sum+1
                prob_sum =  (r_sum / SumExp)  -self.beta*np.max([np.sqrt(self.z[n]@np.linalg.inv(self.V[k])@self.z[n]) for n in S[k]])
                R_sum+=prob_sum
        return R_sum

    def S_argmax(self,n,k):
        R_sum=-1
        tmp=0

        S_max=[[] for _ in range(self.K)]
        for S in self.M:
            if n in S[k]:
                tmp= self.upper_R(S)
                if tmp>R_sum:
                    R_sum=tmp
                    S_max=S
        return S_max

        
    
    def max_lower(self):
        R_sum=0
        tmp=0
        for S in self.M:
            tmp= self.lower_R(S)
            if tmp>R_sum:
                R_sum=tmp
        return R_sum
    
    def elimination(self):
        A=[[] for _ in range(self.K)]

        max_lower=self.max_lower()
        for k in range(self.K):
            for n in self.A[k]:             
                # print('max_lower',max_lower)
                # print('self.upper_R',self.upper_R(self.S_max[k][n]))   
                if max_lower<=self.upper_R(self.S_max[k][n]):
                    A[k].append(n)
        return A  
    
    def objective_function(self,pi, z):
        cov_matrix=(1/(self.r*self.T_tau))*np.eye(self.r)
        cov_matrix += sum(pi[i] * np.outer(z[i], z[i]) for i in range(len(pi)))
        log_det_cov = np.log(np.linalg.det(cov_matrix))
        return -log_det_cov

    def constraint(self,pi):
        return 1.0 - np.sum(pi)

    def find_optimal_policy(self,k):
        num_agents = len(self.A[k])
        # Initial guess for pi
        if num_agents ==0:
            return []
        
        initial_pi = np.ones(num_agents) / num_agents
        if num_agents ==1:
            return initial_pi

        else:
            z=[self.z[i] for i in self.A[k]]
            # Constraints
            constraints = ({'type': 'eq', 'fun': self.constraint})

            # Optimization
            result = minimize(self.objective_function, initial_pi, args=(z,), constraints=constraints, bounds=[(0, 1) for _ in range(num_agents)])

            if result.success:
                optimal_pi = result.x
                return optimal_pi
            else:
                return initial_pi
                raise ValueError("Optimization failed.")
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T,rev):
        print('Elimination')
        np.random.seed(seed)
        random.seed(seed)
        self.rev=rev
        self.seed=seed
        self.y_hist=[]
        self.x=x
        self.S=[]
        self.S_hist=[]
        self.r=0
        self.kappa=1/L**2
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=0
        self.h=np.zeros((N,K))
        self.V=[0]*self.K
        self.z=[[] for _ in range(self.N)]
        self.Ur=[]
        self.A=[]
        self.T1=np.zeros(self.K)
        self.T2=[None]*self.K
        self.t1=np.zeros(self.K)
        self.t2=np.zeros((self.K,self.N))
        self.T_tau=1
        self.start_epoch=True
        self.bool1=False
        self.bool2=False
        self.bool_initial=True
        self.k=0
        self.n=0
        self.beta=(1/160)*(1/self.kappa)*np.sqrt(math.log(self.T*self.K*self.N))
        self.p=np.zeros((self.K,self.N))
        self.b=np.zeros((self.K,self.N))
        self.M=[]
        self.M_original=[]
        self.pi=[None]*self.K
        self.S_max= [[[] for _ in range(self.N)] for _ in range(self.K)]
        self.theta=[None]*self.K
        self.theta_0=[None]*self.K 
        self.epoch=0
        self.S_=[]
        self.initial=True
        self.bool_warmup=True
        self.bool_main=False
        self.main_start=0
        self.eta=0
        self.warmup_initial=True
        self.main_initial=False
        self.action=True
        self.lambda_min= 1
        self.number=0

        for k in range(self.K):
            self.A.append(list(range(self.N)))
        self.A_original=copy.deepcopy(self.A)

        self.M=self.construct_M()

    def run(self,t,index):   
        if self.initial==True:
            self.M=self.construct_M()
            for k in range(self.K):
                X=self.x[self.A[k],:]
                U, Sigma, Vt = np.linalg.svd(X.T)
                self.r=np.count_nonzero(Sigma)
                self.Ur=U[:,:int(self.r)]
                z=(X@self.Ur)
              
                (w,v) = np.linalg.eigh(z.T@z)
                self.lambda_min= np.amin(w)
                for i,n in enumerate(self.A[k]):
                    self.z[n]=z[i]  
                self.theta_0[k]=np.ones(int(self.r))
                self.theta[k]=np.zeros(int(self.r))
                self.V[k]=np.identity(self.r)
                self.M_original=self.construct_M()
            self.eta=math.sqrt(self.T/(self.r*self.K))
            self.initial=False 
            self.T_tau=self.eta  

                     
        if self.bool_warmup==True:
            if self.warmup_initial==True:
                
                for k in range(self.K):
                    self.V[k]=self.lamb*np.eye(self.r)
                self.y_hist=[]
                self.S_hist=[]
                
                self.epoch+=1
                print(self.epoch)
                for k in range(self.K):
                    self.T1[k]=(1/100)*int(self.N)/(self.L*self.kappa**2*self.lambda_min*math.log(self.T*self.K*self.N))*(self.r+math.log(self.T*self.K*self.N))**2
                self.t1[0]=t
                self.k=0
                self.warmup_initial=False

            if t!=1:
                y=self.match_elements(self.S, index) 
                self.y_hist.append(y)
                self.S_hist.append(self.S)   
                # print('self.V',self.V[1])
                # print('self.z',self.z[0])     
                for k in range(self.K):
                    for n in self.S[k]:
                        # print('self.V',self.V)
                        # print('self.z',self.z)
                        self.V[k]+=np.outer(self.z[n],self.z[n])

                    
                    # g=np.array([self.z[i] for i in self.S[k]])
                    # self.V[k]+=g.T@g
                # print('S',self.S)
                # print('t',t)
                # print('self.V',self.V)

            if t <= self.t1[self.k]+self.T1[self.k]-1:

                self.S = self.divide_into_groups_random()
            elif self.k!=self.K-1:
                self.k+=1
                self.t1[self.k]=t
                self.S = self.divide_into_groups_random()

            else:
                self.bool_warmup=False
                self.bool_main=True
                self.main_initial=True
                self.main_start=t
                # self.initial=True

        if self.bool_main==True:
            if self.main_initial==True:
                print('self.A',self.A)    
                for k in range(self.K):
                    y_k=[sublist[k] for sublist in self.y_hist]
                    S_k=[sublist[k] for sublist in self.S_hist]
                    self.theta[k]=self.fit(self.theta_0[k],self.z,y_k,S_k)
                    # print('self.theta',self.theta)
                    for n in self.A[k]:
                        self.p[k][n]=self.z[n]@self.theta[k]                           
                for k in range(self.K):
                    for n in self.A[k]:
                        self.S_max[k][n]=self.S_argmax(n,k)
                self.A_prev=copy.deepcopy(self.A)
                self.A=self.elimination()
                if all(not sublist for sublist in self.A):
                    self.A=self.A_prev
                if self.epoch>1:
                    self.T_tau=self.eta*np.sqrt(self.T_tau)  
                else:
                    self.T_tau=self.eta
                self.M=self.construct_M()
                self.number+=1
                for k in range(self.K):
                    self.pi[k] = self.find_optimal_policy(k)
                    if len(self.pi[k])==0:
                        self.T2[k]=0
                    else:
                        self.T2[k]=[math.ceil(x) for x in self.pi[k]*self.r*self.T_tau]

                    self.k=0
                    while len(self.A[self.k])==0 and self.k!=self.K-1:
                        self.k+=1

                    self.n=self.A[self.k][0]
                    self.t2[self.k][self.n]=t        

                    for k in range(self.K):
                        for n in self.A[k]:
                            self.S_max[k][n]=self.S_argmax(n,k) 

                self.main_initial=False


            if t>1:    
                y=self.match_elements(self.S, index) 
                self.y_hist.append(y)
                self.S_hist.append(self.S)
                for k in range(self.K):
                    for n in self.S[k]:
                        self.V[k]+=np.outer(self.z[n],self.z[n])
                    
            self.action=True
            while self.action==True:
                ind_n=self.A[self.k].index(self.n)
                if t<= self.t2[self.k][self.n]+self.T2[self.k][ind_n]-1:
                    self.S=self.S_max[self.k][self.n]  
                    
                    self.action=False
     
                elif self.n!=self.A[self.k][-1]:
                    self.n=self.A[self.k][ind_n+1]
                    self.t2[self.k][self.n]=t
                elif self.k!=self.K-1:
                    self.k+=1
                    while len(self.A[self.k])==0 and self.k!=self.K-1:
                        self.k+=1
                    if len(self.A[self.k])!=0:
                        self.n=self.A[self.k][0]
                        self.t2[self.k][self.n]=t
                    if len(self.A[self.k])==0 and self.k==self.K-1:
                        self.bool_warmup=True
                        self.warmup_initial=True
                        self.bool_main=False
                        self.main_initial=False
                        self.action=False

                else:
                    self.bool_warmup=True
                    self.warmup_initial=True
                    self.bool_main=False
                    self.main_initial=False
                    self.action=False
 
    def offer(self):
        return self.S   
    def name(self):
        return 'Elimination'    
    def number_result(self):
        return self.number
    def M_len(self):
        return len(self.M)
   
##################################   
   
class Elimination2:
    
    
    
    def segment_with_constraint(self, k, S):
        """
        Generate all possible segmentations of [N] into K lists such that the k-th list is S.

        Args:
            N (int): The size of the set [N].
            K (int): The number of lists to partition into.
            k (int): The index of the list (1-indexed) that must contain S.
            S (set): The specific subset that must be in the k-th list.

        Returns:
            list: A list of segmentations, where each segmentation is a list of K lists.
        """

        # Ensure S is a subset of [N]
        
        
        # full_set = set(self.A[k])
        
        combined_set = set()
        for array in self.A:
            combined_set.update(array)
        full_set=set(combined_set)
        S_set = set(S)
        # print('S_set',S_set)
        # print('full_set',full_set)
        if not S_set.issubset(full_set):
            raise ValueError("S must be a subset of [N].")

        # Remaining elements after assigning S to the k-th list
        remaining_elements = full_set - S_set

        def helper(elements, num_parts):
            """
            Recursively partition the remaining elements into num_parts lists.
            """
            if num_parts == 1:
                return [[list(elements)]]

            partitions = []
            elements = list(elements)
            for i in range(len(elements) + 1):
                for combination in combinations(elements, i):
                  if len(combination) <= self.L:  
                    # print('combination',combination)
                    remaining = set(elements) - set(combination)
                    for sub_partition in helper(remaining, num_parts - 1):
                        # if len(sub_partition[0]) <=self.L:
                    # for sub_partition in helper(remaining, num_parts - 1):
                        partitions.append([list(combination)] + sub_partition)
            return partitions

        # Partition the remaining elements into K - 1 lists
        all_partitions = helper(remaining_elements, self.K - 1)
        # print("all_partitions",all_partitions)
        # Insert S into the k-th position of each partition
        final_partitions = []
        for partition in all_partitions:
            partition.insert(k, list(S_set))
            final_partitions.append(partition)

        return final_partitions
        
    
    def compute_prob_loss(self, theta, x, y):
        means = np.dot(x, theta)
        u = np.exp(means.astype(float))
        SumExp = np.sum(u)+1
        if 1 in y:
            prob = u / SumExp
        else: 
            prob = 1 / SumExp
             
        return prob    
    
    def compute_prob_grad(self, theta, x):
        means = np.dot(x, theta)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        prob = u / SumExp

        return prob        
    
    def cost_function(self, theta, *args):
        x, y, S= args[0], args[1], args[2]
        loss=0
        for t in range(len(S)):
            if len(S[t])!=0:    
                x_=[x[i] for i in S[t]]
                y_=y[t]
                prob = self.compute_prob_loss(theta, x_, y_)
                loss+=-np.sum(np.multiply(y_, np.log(prob)))
        return loss + (1/2)*self.lamb* np.linalg.norm(theta)**2

    def gradient(self, theta, *args):
        x, y, S = args[0], args[1], args[2]
        m = 1
        grad=0
        for t in range(len(S)):
            if len(S[t])!=0:
                x_=x[S[t]]
                y_=y[t]
                prob = self.compute_prob_grad(theta, x_)
                eps = (prob - y_)
                prod = eps[:, np.newaxis] * x_
                grad+=(1/m)*prod.sum(axis=0)
        grad=grad+self.lamb*theta
        return grad

    # def fit(self, theta, *args):
    #     opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=args, ftol=1e-3, disp=False,approx_grad=True)
    #     w = opt_weights[0]
    #     return w
    
    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta[0:self.r])  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.r] = theta[0:self.r] / norm_theta
            
        return theta
    
    def fit(self, theta_prev, *args):
        x, y, S = args[0], args[1], args[2]
        # print('S',S)
        # print('theta_prev',theta_prev)
        # print('x',x)
        self.g=np.zeros(self.r)

        for t in range(len(S)):
            if len(S[t])!=0:
                x_=x[S[t]]
                y_=y[t]
        
                means = np.dot(x_, theta_prev)
                u = np.exp(means)
                SumExp = np.sum(u)+1
                # print('S[t]',S[t])
                for i,n in enumerate(S[t]):
                    # print('theta_prev',theta_prev)
                    p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y_[i])*x[n]
                    self.g += p
        iter = 0
        theta=copy.deepcopy(theta_prev)
        while True:
            theta_=copy.deepcopy(theta)
            grad=self.g+1/2*np.linalg.norm(theta)
            theta = theta - 0.01*grad
            iter += 1
            if np.linalg.norm(theta-theta_) < 1e-5 or iter > 15:
                break
        theta=self.project_to_unit_ball(theta)
        # else:
        #     theta=theta_prev

        return theta       
    def divide_into_groups(self, t,k):
        groups=[[] for _ in range(self.K)]
        l=min(self.L,len(self.A[k]))
        a=(l*(t-1))%len(self.A[k])
        b=(l*t)%len(self.A[k])
        if a<b:
            groups[k]=self.A[k][a:b]
        elif a==b:
            groups[k]=self.A[k]
        else:
            groups[k]=self.A[k][:b]+self.A[k][a:]

        # Split the shuffled array into K groups
        input_array=[]
        for k_ in range(self.K):
            input_array+=self.A[k]
        input_array = set(input_array)
        input_array=np.array(list(input_array.difference(set(groups[k]))))
                
        M=[]
        for S in self.M:
            if set(S[k])==set(groups[k]):
                M.append(S)
        if len(input_array)!=0:
            groups=random.choice(M)
   
        return groups

    def divide_into_groups_max(self,k):

        groups=[[] for _ in range(self.K)]
        groups[k]=self.A[k]
        input_array = np.arange(self.N)
        input_array=np.array(list(set(input_array).difference(set(self.A[k]))))

        R_sum=0
        tmp=0
        for S in self.M:
            if groups[k]==S[k]:
                tmp= self.upper_R(S)
                if tmp>R_sum:
                    R_sum=tmp
                    S_max=S
        return S_max
          
    def divide_into_groups_random(self):
        groups=[[] for _ in range(self.K)]
        # Split the shuffled array into K groups
        input_array = np.arange(self.N)
        A_=[[] for _ in range(self.N)]
        for k in range(self.K):
            # for n in self.A[k]:
            for n in range(self.N):
                A_[n].append(k)
        for n in input_array:
            groups[random.choice(A_[n])].append(n)


        return groups
    
    
    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
    
    def construct_J(self,S):
        """
        Generate all possible combination sets from a set S with limited cardinality L.

        Parameters:
            S (set): The input set of elements.
            L (int): The maximum cardinality for each combination.

        Returns:
            list: A list of all possible combinations as sets.
        """
        # if not isinstance(S, set):
        #     print(S)
        #     raise ValueError("Input S must be a set.")
        # if not isinstance(self.L, int) or self.L < 1:
        #     raise ValueError("Cardinality L must be a positive integer.")

        all_combinations = []
        for r in range(1, self.L + 1):  # r is the size of the combination
            all_combinations.extend(map(list, combinations(S, r)))
            
        return all_combinations        
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            if len(self.A[k])!=0:
                for n in self.A[k]:
                    A_[n].append(k)           
        # print('A_',A_)
        A_=[a if a else [None] for a in A_]

        combinations=list(product(*A_))
        for combination in combinations:
            S=[[] for _ in range(self.K)]
            for n in range(len(combination)):
                if combination[n]!=None: 
                    S[combination[n]].append(n)
            if all(len(sublist) <= self.L for sublist in S):
                M.append(S)
        return M              

            
    def upper_R(self,S):
        R_sum=0
        for k in range(self.K):
            if len(S[k]) > 0: 
                u = np.exp(self.p[k][S[k]])
                u_sum=np.sum(u)
                r_sum=np.sum(self.rev[k][S[k]]*u)
                SumExp =  u_sum+1
                prob_sum =  (r_sum / SumExp) 
                ##check from here
                
                p=self.compute_prob_grad(self.theta[k], self.z[S[k]])
                for n in S[k]:
                    self.z_tilde[n]=self.z[n]-sum(p[i]*self.z[n] for i in range(len(p)))
                
                prob_sum+=self.beta**2*np.max([(self.z[n]@np.linalg.inv(self.H[k])@self.z[n]) for n in S[k]])
                prob_sum+=self.beta**2*np.max([((self.z_tilde[n]@np.linalg.inv(self.H[k])@self.z_tilde[n])) for n in S[k]]) 
                u = np.exp(self.p_prev[k][S[k]])
                u_sum=np.sum(u)
                prob_sum+=self.beta*np.sum((u/u_sum)*[(np.sqrt(self.z_tilde[n]@np.linalg.inv(self.H[k])@self.z_tilde[n])) for n in S[k]])
                R_sum+=prob_sum
        return R_sum
    def lower_R(self,S):
        R_sum=0
        for k in range(self.K):
            if len(S[k]) > 0: 
                
                u = np.exp(self.p[k][S[k]])
                u_sum=np.sum(u)
                r_sum=np.sum(self.rev[k][S[k]]*u)
                SumExp =  u_sum+1
                prob_sum =  (r_sum / SumExp) 
                p=self.compute_prob_grad(self.theta[k], self.z[S[k]])
                for n in S[k]:
                    self.z_tilde[n]=self.z[n]-sum(p[i]*self.z[n] for i in range(len(p)))
                
                prob_sum-=self.beta**2*np.max([(self.z[n]@np.linalg.inv(self.H[k])@self.z[n]) for n in S[k]])
                prob_sum-=self.beta**2*np.max([((self.z_tilde[n]@np.linalg.inv(self.H[k])@self.z_tilde[n])) for n in S[k]]) 
                u = np.exp(self.p_prev[k][S[k]])
                u_sum=np.sum(u)
                prob_sum-=self.beta*np.sum((u/u_sum)*[(np.sqrt(self.z_tilde[n]@np.linalg.inv(self.H[k])@self.z_tilde[n])) for n in S[k]])
                R_sum+=prob_sum

        return R_sum

    def S_argmax(self,n,k):
        R_sum=0
        tmp=0
        # print('A',self.A)
        # print('M',self.M)
        # print('n',n)
        # print('k',k)
        # S_max = None
        # S_max=self.divide_into_groups_random()
        for S in self.M:
            # print(n in S[k])
            if n in S[k]: ##change it more efficient
                tmp= self.upper_R(S)
                # print('tmp',tmp)
                # print('R_sum',R_sum)
                if tmp>R_sum:
                    # print('in')
                    R_sum=tmp
                    S_max=S
                    # print('S_max',S_max)
        return S_max

    def S_argmax2(self,S,k):
        R_sum=-1
        tmp=0
        # print('S',S)
        # print('k',k)
        # print(self.segment_with_constraint(k, S))
        for S_ in self.segment_with_constraint(k, S):
            # print('S_',S_)
            tmp= self.upper_R(S_)
            if tmp>R_sum:
                R_sum=tmp
                S_max=S_
        return S_max
    # def S_argmax3(self,S,k):
    #     R_sum=-1
    #     tmp=0
    #     # print('S',S)
    #     # print('k',k)
    #     # print(self.segment_with_constraint(k, S))
    #     for S_ in self.segment_with_constraint(k, S):
    #         # print('S_',S_)
    #         tmp= self.upper_R(S_)
    #         if tmp>R_sum:
    #             R_sum=tmp
    #             S_max=S_
    #     return S_max
    
    def max_lower(self):
        R_sum=0
        tmp=0
        for S in self.M:
            tmp= self.lower_R(S)
            if tmp>R_sum:
                R_sum=tmp
        return R_sum
    
    def elimination(self):
        A=[[] for _ in range(self.K)]

        max_lower=self.max_lower()
        for k in range(self.K):
            for n in self.A[k]:                
                if max_lower<=self.upper_R(self.S_max[k][n]):
                    # print('max_lower',max_lower)
                    # print('self.upper_R',self.upper_R(self.S_max[k][n]))
                    A[k].append(n)
                    
        for k in range(self.K):
            for j,S in enumerate(self.J[k]):
                if len(S)!=0:
                    if max_lower<=self.upper_R(self.S_max2[k][j]):
                        for n in S:
                            if n not in A[k]:
                                A[k].append(n)
        # print('A',A)
        return A  
    
    def objective_function(self,pi, z):
        # print('r',self.r)
        cov_matrix=(self.lamb/(self.r*self.T_tau))*np.eye(self.r)
        cov_matrix += sum(pi[i] * np.outer(z[i], z[i]) for i in range(len(pi)))
        log_det_cov = np.log(np.linalg.det(cov_matrix))
        return -log_det_cov

    
    def constraint(self,pi):
        return 1.0 - np.sum(pi)

    def find_optimal_policy(self,k):
        num_agents = len(self.A[k])
        # Initial guess for pi
        if num_agents ==0:
            return []
        
        initial_pi = np.ones(num_agents) / num_agents
        if num_agents ==1:
            return initial_pi

        else:
            z=[self.z[i] for i in self.A[k]]
            # p=self.compute_prob_grad(self.theta[k], self.z[self.S[k]])
            # Constraints
            constraints = ({'type': 'eq', 'fun': self.constraint})

            # Optimization
            result1 = minimize(self.objective_function, initial_pi, args=(z,), constraints=constraints, bounds=[(0, 1) for _ in range(num_agents)])

            if result1.success:
                optimal_pi = result1.x
                return np.array(optimal_pi)
            else:
                return np.array(initial_pi)
                raise ValueError("Optimization failed.")
            
    def objective_function2(self,pi,k):
        # print('z',self.z)
        # print('k',k)
        # print('S',self.S)
        cov_matrix=(self.lamb/(self.T_tau*self.r))*np.eye(self.r)
        for j,S in enumerate(self.J[k]):
            if len(S)!=0:
                p=self.compute_prob_grad(self.theta[k], self.z[S])
                for n in S:
                    self.z_tilde[n]=self.z[n]-sum(p[i]*self.z[n] for i in range(len(p)))
                # self.z_tilde[n]
                cov_matrix+=pi[j]*sum(p[i]* np.outer(self.z_tilde[n], self.z_tilde[n]) for i,n in enumerate(S))
        log_det_cov = np.log(np.linalg.det(cov_matrix))
        return -log_det_cov       
            
    def find_optimal_policy2(self,k):
        num_agents = len(self.J[k])
        # Initial guess for pi
        if num_agents ==0:
            return []
        
        initial_pi = np.ones(num_agents) / num_agents
        if num_agents ==1:
            return initial_pi

        else:
            # Constraints
            constraints = ({'type': 'eq', 'fun': self.constraint})

            # Optimization
            result1 = minimize(self.objective_function2, initial_pi, args=(k,), constraints=constraints, bounds=[(0, 1) for _ in range(num_agents)])

            if result1.success:
                optimal_pi = result1.x
                return optimal_pi
            else:
                return initial_pi
            
            
            
    def objective_function3(self,pi,k):
        # print('z',self.z)
        # print('k',k)
        # print('S',self.S)
        cov_matrix=(self.lamb/(self.T_tau*self.r))*np.eye(self.r)
        self.z_tilde=np.zeros_like(self.z)
        for j,S in enumerate(self.J[k]):
            if len(S)!=0:
                p=self.compute_prob_grad(self.theta[k], self.z[S])
                for l,n in enumerate(S):
                    self.z_tilde[n]=self.z[n]-sum(p[i]*self.z[n] for i in range(len(p)))
                # self.z_tilde[n]
                    cov_matrix+=pi[j*self.L+l]*np.outer(self.z_tilde[n], self.z_tilde[n]) 
        log_det_cov = np.log(np.linalg.det(cov_matrix))
        return -log_det_cov            
            
    def find_optimal_policy3(self,k):
        num_agents = len(self.J[k])*self.L
        # Initial guess for pi
        if num_agents ==0:
            return []
        
        initial_pi = np.ones(num_agents) / num_agents
        if num_agents ==1:
            return initial_pi

        else:
            # Constraints
            constraints = ({'type': 'eq', 'fun': self.constraint})

            # Optimization
            result1 = minimize(self.objective_function3, initial_pi, args=(k,), constraints=constraints, bounds=[(0, 1) for _ in range(num_agents)])

            if result1.success:
                optimal_pi = result1.x
                return optimal_pi
            else:
                return initial_pi            
            
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T,rev):
        print('Elimination2')
        np.random.seed(seed)
        random.seed(seed)
        self.rev=rev
        self.seed=seed
        self.y_hist=[]
        self.x=x
        self.S=[]
        self.S_hist=[]
        self.r=0
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=1
        self.h=np.zeros((N,K))
        self.H=[0]*self.K
        self.H_prev=[0]*self.K
        # self.H=[0]*self.K
        # self.T=1
        self.z=[[] for _ in range(self.N)]
        self.z_tilde=[[] for _ in range(self.N)]
        self.Ur=[]
        self.A=[]
        self.T1=[None]*self.K
        self.T2=[None]*self.K
        self.T3=[None]*self.K

        self.t1=np.zeros((K,N))
        self.t2=[None]*self.K
        self.t3=[None]*self.K

        self.T_tau=(1/30)*math.log(self.T*self.K*self.L)**2*math.log(self.T)
        self.start_epoch=True
        self.bool1=False
        self.bool2=False
        self.bool_initial=True
        self.k=0
        self.n=0
        self.p=np.zeros((self.K,self.N))
        self.p_prev=np.zeros((self.K,self.N))
        self.b=np.zeros((self.K,self.N))
        self.M=[]
        self.pi=[None]*self.K
        self.pi_J=[None]*self.K
        self.pi_J2=[None]*self.K
        self.S_max= [[[] for _ in range(self.N)] for _ in range(self.K)]
        self.S_max2=[None]*self.K
        self.theta=[None]*self.K
        self.theta_0=[None]*self.K 
        self.epoch=0
        self.S_=[]
        self.initial=True
        self.bool_play1=False
        self.bool_play2=False
        self.bool_warmup=True
        self.bool_main=False
        self.main_start=1
        self.initial2=True
        self.initial3=True
        self.ind_l=0
        self.eta=0
        self.J=[None]*self.K
        self.bool_exploration2=False
        self.bool_exploration3=False
        self.initial1=True
        self.action=True
        self.epoch_start=True
        for k in range(self.K):
            self.A.append(list(range(self.N)))
        self.M=self.construct_M()
        self.number=0
    def run(self,t,index):   
        # print('t',t)
        # print('self.S',self.S)
        if self.initial==True:
            self.M=self.construct_M()
            # print('length', self.M_len())
            for k in range(self.K):
                X=self.x[self.A[k],:]
                U, Sigma, Vt = np.linalg.svd(X.T)
                self.r=np.count_nonzero(Sigma)
                self.Ur=U[:,:int(self.r)]
                z=(X@self.Ur)
                (w,v) = np.linalg.eigh(z.T@z)
                lambda_min= np.amin(w)
                for i,n in enumerate(self.A[k]):
                    self.z[n]=z[i]  
                self.H[k]=self.lamb*np.eye(self.r)
                self.H_prev[k]=self.lamb*np.eye(self.r)

                self.theta_0[k]=np.ones(int(self.r))
                self.theta[k]=np.zeros(int(self.r))
            self.t1[0]=1
            self.k=0               
            self.z=np.array(self.z)
            self.z_tilde=np.zeros_like(self.z)

            self.eta=math.sqrt(self.T/(self.r*self.K))
            self.lamb=self.r*np.log(self.K)
            self.beta=(1/300)*np.sqrt(self.r)*np.sqrt(math.log(self.T*self.K*self.L))
            self.initial=False
            # self.S=self.divide_into_groups_random()
            for k in range(self.K):
                self.H[k]=self.lamb*np.eye(self.r)
        if t>1:     
            y=self.match_elements(self.S, index) 
            self.y_hist.append(y)
            self.S_hist.append(self.S)
            for k in range(self.K):
                # g=np.array([self.z[i] for i in self.S[k]])
                self.H_prev[k]=copy.deepcopy(self.H[k])
                p=self.compute_prob_grad(self.theta[k], self.z[self.S[k]])
                # self.V[k]+=g.T@g
                for i,n in enumerate(self.S[k]):
                    self.H[k]+=p[i]*np.outer(self.z[n],self.z[n])
                for i,n in enumerate(self.S[k]):
                    for j,m in enumerate(self.S[k]):
                        self.H[k]-=p[i]*p[j]*np.outer(self.z[n],self.z[m])
                        
                        
                        
        if self.epoch_start==True:
            self.epoch+=1         
            print('self.A',self.A)
            print('epoch',self.epoch)                              
            for k in range(self.K):
                y_k=[sublist[k] for sublist in self.y_hist]
                S_k=[sublist[k] for sublist in self.S_hist]
                # print('theta',self.theta_0[k])
                self.theta[k]=self.fit(self.theta_0[k],self.z,y_k,S_k)
                # print(self.theta[k])
                            
            for k in range(self.K):
                self.p_prev[k]=copy.deepcopy(self.p[k])
                # deepcopy(self.p[k])
                for n in self.A[k]:
                    self.p[k][n]=self.z[n]@self.theta[k]
            for k in range(self.K):
                self.J[k]=self.construct_J(self.A[k])
                # print('self.J[k]',self.J[k])
                self.t2[k]=np.zeros(len(self.J[k]))
                self.t3[k]=np.zeros(len(self.J[k]*self.L))

                self.S_max2[k]=[[] for _ in range(len(self.J[k]))]
                for j,S in enumerate(self.J[k]):
                    self.S_max2[k][j]=self.S_argmax2(S,k)

                    # print('S',S)   
                    # print('k',k)
                    # print('self.S_max[k][n]',self.S_max2[k][j])
 
            for k in range(self.K):
                for n in self.A[k]:
                    # print('k',k)
                    # print('self.A[k]',self.A[k])
                    self.S_max[k][n]=self.S_argmax(n,k)
            self.A_prev=copy.deepcopy(self.A)
            self.A=self.elimination()
            # print('self.A',self.A)
            if all(not sublist for sublist in self.A):
                self.A=self.A_prev
            
            if self.epoch>1:
                self.T_tau=self.eta*np.sqrt(self.T_tau)
            self.M=self.construct_M()
            self.number+=1

            # print(self.M)
            for k in range(self.K):
                
                self.pi[k] = self.find_optimal_policy(k)
                # self.pi[k]=np.array(self.pi[k])
                if len(self.pi[k])==0:
                    self.T1[k]=0
                else:
                    self.T1[k]=[math.ceil(x) for x in self.pi[k]*self.r*self.T_tau]
            for k in range(self.K):
                self.J[k]=self.construct_J(self.A[k])
                self.pi_J[k] = self.find_optimal_policy2(k)
                # self.pi_J2[k] = self.find_optimal_policy3(k)

                if len(self.pi_J[k])==0:
                    self.T2[k]=0
                else:
                    self.T2[k]=[math.ceil(x) for x in (1/2)*self.pi_J[k]*self.r*self.T_tau]
            for k in range(self.K):
                # self.J[k]=self.construct_J(self.A[k])
                # self.pi_J[k] = self.find_optimal_policy2(k)
                self.pi_J2[k] = self.find_optimal_policy3(k)

                if len(self.pi_J2[k])==0:
                    self.T3[k]=0
                else:
                    self.T3[k]=[math.ceil(x) for x in (1/2)*self.pi_J2[k]*self.r*self.T_tau]
            self.k=0
            while len(self.A[self.k])==0 and self.k!=self.K-1:
                self.k+=1

            self.n=self.A[self.k][0]
            self.t1[self.k][self.n]=t
            # restart
            for k in range(self.K):
                self.H[k]=self.lamb*np.eye(self.r)
                self.H_prev[k]=self.lamb*np.eye(self.r)
            self.y_hist=[]
            self.S_hist=[]
            
            self.epoch_start=False          
            self.bool_exploration=True
            self.initial1=True
        self.action=True
        if self.bool_exploration==True:
            if self.initial1==True:
                self.k=0
                while len(self.A[self.k])==0 and self.k!=self.K-1:
                    self.k+=1

                self.n=self.A[self.k][0]
                self.t1[self.k][self.n]=t
                self.initial1=False
            while self.action==True:
                # print('self.k',self.k)
                # print('t'   ,t)
                # print(self.n)
                # print(self.A[self.k])
                ind_n=self.A[self.k].index(self.n)
                # print('self.t1',self.T1[self.k][ind_n])
                if t<= self.t1[self.k][self.n]+self.T1[self.k][ind_n]:
                    self.S=self.S_max[self.k][self.n]       
                    # print('self.S',self.S)
                    # print('t',t)

                    self.action=False
                elif self.n!=self.A[self.k][-1]:

                    self.n=self.A[self.k][ind_n+1]
                    self.t1[self.k][self.n]=t
                elif self.k!=self.K-1:

                    self.k+=1
                    while len(self.A[self.k])==0 and self.k!=self.K-1:
                        self.k+=1

                    if len(self.A[self.k])!=0:
                        self.n=self.A[self.k][0]
                        self.t1[self.k][self.n]=t
                    if len(self.A[self.k])==0 and self.k==self.K-1:
                        self.bool_exploration=False
                        self.bool_exploration2=True
                        self.initial2=True
                        self.ind_S=0
                        self.action=False
                else:
                    self.bool_exploration=False
                    self.bool_exploration2=True
                    self.initial2=True
                    self.ind_S=0
                    self.action=False
                
        if self.bool_exploration2==True:

            if self.initial2==True:
                self.k=0
                while len(self.J[self.k])==0 and self.k!=self.K-1:
                    self.k+=1

                self.S_J=self.J[self.k][0]
                self.t2[self.k][0]=t
                self.initial2=False
            self.action=True
            # ind_S=self.J[self.k].index(self.S_J)
            while self.action==True:
                if t<= self.t2[self.k][self.ind_S]+self.T2[self.k][self.ind_S]-1:
                    self.S=self.S_max2[self.k][self.ind_S]
                    self.action=False       
                    # print('t',t)
                    # print('self.S',self.S)
                elif self.ind_S!=len(self.J[self.k])-1:
                    self.ind_S+=1
                    self.t2[self.k][self.ind_S]=t
                elif self.k!=self.K-1:
                    self.k+=1

                    while len(self.J[self.k])==0 and self.k!=self.K-1:
                        self.k+=1
                    if len(self.J[self.k])!=0:
                        self.ind_S=0
                        # self.n=self.J[self.k][0]
                        self.t2[self.k][self.ind_S]=t
                    if len(self.J[self.k])==0 and self.k==self.K-1:
                        self.bool_exploration2=False
                        self.bool_exploration3=True

                        self.action=False
                        self.initial3=True
                        self.ind_S=0
                        self.ind_l=0
                else:
                    self.bool_exploration2=False
                    self.bool_exploration3=True
                    self.action=False
                    self.initial3=True
                    self.ind_S=0
                    self.ind_l=0

                    
        if self.bool_exploration3==True:

            if self.initial3==True:
                self.k=0
                while len(self.J[self.k])==0 and self.k!=self.K-1:
                    self.k+=1

                self.S_J=self.J[self.k][0]
                self.t3[self.k][0]=t
                self.initial3=False
            self.action=True
            # ind_S=self.J[self.k].index(self.S_J)
            while self.action==True:
                if t<= self.t3[self.k][self.ind_S*self.L+self.ind_l]+self.T3[self.k][self.ind_S*self.L+self.ind_l]-1:
                    self.S=self.S_max2[self.k][self.ind_S]
                    self.action=False       
                    # print('t',t)
                    # print('self.S',self.S)
                elif self.ind_l!=self.L-1:
                    self.ind_l+=1
                    self.t3[self.k][self.ind_S*self.L+self.ind_l]=t
                elif self.ind_S!=len(self.J[self.k])-1:
                    self.ind_S+=1
                    self.ind_l=0
                    self.t3[self.k][self.ind_S*self.L+self.ind_l]=t
                elif self.k!=self.K-1:
                    self.k+=1
                    while len(self.J[self.k])==0 and self.k!=self.K-1:
                        self.k+=1
                    if len(self.J[self.k])!=0:
                        self.ind_S=0
                        self.ind_l=0

                        # self.n=self.J[self.k][0]
                        self.t3[self.k][self.ind_S*self.L+self.ind_l]=t
                    if len(self.J[self.k])==0 and self.k==self.K-1:
                        self.bool_exploration3=False
                        self.epoch_start=True
                        self.action=False

                else:
                    self.bool_exploration3=False
                    self.epoch_start=True
                    self.action=False

    def offer(self):
        return self.S   
    def name(self):
        return 'Elimination2'    

    def M_len(self):
        return len(self.M)
    def number_result(self):
        return self.number
 ####################






#######################################33
class UCB_QMB:

    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
                A_[n].append(k)           
                
        combinations=list(product(*A_))
        for combination in combinations:
            S=[[] for _ in range(self.K)]
            for n in range(len(combination)):
                if combination[n]!=None: 
                    S[combination[n]].append(n)
            if all(len(sublist) <= self.L for sublist in S):
                M.append(S)
        return M              

    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta)  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.d] = theta[0:self.d] / norm_theta
            

        return theta
    
    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
    #         obj=self.g@(theta-theta_prev)+(1/2)*(theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev)    
        if len(S)>0:
            iter = 0
            theta=copy.deepcopy(theta_prev)
            while True:
                theta_=copy.deepcopy(theta)
                grad=self.g+V@(theta-theta_prev)
                theta = theta - 0.01*grad
                iter += 1
                if np.linalg.norm(theta-theta_) < 1e-5 or iter > 20:
                    break
            theta=self.project_to_unit_ball(theta)
        else:
            theta=theta_prev

        return theta   
  

    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
    
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
            
    def __init__(self,seed,x,N,K,L,T):
        print('UCB_QMB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[]
        self.d=len(self.x[0])
        self.alpha=1
        self.kappa=1/L**2
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=1
        self.h=np.zeros((N,K))
        self.V=[]
        self.z=[]
        self.Ur=[]
        self.Q=np.ones(N)
        self.theta=np.zeros((self.K,self.d))
        self.number=0
        self.M=self.construct_M()

    def run(self,t,index):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            for k in range(self.K):
                self.V.append(self.lamb*np.identity(self.d))
        else:
            y=self.match_elements(self.S, index) 

            for k in range(self.K):
                for n in self.S[k]:
                    self.V[k]+=(self.kappa/2)*np.outer(self.x[n],self.x[n])
                theta_prev=self.theta[k].copy()
                y_k=y[k]
                S_k=self.S[k]
                self.theta[k]=self.fit(theta_prev,self.x,y_k,S_k,self.V[k])

            self.alpha=np.sqrt(1+(self.d/self.kappa)*np.log(1+(t*self.L)/(self.d*self.lamb)))

            for k in range(self.K):
                for n in range(self.N):
                    self.h[n,k]=self.x[n]@self.theta[k]+self.alpha*np.sqrt(self.x[n]@np.linalg.inv(self.V[k])@self.x[n])
            # self.Q=Q
            R=0
            tmp_R=0
            self.number+=1

            for partition in self.M:
                for k in range(self.K):
                    if len(partition[k])>0:                    
                        tmp_R+=np.sum(np.exp(self.h[partition[k],k])@self.Q[partition[k]])/(1+np.sum(np.exp(self.h[partition[k],k])))
                if R<tmp_R:
                    R=tmp_R
                    self.S=copy.deepcopy(partition)
                tmp_R=0
        self.remove_elements(self.S, self.Q)

    def offer(self):
        return self.S   
    
    def name(self):
        return 'UCB-QMB'    
    def M_len(self):
        return len(self.M)
    def number_result(self):
        return self.number

class TS_QMB:
    
    def construct_M(self):
        A_=[[] for _ in range(self.N)]
        M=[]
        for n in range(self.N):
            A_[n].append(None)
        for k in range(self.K):
            for n in range(self.N):
                A_[n].append(k)           
                
        combinations=list(product(*A_))
        for combination in combinations:
            S=[[] for _ in range(self.K)]
            for n in range(len(combination)):
                if combination[n]!=None: 
                    S[combination[n]].append(n)
        #  and sum(len(sublist) for sublist in S)>=min(self.L*self.K,self.N)
            if all(len(sublist) <= self.L for sublist in S):
                M.append(S)
        return M              

    def project_to_unit_ball(self,theta):
        norm_theta = np.linalg.norm(theta)  # Compute the norm of theta

        if norm_theta > 1:
            # If the norm is greater than 1, scale theta to have norm 1
            theta[0:self.d] = theta[0:self.d] / norm_theta
            

        return theta
    
    def fit(self, theta_prev, *args):
        x, y, S, V = args[0], args[1], args[2], args[3]
        means = np.dot(x[S], theta_prev)
        u = np.exp(means)
        SumExp = np.sum(u)+1
        self.g=np.zeros(self.d)
        for i,n in enumerate(S):
            p = ((np.exp(np.dot(x[n], theta_prev)) / SumExp)-y[i])*x[n]
            self.g += p
    #         obj=self.g@(theta-theta_prev)+(1/2)*(theta-theta_prev)@np.linalg.inv(V)@(theta-theta_prev)    
        if len(S)>0:
            iter = 0
            theta=copy.deepcopy(theta_prev)
            while True:
                theta_=copy.deepcopy(theta)
                grad=self.g+V@(theta-theta_prev)
                theta = theta - 0.01*grad
                iter += 1
                if np.linalg.norm(theta-theta_) < 1e-5 or iter > 20:
                    break
            theta=self.project_to_unit_ball(theta)
        else:
            theta=theta_prev

        return theta   
  


    def match_elements(self, S, index):
        result = []

        for sublist in S:
            # Check if any element in the sublist matches any element in index
            matching_elements = [1 if element in index else 0 for element in sublist]
            result.append(matching_elements)

        return result
        
    def remove_elements(self, S, Q):
        for i, q_value in enumerate(Q):
            if q_value == 0:
                for j,sublist in enumerate(S):
                    if i in sublist:
                        sublist.remove(i)
        return S    
        
    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def __init__(self,seed,x,N,K,L,T):
        print('TS_QMB')
        np.random.seed(seed)
        random.seed(seed)
        self.seed=seed
        self.x=x
        self.S=[]
        self.d=len(self.x[0])
        self.alpha=1
        self.kappa=1/L**2
        self.N=N
        self.K=K
        self.L=L
        self.T=T
        self.lamb=1
        self.h=np.zeros((N,K))
        self.V=[]
        self.z=[]
        self.Ur=[]
        self.Q=np.ones(N)
        self.theta=np.zeros((self.K,self.d))
        self.M=self.construct_M()
        self.number=0
    def run(self,t,index):   
        if t==1:
            self.M=self.construct_M()
            self.S=copy.deepcopy(random.choice(self.M))
            for k in range(self.K):
                self.V.append(self.lamb*np.identity(self.d))
        else:
            y=self.match_elements(self.S, index) 

            for k in range(self.K):
                for n in self.S[k]:
                    self.V[k]+=(self.kappa/2)*np.outer(self.x[n],self.x[n])
                theta_prev=self.theta[k].copy()
                y_k=y[k]
                S_k=self.S[k]

                self.theta[k]=self.fit(theta_prev,self.x,y_k,S_k,self.V[k])

            M=math.ceil(1-(np.log(self.K*self.L)/np.log(1-1/(4*np.log(math.e*math.pi)))))
            self.beta=np.sqrt(1+(self.d/self.kappa)*np.log(1+(t*self.L)/(self.d*self.lamb)))

            for k in range(self.K):
                mean=self.theta[k]
                cov=self.beta**2*np.linalg.inv(self.V[k])
                theta_sample=np.random.multivariate_normal(mean, cov, M)
                for n in range(self.N):
                    self.h[n,k]=max(self.x[n]@theta_sample.T)
                    
            # self.Q=Q    
            R=0
            tmp_R=0
            self.number+=1
            for partition in self.M:
                # print(partition)
                for k in range(self.K):
                    if len(partition[k])>0:                    
                        tmp_R+=np.sum(np.exp(self.h[partition[k],k])@self.Q[partition[k]])/(1+np.sum(np.exp(self.h[partition[k],k])))
                if R<tmp_R:
                    R=tmp_R
                    self.S=copy.deepcopy(partition)
                tmp_R=0
        self.remove_elements(self.S, self.Q)


    def offer(self):
        return self.S   
    
    def name(self):
        return 'TS-QMB'    

    def M_len(self):
        return len(self.M)
    def number_result(self):
        return self.number
##################
