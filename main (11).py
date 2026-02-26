from Algorithms import *
from Environment import *
from Environment import ProphetInequalityEnv
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import multiprocessing



def run(repeat,d,n,i,noniid,noise_std=0.1):
    Q_sum=dict()
    avg_Q_sum=dict()
    oracle_reward=0
    Q_sum_list=dict()
    std=dict()
    index=dict()
    Q=dict()
    Env_dict=dict()
    S=dict()
    exp_reward=dict()
    avg_regret_sum=dict()
    oracle_reward=dict()
    regret_sum_list=dict()
    std_R=dict()
    for algorithm in ['ETD-LCBT(iid)','ε-Greedy-LCBT','ETD-LCBT(non-iid)','ETD-LCBT-WA', 'Gusein-Zade']:
        exp_reward[algorithm]=[]
        oracle_reward[algorithm]=[] 
        regret_sum_list[algorithm]=np.zeros((repeat,1),float)
        avg_regret_sum[algorithm]=np.zeros(1,float)
        std_R[algorithm]=np.zeros(1,float)
        
    print('repeat',i)
    seed=i

    if noniid==True:
        Env=Noniid_ProphetInequalityEnv(seed,d,n,noise_std)
        alg_LCBT_Noniid=ETD_LCBT_NonIID(seed,d,n)
        alg_LCBT_Noniid_Window=ETD_LCBT_NonIID_Window(seed,d,n)
        alg_Secretary=Secretary(seed,d,n)
    else:
        alg_LCBT=ETD_LCBT(seed,d,n)
        alg_greedy=greedy(seed,d,n)
        alg_Secretary=Secretary(seed,d,n)

    if noniid==True:
        algorithms=[alg_LCBT_Noniid, alg_LCBT_Noniid_Window,  alg_Secretary]
        for algorithm in algorithms:
            name=algorithm.name()
            exp_reward[name]=[]              
            Env_dict[name]=Noniid_ProphetInequalityEnv(seed,d,n,noise_std)
    else:
        algorithms=[alg_LCBT, alg_greedy, alg_Secretary]

        for algorithm in algorithms:
            name=algorithm.name()
            exp_reward[name]=[]              
            Env_dict[name]=ProphetInequalityEnv(seed,d,n,noise_std)


    for algorithm in algorithms:
        Env = Env_dict[algorithm.name()]
        algorithm.reset()
        name=algorithm.name()
        print(name)
        for t in tqdm((np.array(range(n)))):
            # if t==0:
            x=Env.get_item(t)
            y=Env.recommend_and_feedback(t)
            if noniid==True:
                l=Env.get_inform_dis()[0]
                h=Env.get_inform_dis()[1]
                if name == 'Gusein-Zade':
                    algorithm.run(t,x,y)
                else:
                    algorithm.run(t,x,y,l,h)
            else:
                algorithm.run(t,x,y)

            if algorithm.stopped:
                if algorithm.tau==n:
                    exp_reward[name].append(0)
                else:
                    exp_reward[name].append(Env.stop_and_choose(algorithm.tau))
                oracle_reward[name].append(Env.get_optimal_reward())
                break

        filename_1=name+'n'+str(n)+'d'+str(d)+'repeat'+str(i)+'noise_std'+str(noise_std)+'alg.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(exp_reward[name], f)
            f.close()   


        filename_1=name+'n'+str(n)+'d'+str(d)+'repeat'+str(i)+'noise_std'+str(noise_std)+'oracle.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(oracle_reward[name], f)
            f.close()

def run_multiprocessing(repeat,d,n,noniid,noise_std=0.1):
    Path("./result").mkdir(parents=True, exist_ok=True)
    
        
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(run, [( repeat, d,n,i,noniid,noise_std) for i in range(repeat)])

    pool.close()
    pool.join()

if __name__=='__main__':
    
    d=2  
    repeat=10
    noniid=False
    noise_std=0.8
    if noniid==True:
        n=30000
    else:
        n=100000 
    print(d,n,repeat,noise_std)
    run_multiprocessing(repeat,d,n,noniid,noise_std)
