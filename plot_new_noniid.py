from Environment import *
from Algorithms import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

def plot(n, repeat, d,noise_std=0.1):
    algorithms = ['ETD-LCBT(non-iid)','ETD-LCBT-WA','Gusein-Zade']
    exp_reward_sum = {alg: 0.0 for alg in algorithms}
    oracle_reward_sum = {alg: 0.0 for alg in algorithms}
    ratio = {alg: 0.0 for alg in algorithms}

    for alg in algorithms:
        for i in range(repeat):
            alg_file = f'./result/{alg}n{n}d{d}repeat{i}noise_std{noise_std}alg.txt'
            oracle_file = f'./result/{alg}n{n}d{d}repeat{i}noise_std{noise_std}oracle.txt'
            with open(alg_file, "rb") as f:
                alg_reward = pickle.load(f)[0]
            with open(oracle_file, "rb") as f:
                oracle_reward = pickle.load(f)[0]
            exp_reward_sum[alg] += alg_reward
            oracle_reward_sum[alg] += oracle_reward
        ratio[alg] = exp_reward_sum[alg] / oracle_reward_sum[alg]

    # Plot all algorithms' ratios in one plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['gray', 'royalblue', 'gold', 'green']
    markers = ['P', '<', 'v', 'o']
    x = np.array([-0.15, 0,0.15])
    y = [ratio[alg] for alg in algorithms]

    for idx, alg in enumerate(algorithms):
        ax.plot(
            x[idx], y[idx],
            marker=markers[idx], color=colors[idx],
            label=alg, markersize=14, linestyle='None', zorder=10
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=18)

    ax.set_xlim(-0.2, 0.2)     
    ax.margins(x=0)            
    ax.axhline(
    y=1/2,       
    color='orange',         
    linestyle='--',        
    linewidth=1.5,        
    label=r'$1/2$'    
    )
    
    ref_value = 1/2
    ref_label = r'$1/2$'

    ax.set_ylim(0, 1.5)

    ax.axhline(ref_value, color='orange', linestyle='--', linewidth=1.5)

    yticks = ax.get_yticks()
    if not np.any(np.isclose(yticks, ref_value)):
        yticks = np.sort(np.append(yticks, ref_value))

    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))

    fig.canvas.draw()

    for lbl, val in zip(ax.get_yticklabels(), yticks):
        if np.isclose(val, ref_value):
            lbl.set_color('orange')
            lbl.set_fontweight('bold')

    ax.set_ylabel(r'$\mathrm{Ratio}$', fontsize=18)
    ax.set_title(f'Competitive Ratio of Algorithms (n={n})', fontsize=20)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'./plot/n{n}d{d}repeat{repeat}noise_std{noise_std}_noniid.pdf', bbox_inches="tight")


if __name__ == '__main__':
    Path("./plot").mkdir(parents=True, exist_ok=True)
    d = 2
    repeat = 10
    n = 30000
    noise_std=0.1
    plot(n, repeat, d ,noise_std)
