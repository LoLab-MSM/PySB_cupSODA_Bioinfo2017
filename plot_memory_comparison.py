"""
Creates supplemental figure X
Comparison of different memory options

"""

import os
import matplotlib.pyplot as plt
import numpy as np

def create_supplement_figure_2():
    datafile = os.path.join('TimingData', 'gpu_memory.csv')
    cupsoda_data = np.genfromtxt(datafile, delimiter=',', dtype=None, names=True)
    plt.figure()
    for count, model in enumerate(['tyson', 'ras', 'earm']):
        if model == 'tyson':
            ax1 = plt.subplot2grid((3, 1), (0, 0))
        if model == 'ras':
            ax2 = plt.subplot2grid((3, 1), (1, 0))
        if model == 'earm':
            ax3 = plt.subplot2grid((3, 1), (2, 0))
        n_simulations = []
        for x in [d['nsims'] for d in cupsoda_data if d['model'] == model]:
            if x not in n_simulations:
                n_simulations.append(x)

        fmt = ['^-', 's-', '*-']
        colors = ['c', 'magenta', 'green', ]
        labels = ['global', 'global+shared', 'global+shared+constant']
        mem = [0, 1, 2]
        for i in range(len(mem)):
            times = []
            num_sims = []
            for x in n_simulations:
                data = cupsoda_data[cupsoda_data['model'] == model]
                data = data[data['mem'] == mem[i]]
                data = data[data['nsims'] == x]
                if len(data) == 0:
                    continue
                else:
                    times.append(float(data['cupsodatime']))
                    num_sims.append(x)

            if model == 'tyson':
                ax1.plot(num_sims, times, fmt[i], ms=10, lw=3, mew=2, mec=colors[i], color=colors[i], label=labels[i])
            if model == 'ras':
                ax2.plot(num_sims, times, fmt[i], ms=10, lw=3, mew=2, mec=colors[i], color=colors[i], label=labels[i])
            if model == 'earm':
                ax3.plot(num_sims, times, fmt[i], ms=10, lw=3, mew=2, mec=colors[i], color=colors[i], label=labels[i])
            plt.xscale('log')
            plt.yscale('log')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    ax3.yaxis.set_tick_params(labelsize=14)

    ax1.legend(fontsize=14, bbox_to_anchor=(.75, 1.0), fancybox=True)
    ax2.set_ylabel('Time (s)', fontsize=14)
    ax3.set_xlabel("Number of simulations", fontsize=14)

    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=20, xytext=(-60, 10), textcoords='offset points',
                 ha='left', va='top')
    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=20, xytext=(-60, 10), textcoords='offset points',
                 ha='left', va='top')
    ax3.annotate('C', xy=(0, 1), xycoords='axes fraction', fontsize=20, xytext=(-60, 10), textcoords='offset points',
                 ha='left', va='top')

    ax1.annotate('Cell Cycle', xy=(0, 1), xycoords='axes fraction', fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction', fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')

    x_lim = [-10, 10000]
    ax1.set_xlim(x_lim)
    ax1.set_ylim(.1, 200)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(2, 200)
    ax3.set_xlim(x_lim)
    ax3.set_ylim(2, 200)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(os.path.join('Figures', 'supp_figure_2_compare_memory.png'), bbox_tight='True')


if __name__ == '__main__':
    create_supplement_figure_2()
