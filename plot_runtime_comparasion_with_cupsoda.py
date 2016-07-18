import os
import matplotlib.pyplot as plt
import numpy as np


"""
Creates figure 1A and 1B.

"""
figs = os.path.join('.', 'Figures')
if not os.path.exists(figs):
    os.makedirs(figs)


def create_supplement_figure_1():
    datafile = os.path.join('TimingData', 'gpu_timings.csv')
    cupsoda_data = np.genfromtxt(datafile, delimiter=',', dtype=None, names=True)

    datafile = os.path.join('TimingData', 'scipy_timings.csv')
    scipy_data = np.genfromtxt(datafile, delimiter=',', dtype=None, names=True)

    fig = plt.figure(figsize=(8, 6))

    for count, model in enumerate(['tyson', 'ras', 'earm']):
        # SciPy
        scipy_n_sim = [d['nsims'] for d in scipy_data if d['model'] == model and d['num_cpu'] == 1]
        scipy_time = [d['scipytime'] for d in scipy_data if d['model'] == model and d['num_cpu'] == 1]
        # cupSODA
        cupsoda_n_sims = []
        for x in [d['nsims'] for d in cupsoda_data if d['model'] == model and d['card'] == 'gtx980-diablo']:
            if x not in cupsoda_n_sims:
                cupsoda_n_sims.append(x)
        cupsoda_time = []
        cupsoda_raw_time = []
        for x in cupsoda_n_sims:
            cupsoda_raw_time.append([d['pythontime'] for d in cupsoda_data
                                     if d['model'] == model and d['mem'] == 2 and d['card'] == 'gtx980-diablo' and d[
                                         'nsims'] == x])
            cupsoda_time.append([d['cupsodatime'] for d in cupsoda_data
                                 if d['model'] == model and d['mem'] == 2 and d['card'] == 'gtx980-diablo' and d[
                                     'nsims'] == x])

        if model == 'tyson':
            ax1 = plt.subplot2grid((3, 1), (0, 0))
            ax1.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)', ms=12, lw=3, mew=0, )
            ax1.plot(cupsoda_n_sims, cupsoda_raw_time, '-v', ms=12, lw=3, mew=2, mec='red', color='red',
                     label='PySB/cupSODA')
            ax1.plot(cupsoda_n_sims, cupsoda_time, '-*', ms=12, lw=3, mew=2, mec='green', color='green',
                     label='cupSODA')

        if model == 'ras':
            ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
            ax2.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)', ms=12, lw=3, mew=0, )
            ax2.plot(cupsoda_n_sims, cupsoda_raw_time, '-v', ms=12, lw=3, mew=2, mec='red', color='red',
                     label='PySB/cupSODA')
            ax2.plot(cupsoda_n_sims, cupsoda_time, '-*', ms=12, lw=3, mew=2, mec='green', color='green',
                     label='cupSODA')
        if model == 'earm':
            ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
            ax3.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)', ms=12, lw=3, mew=0, )
            ax3.plot(cupsoda_n_sims, cupsoda_raw_time, '-v', ms=12, lw=3, mew=2, mec='red', color='red',
                     label='PySB/cupSODA')
            ax3.plot(cupsoda_n_sims, cupsoda_time, '-*', ms=12, lw=3, mew=2, mec='green', color='green',
                     label='cupSODA')

        plt.xscale('log')
        plt.yscale('log')
    ax1.set_xlim(0, 10000)
    ax1.set_ylim(0, 100)

    ax2.set_xlim(0, 10000)
    ax2.set_ylim(0, 1000)

    ax3.set_xlim(0, 10000)
    ax3.set_ylim(0, 1000)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax1.set_yticks(ax1.get_yticks()[3:])
    ax2.set_yticks(ax2.get_yticks()[3:])

    f_size1 = 14
    f_size2 = 20

    ax1.yaxis.set_tick_params(labelsize=f_size1)
    ax2.yaxis.set_tick_params(labelsize=f_size1)
    ax3.yaxis.set_tick_params(labelsize=f_size1)
    ax3.xaxis.set_tick_params(labelsize=f_size1)
    ax1.legend(fontsize=14, bbox_to_anchor=(.9, 1.19), fancybox=True)

    ax2.set_ylabel('Time (s)', fontsize=f_size1)
    ax3.set_xlabel("Number of simulations", fontsize=f_size1)
    distance = (-60, 10)
    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(-60, 18), textcoords='offset points',
                 ha='left', va='top')

    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=distance,
                 textcoords='offset points',
                 ha='left', va='top')
    ax3.annotate('C', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(-60, 10),
                 textcoords='offset points',
                 ha='left', va='top')
    ax1.annotate('Tyson', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(5, -5),
                 textcoords='offset points',
                 ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(5, -5),
                 textcoords='offset points',
                 ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(5, -5),
                 textcoords='offset points',
                 ha='left', va='top')

    plt.tight_layout()
    fig.subplots_adjust(hspace=.02, wspace=.1)
    plt.savefig(os.path.join(figs, 'supp_figure_1_compare_runtime.png'), bbox_tight='True')


if __name__ == '__main__':
    create_supplement_figure_1()
