import os
import matplotlib.pyplot as plt
import numpy as np
from models.earm_lopez_embedded_flat import model as earm

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
        if model == 'tyson':
            ax1 = plt.subplot2grid((3, 1), (0, 0))
        if model == 'ras':
            ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
        if model == 'earm':
            ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

        # SciPy
        xdata = [d['nsims'] for d in scipy_data if d['model'] == model and d['num_cpu'] == 1]
        ydata = [d['scipytime'] for d in scipy_data if d['model'] == model and d['num_cpu'] == 1]
        if model == 'tyson':
            ax1.plot(xdata, ydata, 'b-o', label='SciPy (lsoda)', ms=12, lw=3, mew=0, )
        if model == 'ras':
            ax2.plot(xdata, ydata, 'b-o', label='SciPy (lsoda)', ms=12, lw=3, mew=0, )
        if model == 'earm':
            ax3.plot(xdata, ydata, 'b-o', label='SciPy (lsoda)', ms=12, lw=3, mew=0, )

        # cupSODA
        xdata = []
        for x in [d['nsims'] for d in cupsoda_data if d['model'] == model and d['card'] == 'gtx980-diablo']:
            if x not in xdata:
                xdata.append(x)
        ydata = []
        for x in xdata:
            ydata.append([d['pythontime'] for d in cupsoda_data
                          if d['model'] == model and d['mem'] == 2 and d['card'] == 'gtx980-diablo' and d[
                              'nsims'] == x])
        cupsoda_time = []
        for x in xdata:
            cupsoda_time.append([d['cupsodatime'] for d in cupsoda_data
                          if d['model'] == model and d['mem'] == 2 and d['card'] == 'gtx980-diablo' and d[
                              'nsims'] == x])
        if model == 'tyson':
            ax1.plot(xdata, ydata, '-v', ms=12, lw=3, mew=2, mec='red', color='red', label='PySB/cupSODA')
            ax1.plot(xdata, cupsoda_time, '-*', ms=12, lw=3, mew=2, mec='green', color='green', label='cupSODA')
        if model == 'ras':
            ax2.plot(xdata, ydata, '-v', ms=12, lw=3, mew=2, mec='red', color='red', label='PySB/cupSODA')
            ax2.plot(xdata, cupsoda_time, '-*', ms=12, lw=3, mew=2, mec='green', color='green', label='cupSODA')
        if model == 'earm':
            ax3.plot(xdata, ydata, '-v', ms=12, lw=3, mew=2, mec='red', color='red', label='PySB/cupSODA')
            ax3.plot(xdata, cupsoda_time, '-*', ms=12, lw=3, mew=2, mec='green', color='green', label='cupSODA')

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

    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=distance, textcoords='offset points',
                 ha='left', va='top')
    ax3.annotate('C', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(-60, 10), textcoords='offset points',
                 ha='left', va='top')
    ax1.annotate('Tyson', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(5, -5), textcoords='offset points',
                 ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(5, -5), textcoords='offset points',
                 ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2, xytext=(5, -5), textcoords='offset points',
                 ha='left', va='top')

    plt.tight_layout()
    fig.subplots_adjust(hspace=.02, wspace=.1)
    plt.savefig(os.path.join(figs, 'supp_figure_1_compare_runtime.png'), bbox_tight='True')


if __name__ == '__main__':
    create_supplement_figure_1()
