import numpy as np
import matplotlib.pyplot as plt
import os

"""
Creates supplmental figure X.
Timing of various GPUs.
"""

figs = os.path.join('.', 'FIGS')
if not os.path.exists(figs):
    os.makedirs(figs)

gpu_timings = os.path.join('Data', 'gpu_timings.csv')
cupsoda_data = np.genfromtxt(gpu_timings, delimiter=',', dtype=None, names=True)

plot_index = [(0, 0), (1, 0), (2, 0)]
plt.figure()
for count, model in enumerate(['tyson', 'ras', 'earm']):
    if model == 'tyson':
        ax1 = plt.subplot2grid((3, 1), plot_index[count])
    if model == 'ras':
        ax2 = plt.subplot2grid((3, 1), plot_index[count], sharex=ax1)
    if model == 'earm':
        ax3 = plt.subplot2grid((3, 1), plot_index[count], sharex=ax1)
    n_simulations = []
    for x in [d['nsims'] for d in cupsoda_data if d['model'] == model]:
        if x not in n_simulations:
            n_simulations.append(x)

    fmt = ['>-', '8-', '.-', '*-']
    cards = ['K20c-puma', 'gtx980-mule', 'gtx760-lolab', 'gtx980-diablo']
    colors = ['purple', 'orange', 'darkblue', 'green', ]
    labels = ['K20C', 'gtx970', 'gtx760', 'gtx980-TI']

    for i in range(len(cards)):
        times = []
        num_sims = []
        for x in n_simulations:
            data = cupsoda_data[cupsoda_data['model'] == model]
            data = data[data['card'] == cards[i]]
            data = data[data['nsims'] == x]
            if len(data) == 0:
                continue
            else:
                times.append(float(data['cupsodatime']))
                num_sims.append(x)

        if model == 'tyson':
            ax1.plot(num_sims, times, fmt[i], ms=10, lw=3, mew=2, mec=colors[i], color=colors[i],
                     label=labels[i])
        if model == 'ras':
            ax2.plot(num_sims, times, fmt[i], ms=10, lw=3, mew=2, mec=colors[i], color=colors[i],
                     label=labels[i])
        if model == 'earm':
            ax3.plot(num_sims, times, fmt[i], ms=10, lw=3, mew=2, mec=colors[i], color=colors[i],
                     label=labels[i])
        plt.xscale('log')
        plt.yscale('log')

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
ax1.yaxis.set_tick_params(labelsize=14)
ax2.yaxis.set_tick_params(labelsize=14)
ax3.yaxis.set_tick_params(labelsize=14)
ax1.legend(fontsize=14, bbox_to_anchor=(.6, 1.1), fancybox=True)
ax2.set_ylabel('Time (s)', fontsize=14)
ax3.set_xlabel("Number of simulations", fontsize=14)

ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=20,
             xytext=(-60, 10), textcoords='offset points',
             ha='left', va='top')
ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=20,
             xytext=(-60, 10), textcoords='offset points',
             ha='left', va='top')
ax3.annotate('C', xy=(0, 1), xycoords='axes fraction', fontsize=20,
             xytext=(-60, 10), textcoords='offset points',
             ha='left', va='top')

ax1.annotate('Tyson', xy=(0, 1), xycoords='axes fraction', fontsize=20,
             xytext=(5, -5), textcoords='offset points',
             ha='left', va='top')
ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction', fontsize=20,
             xytext=(5, -5), textcoords='offset points',
             ha='left', va='top')
ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=20,
             xytext=(5, -5), textcoords='offset points',
             ha='left', va='top')
y_lim = [1, 500]
x_lim = [10, 10000]
ax1.set_xlim(x_lim)
ax1.set_ylim(.1, 500)
ax2.set_xlim(x_lim)
ax2.set_ylim(y_lim)
ax3.set_xlim(x_lim)
ax3.set_ylim(y_lim)
plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
plt.savefig(os.path.join(figs, 'gpu_comparison.png'), bbox_tight='True')
