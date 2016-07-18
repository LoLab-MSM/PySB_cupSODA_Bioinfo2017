import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gridspec
import os
from models.earm_lopez_embedded_flat import model

""""
Creates supplemental figure X and Y
Sensitivity of EARM with two different parameter sets that fit equally well.
"""


def create_earm_boxplot(sensitivity_matrix, save_name):
    proteins_of_interest = []
    for i in model.initial_conditions:
        proteins_of_interest.append(i[1].name)
    proteins_of_interest = np.sort(proteins_of_interest)
    colors = 'seismic'

    vals = np.linspace(.8, 1.2, 11)
    median = int(np.median(range(0, len(vals))))

    all_runs_1 = []
    length_matrix = len(sensitivity_matrix)
    len_vals = len(vals)
    for j in range(0, length_matrix, len_vals):
        per_protein1 = []
        for i in range(0, length_matrix, len_vals):
            if i == j:
                continue
            tmp = sensitivity_matrix[j:j + len_vals, i:i + len_vals].copy()
            tmp -= tmp[median, :]
            per_protein1.append(tmp)
        all_runs_1.append(per_protein1)

    fig = plt.figure(figsize=(6, 12))
    gs1 = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])

    v_max = max(np.abs(sensitivity_matrix.min()), sensitivity_matrix.max())
    v_min = -1 * v_max

    n = len(sensitivity_matrix)
    im = ax1.imshow(sensitivity_matrix, interpolation='nearest',
                    origin='lower', cmap=plt.get_cmap(colors),
                    vmin=v_min, vmax=v_max,
                    extent=[0, n, 0, n])

    shape_label = ([i + len_vals / 2. for i in range(0, length_matrix, len_vals)])
    ax1.set_xticks(shape_label, )
    ax1.set_xticklabels(labels=proteins_of_interest, rotation='vertical', fontsize=14)
    ax1.set_yticks(shape_label)
    ax1.set_yticklabels(labels=proteins_of_interest, fontsize=14)
    x_ticks = ([i for i in range(0, length_matrix, len_vals)])
    ax1.set_xticks(x_ticks, minor=True)
    ax1.set_yticks(x_ticks, minor=True)
    ax1.grid(True, which='minor', axis='both', linestyle='--')
    divider = axgrid.make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    cb = plt.colorbar(im, cax=cax, orientation='horizontal', use_gridspec=True)
    cb.set_label('% change', labelpad=-40, y=0.45)

    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()

    ax1.set_aspect((float(x1) - float(x0)) / (float(y1) - float(y0)))

    ax2.boxplot(all_runs_1, vert=False, labels=None, showfliers=False)
    ax2.set_xlabel('Percent change in time-to-death', fontsize=14)
    ax2.set_yticklabels(labels=proteins_of_interest, fontsize=14)

    ax2.yaxis.tick_left()

    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=24,
                 xytext=(-65, 55), textcoords='offset points',
                 ha='left', va='top')
    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=24,
                 xytext=(-60, 55), textcoords='offset points',
                 ha='left', va='top')

    plt.tight_layout(rect=[0, 0, 1, 1], h_pad=.4)
    plt.subplots_adjust(top=0.95, hspace=.2)

    plt.savefig(os.path.join('Figures', '%s.png') % save_name, bbox_tight='True', dpi=300)
    plt.savefig(os.path.join('Figures', '%s.eps') % save_name, bbox_tight='True')
    plt.savefig(os.path.join('Figures', '%s.pdf') % save_name, bbox_tight='True')


def create_supplement_figure_9():
    data = np.loadtxt(os.path.join('SensitivityData', 'sens_earm_parameter_set_one.csv'))
    create_earm_boxplot(data, 'supp_figure_9_earm_parameter_set_1')


def create_supplement_figure_10():
    data = np.loadtxt(os.path.join('SensitivityData', 'sens_earm_parameter_set_two.csv'))
    create_earm_boxplot(data, 'supp_figure_10_earm_parameter_set_2')


if __name__ == '__main__':
    create_supplement_figure_9()
    create_supplement_figure_10()
