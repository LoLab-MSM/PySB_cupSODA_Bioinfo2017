import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid
import os
from pysb.examples.tyson_oscillator import model as tyson
from ras_amp_pka import model as ras

"""
Creates supplmental figure X and Y.
Calculates sensitivities of each initial condition from pairwise interactions.

"""

figs = os.path.join('.', 'FIGS')
if not os.path.exists(figs):
    os.makedirs(figs)


def create_boxplot_and_heatplot(model, data, x_axis_label, savename):
    proteins_of_interest = []
    for i in model.initial_conditions:
        proteins_of_interest.append(i[1].name)

    colors = 'seismic'

    sens_matrix = np.loadtxt(data)
    values = np.linspace(.8, 1.2, 21)
    length_values = len(values)
    length_image = len(sens_matrix)
    median = int(np.median(range(0, length_values)))
    sens_ij_nm = []
    for j in range(0, length_image, length_values):
        per_protein1 = []
        for i in range(0, length_image, length_values):
            if i == j:
                continue
            tmp = sens_matrix[j:j + length_values, i:i + length_values].copy()
            tmp -= tmp[median, :]  # sens_ij_0m
            per_protein1.append(tmp)
        sens_ij_nm.append(per_protein1)

    v_max = max(np.abs(sens_matrix.min()), sens_matrix.max())
    v_min = -1 * v_max

    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(2, 1, 1)

    im = ax1.imshow(sens_matrix, interpolation='nearest', origin='lower', cmap=plt.get_cmap(colors), vmin=v_min,
                    vmax=v_max,
                    extent=[0, length_image, 0, length_image])
    shape_label = np.arange(length_values / 2, length_image, length_values)
    plt.xticks(shape_label, proteins_of_interest, rotation='vertical', fontsize=12)
    plt.yticks(shape_label, proteins_of_interest, fontsize=12)
    xticks = ([i for i in range(0, length_image, length_values)])
    ax1.set_xticks(xticks, minor=True)
    ax1.set_yticks(xticks, minor=True)
    plt.grid(True, which='minor', axis='both', linestyle='--')
    divider = axgrid.make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="5%", pad=0.3)
    cax.tick_params(labelsize=12)

    if savename == 'tyson_sensitivity_boxplot.png':
        ticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    if savename == 'ras_sensitivity_boxplot.png':
        ticks = [-120, -60, 0, 60, 120]

    color_bar = fig.colorbar(im, cax=cax, ticks=ticks, orientation='horizontal')
    color_bar.set_ticks(ticks)
    color_bar.ax.set_xticklabels(ticks)
    color_bar.set_label('% change', labelpad=-40, y=0.45)
    ax2 = plt.subplot(2, 1, 2)
    ax2.boxplot(sens_ij_nm, vert=False, labels=None, showfliers=False)
    ax2.set_xlabel(x_axis_label, fontsize=12)
    xtickNames = plt.setp(ax2, yticklabels=proteins_of_interest)
    ax2.yaxis.tick_left()
    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=16,
                 xytext=(-55, 75), textcoords='offset points',
                 ha='left', va='top')
    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=16,
                 xytext=(-25, 25), textcoords='offset points',
                 ha='left', va='top')
    plt.tight_layout(h_pad=2.5)
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join('FIGS', savename), bbox_tight='True')
    plt.show()


# Supplemental figure
create_boxplot_and_heatplot(tyson,
                            os.path.join('Data', 'here_here_image_matrix.csv'),
                            'Percent change in period',
                            'tyson_sensitivity_boxplot.png')

# Supplemental figure
create_boxplot_and_heatplot(ras,
                            os.path.join('Data', 'sens_ras_matrix.csv'),
                            'Percent change in cAMP count',
                            'ras_sensitivity_boxplot.png')
