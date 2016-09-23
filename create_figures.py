import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os

pd.options.display.float_format = '{:,.3f}'.format
figs = os.path.join('.', 'Figures')
if not os.path.exists(figs):
    os.makedirs(figs)

gpu_timings = os.path.join('TimingData', 'new_gpu_timing_all.csv')
cupsoda = pd.read_csv(gpu_timings)


# speedup of GPU
def print_table_1(model):
    """ prints information on speedup of pysb/cupsoda over scipy

    :param model:
    :return:
    """
    data = cupsoda[cupsoda['model'] == model]
    data = data[data['tpb'] == 16]
    data = data[data['mem'] == 2]
    # main comparasion is to the gtx980-ti on diablo
    data = data[data['card'] == 'gtx-980-ti']
    data = data[['nsims', 'cupsodatime', 'pythontime', 'model']]
    py_time = np.array(data['pythontime'])
    cs_raw_time = np.array(data['cupsodatime'])
    datafile = os.path.join('TimingData', 'scipy_timings.csv')
    scipy_data = pd.read_csv(datafile)
    scipy_data = scipy_data[scipy_data['num_cpu'] == 1]
    scipy_data = scipy_data[scipy_data['model'] == model]
    sc_times = np.array(scipy_data['scipytime'])

    data['scipy'] = sc_times
    data['speed_up'] = sc_times / py_time
    data['overhead_fraction'] = (py_time - cs_raw_time) / py_time * 100
    data['pysb_cupsoda'] = data['pythontime']
    index = ['model', 'nsims', 'scipy', 'pysb_cupsoda', 'speed_up',
             'cupsodatime', 'overhead_fraction']
    string = data[index].to_string(index=False)
    print(string)


def table_2_print_stats_on_memory(model):
    data = cupsoda[cupsoda['model'] == model]
    data = data[data['tpb'] == 16]
    data = data[data['card'] == 'gtx-980-ti']

    glob = np.array(data[data['mem'] == 0]['cupsodatime'])
    shared = np.array(data[data['mem'] == 1]['cupsodatime'])
    shar_cons = np.array(data[data['mem'] == 2]['cupsodatime'])
    data2 = pd.DataFrame()
    data2['nsims'] = data[data['mem'] == 2]['nsims']
    data2['global'] = glob
    data2['model'] = model
    data2['shared'] = shared
    data2['sharedconstant'] = shar_cons
    data2['speed_1'] = (glob - shared) / glob * -100
    data2['speed_2'] = (shared - shar_cons) / shared * -100

    string = data2[
        ['model', 'nsims', 'global', 'shared', 'speed_1', 'sharedconstant',
         'speed_2']].to_string(index=False)
    print(string)


def table_3_print_stats_on_memory(model):
    data = cupsoda[cupsoda['model'] == model]
    data = data[data['tpb'] == 16]
    data = data[data['mem'] == 2]

    out = pd.DataFrame()
    out['model'] = data[data['card'] == 'K20C']['model']
    out['nsims'] = data[data['card'] == 'K20C']['nsims']
    out['gtx980ti'] = np.array(
            data[data['card'] == 'gtx-980-ti']['cupsodatime'])
    out['gtx970'] = np.array(data[data['card'] == 'gtx-970']['cupsodatime'])
    out['gtx-760'] = np.array(data[data['card'] == 'gtx-760']['cupsodatime'])
    out['k20c'] = np.array(data[data['card'] == 'K20C']['cupsodatime'])

    string = out.to_string(index=False)
    print(string)


def create_figure_1():
    from models.earm_lopez_embedded_flat import model as earm
    datafile = os.path.join('TimingData', 'new_gpu_timing_all.csv')
    cupsoda_data = pd.read_csv(datafile)
    cupsoda_data = cupsoda_data[cupsoda_data['card'] == 'gtx-980-ti']
    cupsoda_data = cupsoda_data[cupsoda_data['tpb'] == 16]
    cupsoda_data = cupsoda_data[cupsoda_data['mem'] == 2]
    datafile = os.path.join('TimingData', 'scipy_timings.csv')
    scipy_data = pd.read_csv(datafile)
    scipy_data = scipy_data[scipy_data['num_cpu'] == 1]

    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (1, 0), sharex=ax1)
    ax3 = plt.subplot2grid((3, 2), (2, 0), sharex=ax1)
    ax4 = plt.subplot2grid((3, 2), (0, 1), rowspan=3)

    for model in ['tyson', 'ras', 'earm']:
        # SciPy
        tmp_data = scipy_data[scipy_data['model'] == model]
        scipy_n_sim = np.array(tmp_data['nsims'])
        scipy_time = np.array(tmp_data['scipytime'])

        # cupSODA
        cupsoda_tmp = cupsoda_data[cupsoda_data['model'] == model]
        cupsoda_n_sims = np.array(cupsoda_tmp['nsims'])
        cupsoda_time = np.array(cupsoda_tmp['cupsodatime'])

        if model == 'tyson':
            ax1.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)',
                     ms=12, lw=3, mew=0, )
            ax1.plot(cupsoda_n_sims, cupsoda_time, marker='^', ls='-', ms=12,
                     lw=3, mew=2, mec='red', color='red', label='PySB/cupSODA')
        if model == 'ras':
            ax2.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)',
                     ms=12, lw=3, mew=0, )
            ax2.plot(cupsoda_n_sims, cupsoda_time, marker='^', ls='-', ms=12,
                     lw=3, mew=2, mec='red', color='red', label='PySB/cupSODA')
        if model == 'earm':
            ax3.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)',
                     ms=12, lw=3, mew=0, )
            ax3.plot(cupsoda_n_sims, cupsoda_time, marker='^', ls='-', ms=12,
                     lw=3, mew=2, mec='red', color='red', label='PySB/cupSODA')

    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    x_limit = [0, 11000]
    ax1.set_xlim(x_limit)
    ax1.set_ylim(0, 100)
    ax2.set_xlim(x_limit)
    ax2.set_ylim(0, 1000)
    ax3.set_xlim(x_limit)
    ax3.set_ylim(0, 1000)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax1.set_yticks(ax1.get_yticks()[3:])
    ax2.set_yticks(ax2.get_yticks()[3:])

    f_size1 = 18
    f_size2 = 24

    ax1.yaxis.set_tick_params(labelsize=f_size1)
    ax2.yaxis.set_tick_params(labelsize=f_size1)
    ax3.yaxis.set_tick_params(labelsize=f_size1)
    ax3.xaxis.set_tick_params(labelsize=f_size1)
    ax1.legend(fontsize=14, bbox_to_anchor=(.9, 1.19), fancybox=True)

    ax2.set_ylabel('Time (s)', fontsize=f_size1)
    ax3.set_xlabel("Number of simulations", fontsize=f_size1)

    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(-70, 25),
                 textcoords='offset points', ha='left', va='top')
    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(-70, 10),
                 textcoords='offset points', ha='left', va='top')
    ax3.annotate('C', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(-70, 10),
                 textcoords='offset points', ha='left', va='top')

    ax1.annotate('Cell cycle', xy=(0, 1), xycoords='axes fraction',
                 fontsize=f_size2, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction',
                 fontsize=f_size2, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')

    proteins_of_interest = []
    for i in earm.initial_conditions:
        proteins_of_interest.append(i[1].name)
    proteins_of_interest = sorted(proteins_of_interest)
    vals = np.linspace(.8, 1.2, 11)
    median = int(np.median(range(0, len(vals))))
    p_matrix = np.loadtxt(
            os.path.join('SensitivityData', 'earm_parameters_1_p_matrix.csv'))
    p_prime_matrix = np.loadtxt(
            os.path.join('SensitivityData',
                         'earm_parameters_1_p_prime_matrix.csv'))

    sens_matrix = p_matrix - p_prime_matrix
    all_runs_1 = []
    length_values = len(vals)
    length_image = len(sens_matrix)
    for j in range(0, length_image, length_values):
        per_protein1 = []
        for i in range(0, length_image, length_values):
            if i == j:
                continue
            tmp = sens_matrix[j:j + length_values, i:i + length_values].copy()
            tmp -= tmp[median, :]
            per_protein1.append(tmp)
        all_runs_1.append(per_protein1)

    ax4.boxplot(all_runs_1, vert=False, labels=None, showfliers=False)
    ax4.set_xlabel('Percent change in time-to-death', fontsize=f_size1)
    xtick_names = plt.setp(ax4, yticklabels=reversed(proteins_of_interest))
    ax4.yaxis.tick_right()
    plt.setp(xtick_names, fontsize=30)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.tick_params(axis='x', which='major', labelsize=18)
    v_max = max(np.abs(p_matrix.min()), p_matrix.max())
    v_min = -1 * v_max
    ax4.set_xlim(v_min - 2, v_max + 2)
    ax4.annotate('D', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(0, 25), textcoords='offset points',
                 ha='left', va='top')
    plt.tight_layout()
    fig.subplots_adjust(hspace=.1, wspace=.01, left=.084, top=.93, bottom=0.1)
    plt.savefig(os.path.join(figs, 'figure_1.png'), bbox_tight='True')
    plt.savefig(os.path.join(figs, 'figure_1.eps'), bbox_tight='True')
    plt.savefig(os.path.join(figs, 'figure_1.pdf'), bbox_tight='True')


# compare pysb to cupsoda and scipy
def create_supplement_figure_1():
    gpu_timings = os.path.join('TimingData', 'new_gpu_timing_all.csv')
    cupsoda = pd.read_csv(gpu_timings)
    cupsoda = cupsoda[cupsoda['card'] == 'gtx-980-ti']
    cupsoda = cupsoda[cupsoda['tpb'] == 32]
    cupsoda = cupsoda[cupsoda['mem'] == 2]

    datafile = os.path.join('TimingData', 'scipy_timings.csv')
    scipy_data = pd.read_csv(datafile)
    scipy_data = scipy_data[scipy_data['num_cpu'] == 1]
    fig = plt.figure(figsize=(8, 6))

    for count, model in enumerate(['tyson', 'ras', 'earm']):
        # SciPy
        tmp_data = scipy_data[scipy_data['model'] == model]
        scipy_n_sim = np.array(tmp_data['nsims'])
        scipy_time = np.array(tmp_data['scipytime'])
        # cupSODA
        cupsoda_tmp = cupsoda[cupsoda['model'] == model]

        cupsoda_n_sims = np.array(cupsoda_tmp['nsims'])
        cupsoda_time = np.array(cupsoda_tmp['cupsodatime'])
        cupsoda_raw_time = np.array(cupsoda_tmp['pythontime'])

        if model == 'tyson':
            ax1 = plt.subplot2grid((3, 1), (0, 0))
            ax1.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)',
                     ms=12, lw=3, mew=0, )
            ax1.plot(cupsoda_n_sims, cupsoda_raw_time, '-v', ms=12, lw=3,
                     mew=2, mec='red', color='red',
                     label='PySB/cupSODA')
            ax1.plot(cupsoda_n_sims, cupsoda_time, '-*', ms=12, lw=3, mew=2,
                     mec='green', color='green',
                     label='cupSODA')

        if model == 'ras':
            ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
            ax2.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)',
                     ms=12, lw=3, mew=0, )
            ax2.plot(cupsoda_n_sims, cupsoda_raw_time, '-v', ms=12, lw=3,
                     mew=2, mec='red', color='red',
                     label='PySB/cupSODA')
            ax2.plot(cupsoda_n_sims, cupsoda_time, '-*', ms=12, lw=3, mew=2,
                     mec='green', color='green',
                     label='cupSODA')
        if model == 'earm':
            ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
            ax3.plot(scipy_n_sim, scipy_time, 'b-o', label='SciPy (lsoda)',
                     ms=12, lw=3, mew=0, )
            ax3.plot(cupsoda_n_sims, cupsoda_raw_time, '-v', ms=12, lw=3,
                     mew=2, mec='red', color='red',
                     label='PySB/cupSODA')
            ax3.plot(cupsoda_n_sims, cupsoda_time, '-*', ms=12, lw=3, mew=2,
                     mec='green', color='green',
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
    ax1.legend(fontsize=14, bbox_to_anchor=(.65, 1.19), fancybox=True)

    ax2.set_ylabel('Time (s)', fontsize=f_size1)
    ax3.set_xlabel("Number of simulations", fontsize=f_size1)
    distance = (-60, 10)
    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(-60, 18), textcoords='offset points',
                 ha='left', va='top')

    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=distance,
                 textcoords='offset points',
                 ha='left', va='top')
    ax3.annotate('C', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(-60, 10),
                 textcoords='offset points',
                 ha='left', va='top')
    ax1.annotate('Cell cycle', xy=(0, 1), xycoords='axes fraction',
                 fontsize=f_size2, xytext=(5, -5),
                 textcoords='offset points',
                 ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction',
                 fontsize=f_size2, xytext=(5, -5),
                 textcoords='offset points',
                 ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=f_size2,
                 xytext=(5, -5),
                 textcoords='offset points',
                 ha='left', va='top')

    plt.tight_layout()
    fig.subplots_adjust(hspace=.02, wspace=.1)
    plt.savefig(os.path.join('Figures', 'supp_figure_1_compare_runtime.png'),
                bbox_tight='True')
    plt.savefig(os.path.join('Figures', 'supp_figure_1_compare_runtime.eps'),
                bbox_tight='True')


# compare memory
def create_supplement_figure_2():
    gpu_timings = os.path.join('TimingData', 'new_gpu_timing_all.csv')
    cupsoda = pd.read_csv(gpu_timings)
    cupsoda = cupsoda[cupsoda['card'] == 'gtx-760']
    cupsoda = cupsoda[cupsoda['tpb'] == 16]
    plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax3 = plt.subplot2grid((3, 1), (2, 0))
    for model in ['tyson', 'ras', 'earm']:
        # cupSODA
        cupsoda_tmp = cupsoda[cupsoda['model'] == model]

        fmt = ['^-', 's-', '*-']
        colors = ['c', 'magenta', 'green', ]
        labels = ['global', 'shared', 'sharedconstant']
        mem = [0, 1, 2]
        for i in range(len(mem)):
            cupsoda_tmp2 = cupsoda_tmp[cupsoda_tmp['mem'] == mem[i]]
            cupsoda_n_sims = np.array(cupsoda_tmp2['nsims'])
            cupsoda_time = np.array(cupsoda_tmp2['cupsodatime'])
            if model == 'tyson':
                ax1.plot(cupsoda_n_sims, cupsoda_time, fmt[i], ms=10, lw=3,
                         mew=2, mec=colors[i], color=colors[i],
                         label=labels[i])
            if model == 'ras':
                ax2.plot(cupsoda_n_sims, cupsoda_time, fmt[i], ms=10, lw=3,
                         mew=2, mec=colors[i], color=colors[i],
                         label=labels[i])
            if model == 'earm':
                ax3.plot(cupsoda_n_sims, cupsoda_time, fmt[i], ms=10, lw=3,
                         mew=2, mec=colors[i], color=colors[i],
                         label=labels[i])

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    ax3.yaxis.set_tick_params(labelsize=14)

    ax1.legend(fontsize=14, bbox_to_anchor=(.75, 1.0), fancybox=True)
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

    ax1.annotate('Cell Cycle', xy=(0, 1), xycoords='axes fraction',
                 fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction',
                 fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=20,
                 xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')

    x_lim = [-10, 10000]
    ax1.set_xlim(x_lim)
    ax1.set_ylim(.01, 50)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(2, 200)
    ax3.set_xlim(x_lim)
    ax3.set_ylim(2, 200)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(os.path.join('Figures', 'supp_figure_2_compare_memory.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join('Figures', 'supp_figure_2_compare_memory.eps'),
                bbox_inches='tight')


def create_supplement_figure_3():
    # compare GPUs

    plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
    ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
    for model in ['tyson', 'ras', 'earm']:

        tmp_data = cupsoda[cupsoda['model'] == model]
        tmp_data = tmp_data[tmp_data['mem'] == 2]
        tmp_data = tmp_data[tmp_data['tpb'] == 16]
        tmp_data = tmp_data[tmp_data['card'] != 'gtx-1080']

        fmt = {'K20C'      : '>-',
               'gtx-970'   : '8-',
               'gtx-760'   : '.-',
               'gtx-980-ti': '*-',
               'gtx-1080'  : '<-'}

        colors = {'K20C'      : 'purple',
                  'gtx-970'   : 'orange',
                  'gtx-760'   : 'darkblue',
                  'gtx-980-ti': 'green',
                  'gtx-1080'  : 'red'}
        labels = {'K20C'      : 'K20C',
                  'gtx-970'   : 'gtx970',
                  'gtx-760'   : 'gtx760',
                  'gtx-980-ti': 'gtx980-TI',
                  'gtx-1080'  : 'gtx1080'}
        grouped = tmp_data.groupby('card')
        for n, j in grouped:
            times = np.array(j['cupsodatime'])
            num_sims = np.array(j['nsims'])
            if model == 'tyson':
                ax1.plot(num_sims, times, fmt[n], ms=10, lw=3, mew=2,
                         mec=colors[n], color=colors[n], label=labels[n])
            if model == 'ras':
                ax2.plot(num_sims, times, fmt[n], ms=10, lw=3, mew=2,
                         mec=colors[n], color=colors[n], label=labels[n])
            if model == 'earm':
                ax3.plot(num_sims, times, fmt[n], ms=10, lw=3, mew=2,
                         mec=colors[n], color=colors[n], label=labels[n])
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    ax3.yaxis.set_tick_params(labelsize=14)
    ax1.legend(fontsize=14, bbox_to_anchor=(.8, 1.1), fancybox=True, ncol=2)
    ax2.set_ylabel('Time (s)', fontsize=14)
    ax3.set_xlabel("Number of simulations", fontsize=14)

    ax1.annotate('A', xy=(0, 1), xycoords='axes fraction', fontsize=20,
                 xytext=(-60, 10), textcoords='offset points', ha='left',
                 va='top')
    ax2.annotate('B', xy=(0, 1), xycoords='axes fraction', fontsize=20,
                 xytext=(-60, 10), textcoords='offset points', ha='left',
                 va='top')
    ax3.annotate('C', xy=(0, 1), xycoords='axes fraction', fontsize=20,
                 xytext=(-60, 10), textcoords='offset points', ha='left',
                 va='top')

    ax1.annotate('Cell Cycle', xy=(0, 1), xycoords='axes fraction',
                 fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction',
                 fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=20,
                 xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')

    y_lim = [1, 500]
    x_lim = [10, 10000]
    ax1.set_xlim(x_lim)
    ax1.set_ylim(.01, 500)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(os.path.join('Figures', 'supp_figure_3_compare_gpu.png'),
                bbox_tight='True')
    plt.savefig(os.path.join('Figures', 'supp_figure_3_compare_gpu.eps'),
                bbox_tight='True')


def plot_tpb(cupsoda):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    labels = {0: 'global', 1: 'shared', 2: 'shared+constant'}
    for i in [0, 1, 2]:
        cupsoda_tmp = cupsoda[cupsoda['mem'] == i]
        data = cupsoda_tmp[cupsoda_tmp['card'] == 'gtx-760']
        earm = data[data['model'] == 'earm']
        earm = earm[earm['nsims'] == 1000]
        tpb = np.array(earm['tpb'])
        time = np.array(earm['cupsodatime'])
        ax1.plot(tpb, time, 'o-', label=labels[i])
    plt.xlim(2, 64)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(tpb)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(loc=0)

    plt.ylabel('Time(s)')
    plt.xlabel('Threads per block')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig('compare_tpb_test.png', bbox_tight='True')


def create_tpb(mem, cupsoda):
    cupsoda = cupsoda[cupsoda['mem'] == mem]

    data = cupsoda[cupsoda['card'] == 'gtx-760']
    tyson = data[data['model'] == 'tyson']
    tyson = tyson.groupby(('tpb'))
    ras = data[data['model'] == 'ras']
    ras = ras.groupby(('tpb'))
    earm = data[data['model'] == 'earm']
    earm = earm.groupby(('tpb'))
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    markers = {16: 'r>-', 32: 'go-', 64: 'b^-', 4: 'y*-'}

    count = 0
    for i, j in tyson:
        x = np.array(j['nsims'])
        y = np.array(j['cupsodatime'])
        ax1.plot(x, y, markers[i], label=int(i))
        count += 1
    # plt.title('Comparing threads per block')
    plt.xscale('log')
    plt.yscale('log')
    count = 0
    ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
    for i, j in ras:
        x = np.array(j['nsims'])
        y = np.array(j['cupsodatime'])
        ax2.plot(x, y, markers[i], label=int(i))
        count += 1
    plt.xscale('log')
    plt.yscale('log')

    ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
    count = 0
    for i, j in earm:
        x = np.array(j['nsims'])
        y = np.array(j['cupsodatime'])
        ax3.plot(x, y, markers[i], label=int(i))
        count += 1
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Number of simulations')
    plt.ylabel('Time(s)')
    ax1.annotate('Cell Cycle', xy=(0, 1), xycoords='axes fraction',
                 fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax2.annotate('Ras/cAMP/PKA', xy=(0, 1), xycoords='axes fraction',
                 fontsize=20, xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    ax3.annotate('EARM', xy=(0, 1), xycoords='axes fraction', fontsize=20,
                 xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top')
    # plt.legend(loc=0,ncol=2)
    ax1.legend(fontsize=14, bbox_to_anchor=(.55, 1.0), fancybox=True, ncol=1)
    ax3.legend(fontsize=14, loc=0, fancybox=True, ncol=1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    ax3.yaxis.set_tick_params(labelsize=14)
    y_lim = [1, 5000]
    x_lim = [10, 100000]
    ax1.set_xlim(x_lim)
    ax1.set_ylim(.01, 500)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig('compare_tpb_{}.png'.format(mem), bbox_tight='True')


if __name__ == '__main__':
    print_table_1('tyson')
    print_table_1('ras')
    print_table_1('earm')

    table_2_print_stats_on_memory('tyson')
    table_2_print_stats_on_memory('ras')
    table_2_print_stats_on_memory('earm')
    table_3_print_stats_on_memory('tyson')
    table_3_print_stats_on_memory('ras')
    table_3_print_stats_on_memory('earm')

    create_figure_1()
    create_supplement_figure_1()
    create_supplement_figure_2()
    create_supplement_figure_3()
