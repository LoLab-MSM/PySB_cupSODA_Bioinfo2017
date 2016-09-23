import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.interpolate
import pysb
import pysb.integrate as integrate
from pysb.bng import generate_equations
from pysb.util import update_param_vals, load_params

ATOL = 1e-12
RTOL = 1e-12
mxstep = 20000


def print_model_stats(model):
    """ provides stats for the models

    :param model:
    """
    generate_equations(model)
    print("Information about model {0}".format(model.name))
    print("Number of rules {0}".format(len(model.rules)))
    print("Number of parameters {0}".format(len(model.parameters)))
    print(
    "Number of parameter rules {0}".format(len(model.parameters_rules())))
    print("Number of reactions {0}".format(len(model.reactions)))
    print(
    "Number of initial conditions {0}".format(len(model.initial_conditions)))
    print("Number of species {0}".format(len(model.species)))
    print('{}'.format('-' * 24))


def create_supplement_figure_4():
    """ plots cell cycle observable """

    from models.tyson_oscillator_in_situ import model
    time = np.linspace(0, 300, 1000)
    observable_name = 'M'
    save_name = 'supp_figure_4_tyson_output'

    solver = pysb.integrate.Solver(model, time, rtol=RTOL, atol=ATOL,
                                   integrator='lsoda', mxstep=mxstep)
    solver.run()
    out = solver.yobs[observable_name]
    out /= model.parameters['cyc0'].value
    timestep = time[:-1]
    y = out[:-1] - out[1:]
    times = []
    prev = y[0]
    for n in range(1, len(y)):
        if y[n] > 0 > prev:
            times.append(timestep[n])
        prev = y[n]

    plt.figure()
    plt.plot(time, out)
    plt.xlabel('Time (min)', fontsize=16)
    plt.ylabel('active MPF / total cdc2', fontsize=16)
    x1 = np.where(time == times[0])
    x2 = np.where(time == times[1])
    x = [times[0], times[1]]
    yy = [out[x1], out[x2]]
    plt.axvline(x[0], linestyle='dashed', color='black')
    plt.axvline(x[1], linestyle='dashed', color='black')
    arrow_x_0 = x[1] - x[0]
    y_distance = yy[1]
    plt.arrow(arrow_x_0, y_distance, x[1] - arrow_x_0 - 5, 0, head_width=.05,
              head_length=3, color='k')
    plt.arrow(arrow_x_0, y_distance, x[0] - arrow_x_0 + 5, 0, head_width=.05,
              head_length=3, color='k')
    plt.xlim(0, 60)
    plt.ylim(0, 1.0)
    plt.savefig('%s/%s_%s.png' % ('Figures', save_name, observable_name),
                dpi=150)
    plt.savefig('%s/%s_%s.eps' % ('Figures', save_name, observable_name))
    plt.close()
    print_model_stats(model)


def create_supplement_figure_6():
    from models.ras_camp_pka import model
    time = np.linspace(0, 1500, 500)
    observable_name = 'obs_cAMP'
    save_name = 'supp_figure_6_ras_output'

    solver = pysb.integrate.Solver(model, time, rtol=RTOL, atol=ATOL,
                                   integrator='lsoda', mxstep=mxstep)
    solver.run()
    out = solver.yobs[observable_name]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, solver.yobs[observable_name], linewidth=2, )
    max_y = np.where(out == out.max())
    ax.plot(time[max_y], out.max(), 'o', color='red', markersize=14, mew=3,
            mfc='none', alpha=.75)
    plt.axhline(out.max(), linestyle='dashed', color='black')
    plt.ylabel('cAMP (count) ', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.ylim(0, 135000)
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig('%s/%s_%s.png' % ('Figures', save_name, observable_name),
                dpi=150)
    plt.savefig('%s/%s_%s.eps' % ('Figures', save_name, observable_name))
    plt.close()
    print_model_stats(model)


def create_supplement_figure_8():
    """ plots EARM observable """
    from models.earm_lopez_embedded_flat import model
    time = np.linspace(0, 20000, 101)
    observable_name = 'aSmac'
    new_params = load_params('Params/earm_parameter_set_one.txt')
    save_name = 'supp_figure_8_earm_output_parameter_set_one'
    update_param_vals(model, new_params)
    solver = pysb.integrate.Solver(model, time, rtol=RTOL, atol=ATOL,
                                   integrator='lsoda', mxstep=mxstep)
    solver.run()
    ysim_momp_norm = solver.yobs[observable_name] / np.nanmax(
            solver.yobs[observable_name])
    st, sc, sk = scipy.interpolate.splrep(time, ysim_momp_norm)
    try:
        t10 = scipy.interpolate.sproot((st, sc - 0.10, sk))[0]
        t90 = scipy.interpolate.sproot((st, sc - 0.90, sk))[0]
    except IndexError:
        t10 = 0
        t90 = 0
    td = (t10 + t90) / 2
    plt.figure()
    plt.plot(time / 3600, ysim_momp_norm, 'b-', linewidth=2)
    plt.xlabel("Time (hr)", fontsize=16)
    plt.ylabel('aSmac / SMAC_0', fontsize=16)
    plt.plot(td / 3600, .5, 'ok', ms=15, mfc='none', mew=3)
    plt.axvline(td / 3600, linestyle='dashed', color='black')
    plt.ylim(-.05, 1.05)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('%s/%s_%s.png' % ('Figures', save_name, observable_name),
                dpi=150)
    plt.savefig('%s/%s_%s.eps' % ('Figures', save_name, observable_name))
    plt.close()
    print_model_stats(model)


if __name__ == '__main__':
    create_supplement_figure_4()
    create_supplement_figure_6()
    create_supplement_figure_8()
