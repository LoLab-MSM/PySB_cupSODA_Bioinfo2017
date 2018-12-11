import os
import sys
sys.path.append('C:\Users\James Pino\PycharmProjects\pysb')
import numpy as np
import scipy.interpolate
from models.earm_lopez_embedded_flat import model
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.simulator.cupsoda import CupSodaSimulator
from pysb.util import update_param_vals, load_params
from pysb.tools.sensitivity_analysis import InitialsSensitivity
import logging
from pysb.logging import setup_logger
setup_logger(logging.INFO, file_output='earm.log', console_output=True)

tspan = np.linspace(0, 20000, 101)
observable = 'aSmac'


def likelihood(ysim_momp):
    if np.nanmax(ysim_momp) == 0:
        return -1
    else:
        ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
        st, sc, sk = scipy.interpolate.splrep(tspan, ysim_momp_norm)
        try:
            t10 = scipy.interpolate.sproot((st, sc - 0.10, sk))[0]
            t90 = scipy.interpolate.sproot((st, sc - 0.90, sk))[0]
        except IndexError:
            t10 = 0
            t90 = 0
    td = (t10 + t90) / 2
    return td


def run():
    """ Runs EARM sensitivity to initial conditions

    Provided are two parameter sets


    """
    vals = np.linspace(.8, 1.2, 11)
    vol = 1e-19
    directory = 'SensitivityData'

    integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'mxsteps': 20000}
    integrator_opt_scipy = {'rtol': 1e-6, 'atol': 1e-6, 'mxstep': 20000}

    new_params = load_params(os.path.join('Params',
                                          'earm_parameter_set_one.txt'))
    savename = 'local_earm_parameters_1'
    update_param_vals(model, new_params)

    cupsoda_solver = CupSodaSimulator(model, tspan, verbose=False, gpu=0,
                                      memory_usage='sharedconstant',
                                      vol=vol,
                                      integrator_options=integrator_opt)

    scipy_solver = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda',
                                     integrator_options=integrator_opt_scipy)

    sens = InitialsSensitivity(
            # cupsoda_solver,
            scipy_solver,
            values_to_sample=vals,
            observable=observable,
            objective_function=likelihood)

    sens.run(save_name=savename, out_dir=directory)
    # p_matrix = np.loadtxt(
    #         os.path.join('SensitivityData', 'earm_parameters_1_p_matrix.csv'))
    # p_prime_matrix = np.loadtxt(
    #         os.path.join('SensitivityData',
    #                      'earm_parameters_1_p_prime_matrix.csv'))
    # sens.p_matrix = p_matrix
    # sens.p_prime_matrix = p_prime_matrix
    sens.create_boxplot_and_heatplot(x_axis_label='% change in time to death',
                                     save_name='earm_sensitivity_set_1')

    new_params = load_params(os.path.join('Params',
                                          'earm_parameter_set_two.txt'))
    savename = 'local_earm_parameters_2'
    update_param_vals(model, new_params)
    cupsoda_solver = CupSodaSimulator(model, tspan, verbose=False, gpu=0,
                                      memory_usage='sharedconstant', vol=vol,
                                      integrator_options=integrator_opt)
    scipy_solver = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda',
                                     integrator_options=integrator_opt_scipy)

    sens = InitialsSensitivity(cupsoda_solver,
                               values_to_sample=vals,
                               observable=observable,
                               objective_function=likelihood)

    sens.run(save_name=savename, out_dir=directory)
    sens.create_boxplot_and_heatplot(x_axis_label='% change in time to death',
                                     save_name='earm_sensitivity_set_2')

if __name__ == '__main__':
    run()
