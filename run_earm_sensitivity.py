import time
import numpy as np
import scipy.interpolate
from models.earm_lopez_embedded_flat import model
from pysb.integrate import Solver
from pysb.util import update_param_vals, load_params
from pysb.simulator.cupsoda import set_cupsoda_path, CupSodaSolver
from pysb.tools.sensitivity_analysis import InitialConcentrationSensitivityAnalysis

tspan = np.linspace(0, 20000, 100)
option = '2'
observable = 'cPARP'


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


def cupsoda_solver(matrix):
    size_of_matrix = len(matrix)
    solver = CupSodaSolver(model, tspan, verbose=False)
    start_time = time.time()
    solver.run(y0=matrix,
               gpu=0,
               max_steps=20000,
               obs_species_only=True,
               memory_usage='sharedconstant',
               vol=10e-20)
    end_time = time.time()
    print("Time taken {0}".format(end_time-start_time))
    obs = np.array(solver.concs_observables(squeeze=False))
    print('out==', obs[0][0], obs[0][-1], '==out')
    sensitivity_matrix = np.zeros((len(tspan), size_of_matrix))
    for i in range(size_of_matrix):
        sensitivity_matrix[:, i] = obs[observable][i]
    return sensitivity_matrix


def run_solver(matrix):
    size_of_matrix = len(matrix)
    solver = Solver(model, tspan, integrator='lsoda')
    sensitivity_matrix = np.zeros((len(tspan), size_of_matrix))
    start_time = time.time()
    for k in range(size_of_matrix):
        solver.run(y0=matrix[k, :])
        sensitivity_matrix[:, k] = solver.yobs[observable]
    end_time = time.time()
    print("Time taken {0}".format(end_time - start_time))
    return sensitivity_matrix

def run():
    """ Runs EARM sensitivty to initial conditions

    Provided are two parameter sets

    :return:
    """
    vals = np.linspace(.8, 1.2, 11)

    if option == '1':
        new_params = load_params('Params/earm_parameter_set_1.txt')
        savename = 'parameters_1_gpu_new'
        directory = 'OUT'

    if option == '2':
        new_params = load_params('Params/earm_parameter_set_2.txt')
        savename = 'parameters_2_gpu_new'
        directory = 'OUT'

    update_param_vals(model, new_params)
    set_cupsoda_path('/home/pinojc/git/cupSODA')

    sens = InitialConcentrationSensitivityAnalysis(model, tspan,
                                                   values_to_sample=vals,
                                                   observable=observable,
                                                   objective_function=likelihood)

    sens.run(run_solver=cupsoda_solver, save_name=savename, output_directory=directory)

if __name__ == '__main__':
    run()