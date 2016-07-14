import time
import numpy as np
from pysb.integrate import Solver
from pysb.simulator.cupsoda import set_cupsoda_path, CupSodaSolver
from pysb.tools.sensitivity_analysis import InitialConcentrationSensitivityAnalysis
from models.tyson_oscillator_in_situ import model

tspan = np.linspace(0, 200, 1001)
vol = 1e-19
observable = 'Y3'


def obj_func_cell_cycle(out):
    timestep = tspan[:-1]
    y = out[:-1] - out[1:]
    freq = 0
    local_times = []
    prev = y[0]
    for n in range(1, len(y)):
        if y[n] > 0 > prev:
            local_times.append(timestep[n])
            freq += 1
        prev = y[n]

    local_times = np.array(local_times)
    local_freq = np.average(local_times)/len(local_times)*2
    return local_freq


def cupsoda_solver(matrix):
    size_of_matrix = len(matrix)
    solver = CupSodaSolver(model, tspan, verbose=False)
    start_time = time.time()
    solver.run(y0=matrix,
               gpu=0,
               max_steps=20000,
               obs_species_only=True,
               memory_usage='sharedconstant',
               vol=vol)
    end_time = time.time()
    print("Time taken {0}".format(end_time-start_time))
    obs = solver.concs_observables(squeeze=False)
    obs = np.array(obs)
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

    vals = np.linspace(.8, 1.2, 21)
    set_cupsoda_path("/home/pinojc/git/cupSODA")
    savename = 'tyson_sensitivity_new'
    directory = 'SensitivityData'
    sens = InitialConcentrationSensitivityAnalysis(model, tspan,
                                                   values_to_sample=vals,
                                                   observable=observable,
                                                   objective_function=obj_func_cell_cycle)

    sens.run(run_solver=cupsoda_solver, save_name=savename, output_directory=directory)

if __name__ == '__main__':
    run()