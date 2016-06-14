import time
import numpy as np
from models.ras_amp_pka import model
from pysb.integrate import Solver
from pysb.simulator.cupsoda import set_cupsoda_path, CupSodaSolver
from pysb.tools.sensitivity_analysis import InitialConcentrationSensitivityAnalysis

tspan = np.linspace(0, 1500, 100)
observable = 'obs_cAMP'


def obj_func_ras(out):
    return out.max()

def cupsoda_solver(matrix):
    size_of_matrix = len(matrix)
    solver = CupSodaSolver(model, tspan, verbose=False)
    start_time = time.time()
    solver.run(y0=matrix,
               gpu=0,
               max_steps=20000,
               obs_species_only=True,
               memory_usage='shared',
               vol=10e-19)
    end_time = time.time()
    obs = solver.concs_observables(squeeze=False)
    print("Time taken {0}".format(end_time-start_time))
    print('out==', obs[0][0], obs[0][-1], '==out')
    sensitivity_matrix = np.zeros((len(tspan), size_of_matrix))
    return sensitivity_matrix


def run_solver(matrix):
    size_of_matrix = len(matrix)
    solver = Solver(model, tspan, integrator='lsoda')
    sensitivity_matrix = np.zeros((len(tspan), size_of_matrix))
    start_time = time.time()
    for k in range(size_of_matrix):
        print(k,size_of_matrix)
        solver.run(y0=matrix[k, :])
        sensitivity_matrix[:, k] = solver.yobs[observable]
    end_time = time.time()
    print("Time taken {0}".format(end_time - start_time))
    return sensitivity_matrix


def run():

    savename = 'cAMP'
    vals = np.linspace(.8, 1.2, 21)
    set_cupsoda_path("/home/pinojc/git/cupSODA")
    directory = 'OUT'
    sens = InitialConcentrationSensitivityAnalysis(model, tspan,
                                                   values_to_sample=vals,
                                                   observable=observable,
                                                   objective_function=obj_func_ras)

    sens.run(run_solver=run_solver, save_name=savename, output_directory=directory)

if __name__ == '__main__':
    run()