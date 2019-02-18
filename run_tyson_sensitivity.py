import numpy as np
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.simulator.cupsoda import CupSodaSimulator
from pysb.tools.sensitivity_analysis4 import InitialsSensitivity
from models.tyson_oscillator_in_situ import model
import logging
from pysb.logging import setup_logger
setup_logger(logging.INFO, file_output='tyson_run.log', console_output=True)
import matplotlib
matplotlib.use('Agg')

def run():
    # simulation time
    tspan = np.linspace(0, 200, 5001)
    # volume
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
        local_freq = np.average(local_times) / len(local_times) * 2
        return local_freq

    # values to sample over
    vals = np.linspace(.8, 1.2, 21)

    savename = 'tyson_sensitivity_new'
    directory = 'SensitivityData'
    integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'mxsteps': 20000}
    integrator_opt_scipy = {'rtol': 1e-6, 'atol': 1e-6, 'mxstep': 20000}
    cupsoda_solver = CupSodaSimulator(model, tspan, verbose=False, gpu=0)
                                    # integrator_options=integrator_opt, memory_usage = 'sharedconstant', vol=vol)

    scipy_solver = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda',
                                     integrator_options=integrator_opt_scipy)

    sens = InitialsSensitivity(
            #cupsoda_solver,
            scipy_solver,
            values_to_sample=vals,
            observable=observable,
            objective_function=obj_func_cell_cycle, sens_type='initials')

    sens.run(save_name=savename, out_dir=directory)

    sens.create_boxplot_and_heatplot(save_name='tyson_sens')
    sens.create_individual_pairwise_plots(save_name='tyson_pairwise')
    sens.create_plot_p_h_pprime('tyson_phprime')


if __name__ == '__main__':
    run()
