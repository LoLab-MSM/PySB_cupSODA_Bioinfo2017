import numpy as np
from models.ras_camp_pka import model
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.simulator.cupsoda import CupSodaSimulator
from pysb.tools.sensitivity_analysis import InitialsSensitivity
import logging
from pysb.logging import setup_logger
setup_logger(logging.INFO, file_output='ras.log', console_output=True)


def run():
    def obj_func_ras(out):
        return out.max()
    tspan = np.linspace(0, 1500, 301)

    observable = 'obs_cAMP'

    vals = np.linspace(.8, 1.2, 11)

    integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'mxsteps': 20000}
    integrator_opt_scipy = {'rtol': 1e-6, 'atol': 1e-6, 'mxstep': 20000}
    vol = 1e-19

    cupsoda_solver = CupSodaSimulator(model, tspan, gpu=0,
                                      memory_usage='sharedconstant',
                                      vol=vol,
                                      integrator_options=integrator_opt)

    scipy_solver = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda',
                                     integrator_options=integrator_opt_scipy)

    sens = InitialsSensitivity(
            cupsoda_solver,
            # scipy_solver,
            values_to_sample=vals,
            observable=observable,
            objective_function=obj_func_ras
    )

    directory = 'SensitivityData'
    savename = 'local_ras_sensitivity'

    sens.run(save_name=savename, out_dir=directory)

    sens.create_boxplot_and_heatplot(x_axis_label='',
                                     save_name='ras_model_sensitivity',
                                     show=True)

if __name__ == '__main__':
    run()
