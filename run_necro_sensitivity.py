import os
import sys
sys.path.append('C:\Users\James Pino\PycharmProjects\pysb')
import numpy as np
import scipy.interpolate
from models.necro import model
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.simulator.cupsoda import CupSodaSimulator
from pysb.util import update_param_vals, load_params
from pysb.tools.sensitivity_analysis import InitialsSensitivity
import logging
from pysb.logging import setup_logger
setup_logger(logging.INFO, file_output='necro.log', console_output=True)

tspan = np.linspace(0, 480, 481)
observable = 'MLKLa_obs'

#make an array for each of the kd made up data for mlklp
#switching at 5 hours
wtx = np.array([0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., 11.,  12.])
wty = np.array([0., 0., 0., 0.10, 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.,1.])

#A20 data switching at 3 hours
a20x = np.array([0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., 11.,  12.])
a20y = np.array([0.,0., 0.2, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1., 1.])

#Tradd data switching at 5
tdx = np.array([0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., 11.,  12.])
tdy = np.array([0., 0., 0., 0.10, 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.,1.])

#Fadd Data switching at 4 hours
fdx = np.array([0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., 11.,  12.])
fdy = np.array([0., 0., 0.10,0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1.])

#C8 Data switching at 4
c8x = np.array([0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., 11.,  12.])
c8y = np.array([0., 0., 0.10,0.25, 0.5, 0.75, 1., 1., 1., 1., 1., 1.,1.])

#data = collections.OrderedDict([('wt', wty), ('a20', a20y), ('td', tdy),('fd', fdy), ('c8', c8y)])
# data = collections.OrderedDict([('wt', wty), ('fd', fdy)])
# data = collections.OrderedDict()
# data = {'wt': wty, 'a20': a20y, 'td': tdy, 'fd': fdy, 'c8': c8y}
# data = collections.OrderedDict(sorted(data.items(), key = lambda t:t[1]))


ydata_norm = wty

rate_params = model.parameters_rules()
# print(len(rate_params))
param_values = np.array([p.value for p in model.parameters])
# print(len(param_values))
rate_mask = np.array([p in rate_params for p in model.parameters])
# print(len(rate_mask))
# quit()

original_values = np.array([p.value for p in model.parameters])

# We search in log10 space for the parameters
log10_original_values = np.log10(original_values[rate_mask])

def likelihood(params):
    # print('obj function')
    # Y = np.copy(parameter_2)
    # param_values[rate_mask] = 10 ** Y
    params_tmp = np.copy(params)
    # rate_params = 10 ** params_tmp #don't need to change
    param_values[rate_mask] = 10 ** params_tmp  # don't need to change
    # # print(len(param_values[rate_mask]))
    # # quit()
    # #make a new parameter value set for each of the KD
    # a20_params = np.copy(param_values)
    # a20_params[6] = 2700
    # tradd_params = np.copy(param_values)
    # tradd_params[2] = 2700
    # fadd_params = np.copy(param_values)
    # fadd_params[8] = 2424
    # c8_params = np.copy(param_values)
    # c8_params[11] = 2700
    ko_pars = [param_values]

    result = solver1.run(param_values=ko_pars)
    # solver2.run(param_values=a20_params)
    # solver3.run(param_values=tradd_params)
    # solver4.run(param_values=fadd_params)
    # solver5.run(param_values=c8_params)

    # list = [y1, y2, y3, y4, y5]
    # for i in list:
    ysim_array1 = result.observables[0]['MLKLa_obs']

    # ysim_array = extract_records(solver.yobs, obs_names)
    ysim_norm1 = normalize(ysim_array1)

    # mlkl_var = np.var(y)
    mlkl_wt = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    e1 = np.sum((ydata_norm - ysim_norm1) ** 2 / (mlkl_wt))

    return e1,



def run():
    """ Runs EARM sensitivity to initial conditions

    Provided are two parameter sets

    """
    vals = np.linspace(.8, 1.2, 11)
    vol = 1e-19
    directory = 'SensitivityData'

    integrator_opt = {'rtol': 1e-6, 'atol': 1e-6, 'max_steps': 20000, 'memory_usage':'global','vol':vol}
    integrator_opt_scipy = {'rtol': 1e-6, 'atol': 1e-6, 'mxstep': 20000}

    new_params1 = load_params(os.path.join('Params',
                                          'necro_uncal_param.txt'))
    savename = 'local_necro_parameters_1'
    update_param_vals(model, new_params1)

    cupsoda_solver = CupSodaSimulator(model, tspan, verbose=False, gpu=0, integrator_options=integrator_opt)

    scipy_solver = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda',
                                     integrator_options=integrator_opt_scipy)

    sens = InitialsSensitivity(
            cupsoda_solver,
            #scipy_solver,
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
    sens.create_boxplot_and_heatplot(save_name='necro_sensitivity_set_1')

    new_params2 = load_params(os.path.join('Params',
                                          'necro_cal_param.txt'))
    savename = 'local_necro_parameters_2'
    update_param_vals(model, new_params2)
    cupsoda_solver = CupSodaSimulator(model, tspan, verbose=False, gpu=0,
                                      #memory_usage='sharedconstant', vol=vol,
                                      integrator_options=integrator_opt)
    scipy_solver = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda',
                                     integrator_options=integrator_opt_scipy)

    sens = InitialsSensitivity(cupsoda_solver,
                               values_to_sample=vals,
                               observable=observable,
                               objective_function=likelihood)

    sens.run(save_name=savename, out_dir=directory)
    sens.create_boxplot_and_heatplot(save_name='necro_sensitivity_set_2')

if __name__ == '__main__':
    run()
