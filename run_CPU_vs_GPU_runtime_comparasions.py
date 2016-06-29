import os
import time

import numpy as np

import pysb
import pysb.integrate as integrate
from pysb.bng import generate_equations
from pysb.simulator.cupsoda import set_cupsoda_path, CupSodaSolver

#name = sys.argv[1]
name = 'earm'
if name == 'ras':
    from models.ras_amp_pka import model
    tspan = np.linspace(0, 1500, 100)
    simulations = [10, 100, 1000, 10000, 100000]
    vol = 1e-19

elif name == 'tyson':
    from pysb.examples.tyson_oscillator import model
    tspan = np.linspace(0, 100, 100)
    simulations = [10, 100, 1000, 10000, 100000]
    vol = 1e-19

elif name == 'earm':
    from models.earm_lopez_embedded_flat import  model
    tspan = np.linspace(0, 20000, 100)
    simulations = [10, 100, 1000, 10000]
    vol = 1.661e-20

run = 'cupSODA'
multi = False
run ="scipy"

generate_equations(model)

ATOL = 1e-6
RTOL = 1e-6
mxstep = 20000
det = 1


card = 'gpu'
CPU = 'cpu'
GHZ = 'speed'


params_names = [p.name for p in model.parameters_rules()]
init_name = [p[1].name for p in model.initial_conditions]
par_names = []
for parm in params_names:
    if parm in init_name:
        continue
    else:
        par_names.append(parm)
rate_params = model.parameters_rules()
rate_mask = np.array([p in rate_params for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
par_dict = {par_names[i]: i for i in range(len(par_names))}

if run == 'scipy':
    output = "model,nsims,scipytime,rtol,atol,mxsteps,t_end,n_steps,cpu,GHz\n"
    solver = pysb.integrate.Solver(model, tspan, rtol=RTOL, atol=ATOL, integrator='lsoda', )

if run == 'cupSODA':
    set_cupsoda_path("/home/pinojc/git/cupSODA")
    solver = CupSodaSolver(model, tspan, atol=ATOL, rtol=RTOL, verbose=False)
    output = "model,nsims,tpb,mem,cupsodatime,pythontime,rtol,atol,mxsteps,t_end,n_steps,deterministic,vol,card\n"


def main(number_particles):
    num_particles = int(number_particles)
    c_matrix = np.zeros((num_particles, len(nominal_values)))
    c_matrix[:, :] = nominal_values
    global output
    if run == "cupSODA":
        mx_0 = np.zeros((num_particles, len(model.species)))
        for i in xrange(len(model.initial_conditions)):
            for j in xrange(len(model.species)):
                if str(model.initial_conditions[i][0]) == str(model.species[j]):
                    x = model.initial_conditions[i][1]
                    mx_0[:, j] = x.value
                    break
        mem = 2
        i = 16
        start_time = time.time()
        solver.run(c_matrix,
                   mx_0,
                   outdir=os.path.join('/tmp/ramdisk', 'CUPSODA_%s') % model.name,
                   gpu=0,
                   max_steps=mxstep,
                   vol=vol,
                   obs_species_only=False,
                   memory_usage=mem)
        end_time = time.time()
        new_line = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (name,
                                                                    num_particles,
                                                                    str(i),
                                                                    mem,
                                                                    solver._cupsoda_time,
                                                                    end_time - start_time,
                                                                    RTOL,
                                                                    ATOL,
                                                                    len(tspan),
                                                                    mxstep,
                                                                    np.max(tspan),
                                                                    det,
                                                                    vol,
                                                                    card)
        output += new_line
        print(new_line)
        print('out==', solver.yobs[0][0], solver.yobs[0][-1], '==out')
        os.system('rm -r %s' % os.path.join('/tmp/ramdisk', 'CUPSODA_%s') % model.name)

    if run == 'scipy':
        start_time = time.time()
        for i in xrange(number_particles):
            solver.run()
        total_time = time.time() - start_time
        print('sim = %s , time = %s sec' % (number_particles, total_time))
        new_line = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\n' % (name,
                                                         num_particles,
                                                         total_time,
                                                         RTOL,
                                                         ATOL,
                                                         mxstep,
                                                         np.max(tspan),
                                                         len(tspan),
                                                         CPU,
                                                         GHZ,)
        output += new_line



main(10)

#main(10000)
#for j in simulations:
#    main(j)
print(output)


