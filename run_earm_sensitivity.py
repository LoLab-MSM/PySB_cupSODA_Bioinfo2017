import os
import time
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import scipy.interpolate
import pysb
from models.earm_lopez_embedded_flat import  model
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb.util import update_param_vals, load_params
from pysb_cupsoda import set_cupsoda_path, CupsodaSolver

tspan = np.linspace(0, 20000, 100)
option = '1'
observable = 'cPARP'


if option == '1':
    new_params = load_params('Params/earm_parameter_set_1.txt')
    savename = 'parameters_1_gpu_new'
    directory = 'OUT'

if option == '2':
    new_params = load_params('Params/earm_parameter_set_2.txt')
    savename = 'parameters_2_gpu_new'
    directory = 'OUT'


update_param_vals(model, new_params)

set_cupsoda_path("/home/pinojc/git/cupSODA")
run = 'cupSODA'
run = 'scipy'

ATOL = 1e-6
RTOL = 1e-6
mxstep = 20000
det = 1
vol = 10e-20

generate_equations(model)

proteins_of_interest = []
for i in model.initial_conditions:
   proteins_of_interest.append(i[1].name)


nominal_values = np.array([p.value for p in model.parameters])
vals = np.linspace(.8, 1.2, 11)
n_sam = len(vals)
n_proteins = len(proteins_of_interest)

size_of_matrix = (n_proteins * n_proteins - n_proteins) * (n_sam * n_sam )/ 2

c_matrix = np.zeros((size_of_matrix, len(nominal_values)))
c_matrix[:,:] = nominal_values
index_of_species_of_interest = {}
MX_0 = np.zeros((size_of_matrix, len(model.species)))
for i in xrange(len(model.initial_conditions)):
    for j in xrange(len(model.species)):
        if str(model.initial_conditions[i][0]) == str(model.species[j]):
            x = model.initial_conditions[i][1].value
            MX_0[:, j] = x
            if model.initial_conditions[i][1].name in proteins_of_interest:
                index_of_species_of_interest[model.initial_conditions[i][1].name] = j

counter = 0
done = []
for i in proteins_of_interest:
    for j in proteins_of_interest:
        if j in done:
            continue
        if i == j:
            continue
        for a, c in enumerate(vals):
            for b, d in enumerate(vals):
                x = index_of_species_of_interest[i]
                y = index_of_species_of_interest[j]
                MX_0[counter, x] *= c
                MX_0[counter, y] *= d
                counter += 1
    done.append(i)

print("Number of simulations to run = %s" % counter)


def likelihood1():
    solver = pysb.integrate.Solver(model, tspan, rtol=RTOL, atol=ATOL,
                                   integrator='lsoda', mxstep=mxstep)
    solver.run()
    ysim_momp_norm = solver.yobs[observable] / np.nanmax(solver.yobs['cPARP'])
    st, sc, sk = scipy.interpolate.splrep(tspan, ysim_momp_norm)
    try:
        t10 = scipy.interpolate.sproot((st, sc - 0.10, sk))[0]
        t90 = scipy.interpolate.sproot((st, sc - 0.90, sk))[0]
    except IndexError:
        t10 = 0
        t90 = 0
    td = (t10 + t90) / 2

    return td


tod = likelihood1()

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
    return (td - tod) / tod * 100.


def main():
    if run == "cupSODA":
        num_particles = len(MX_0)
        mem = 2
        threads_per_block = 16
        solver = CupsodaSolver(model, tspan, atol=ATOL, rtol=RTOL, verbose=False)
        start_time = time.time()
        solver.run(c_matrix,
                   MX_0,
                   n_blocks=np.int(num_particles / threads_per_block),
                   outdir=os.path.join('.', 'CUPSODA_%s') % model.name,
                   gpu=2,
                   max_steps=mxstep,
                   obs_species_only=True,
                   memory_usage=mem,
                   vol=vol)
        time_taken = time.time() - start_time
        print('sim = %s , time = %s sec' % (size_of_matrix, time_taken))
        print('out==', solver.yobs[0][0], solver.yobs[0][-1], '==out')
        for n in range(num_particles):
            np.savetxt('OUT/output_%s.txt' % str(n), solver.yobs[n][observable])
    if run == 'scipy':
        solver = pysb.integrate.Solver(model, tspan, rtol=RTOL, atol=ATOL,
                                       integrator='lsoda', mxstep=mxstep)
        start_time = time.time()
        for k in range(size_of_matrix):
            solver.run(y0=MX_0[k, :])
            np.savetxt('%s/output_%s.txt' %(directory, str(k)), solver.yobs[observable])
        time_taken = time.time() - start_time
        print('sim = %s , time = %s sec' % (size_of_matrix, time_taken))

main()


def load_results():
    cparp = np.zeros((len(tspan), size_of_matrix))
    counter = 0
    for i in range(len(proteins_of_interest)):
        for j in range(i, len(proteins_of_interest)):
            if i == j:
                continue
            for a in range(len(vals)):
                for b in range(len(vals)):
                    tmp = np.loadtxt('%s/output_%s.txt' %(directory, str(counter)))
                    cparp[:, counter] = tmp
                    counter += 1
    return cparp

print("Started to load files")
cparp = load_results()
print("Done loading files")
image = np.zeros((len(proteins_of_interest) * len(vals), len(proteins_of_interest) * len(vals)))
counter = 0
for i in range(len(proteins_of_interest)):
    y = i * len(vals)
    for j in range(i, len(proteins_of_interest)):
        x = j * len(vals)
        if x == y:
            continue
        for a in range(len(vals)):
            for b in range(len(vals)):
                tmp = likelihood(cparp[:, counter])
                image[y + a, x + b] = tmp
                image[x + b, y + a] = tmp
                counter += 1
print("Image file created ")
np.savetxt('%s_image_matrix.csv' % savename, image)
