import os
import time

import matplotlib
import numpy as np

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pysb
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb_cupsoda import set_cupsoda_path, CupsodaSolver



from models.ras_amp_pka import model
tspan = np.linspace(0, 1000, 301)
observable = 'obs_cAMP'
savename = 'ras'


set_cupsoda_path("/home/pinojc/git/cupSODA")
#run = 'cupSODA'
run = 'scipy'

ATOL = 1e-6
RTOL = 1e-6
mxstep = 20000
det = 1
vol = 10e-20
card = 'K20c'
# puma
CPU = 'Intel-Xeon-E5-2687W-v2'
GHZ = '3.40'

generate_equations(model)
proteins_of_interest = []
for i in model.initial_conditions:
    proteins_of_interest.append(i[1].name)

nominal_values = np.array([p.value for p in model.parameters])

vals = np.linspace(.8, 1.2, 21)
n_sam = len(vals)
n_proteins = len(proteins_of_interest)
size_of_matrix = (n_proteins * n_proteins - n_proteins) * (n_sam * n_sam) / 2
c_matrix = np.zeros((size_of_matrix, len(nominal_values)))
c_matrix[:, :] = nominal_values
MX_0 = np.zeros((size_of_matrix, len(model.species)))
index_of_species_of_interest = {}

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


def likelihood_one():
    solver = pysb.integrate.Solver(model, tspan, rtol=RTOL, atol=ATOL,
                                   integrator='lsoda', mxstep=mxstep)
    solver.run()
    out = solver.yobs[observable]
    plot = False
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(solver.tspan, solver.yobs[observable], linewidth=2, )
        max_y = np.where(out == out.max())
        ax.plot(solver.tspan[max_y], out.max(), 'o', color='red', markersize=14,mew=3,mfc='none', alpha=.75)
        plt.axhline(out.max(), linestyle='dashed', color='black')
        plt.ylabel('cAMP (count) ',fontsize=16)
        plt.xlabel('Time (s)',fontsize=16)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.ylim(0,125000)
        plt.tight_layout()
        plt.savefig(observable + '.png')
        plt.savefig(observable + '.eps')
        plt.close()
    return out.max()


times = likelihood_one()


def likelihood(ysim_momp):
    out = ysim_momp
    return (out.max() - times) / times * 100.


def main():
    if run == "cupSODA":
        global c_matrix, MX_0
        num_particles = len(MX_0)
        mem = 2
        solver = CupsodaSolver(model, tspan, atol=ATOL, rtol=RTOL, verbose=False)
        start_time = time.time()
        solver.run(c_matrix,
                   MX_0,
                   outdir=os.path.join('.', 'CUPSODA_%s') % model.name,
                   gpu=0,
                   max_steps=mxstep,
                   load_conc_data=False,
                   obs_species_only=True,
                   vol=vol,
                   memory_usage=mem)
        time_taken = time.time() - start_time
        print 'sim = %s , time = %s sec' % (num_particles, time_taken)
        print('out==', solver.yobs[0][0], solver.yobs[0][-1], '==out')
        for threads_per_block in range(num_particles):
            np.savetxt('OUT/test-output_%s.txt' % str(threads_per_block), solver.yobs[observable][threads_per_block])
    if run == 'scipy':
        solver = pysb.integrate.Solver(model, tspan, rtol=RTOL, atol=ATOL,
                                       integrator='lsoda', mxstep=mxstep)
        start_time = time.time()
        for k in range(size_of_matrix):
            solver.run(y0=MX_0[k, :])
            #np.savetxt('OUT/test-output_%s.txt' % str(k), solver.yobs[observable])
        time_taken = time.time() - start_time
        print 'sim = %s , time = %s sec' % (size_of_matrix, time_taken)

main()

print("Started to load files")
def load_results():
    camp = np.zeros((len(tspan), size_of_matrix))
    counter = 0
    for i in range(len(proteins_of_interest)):
        for j in range(i, len(proteins_of_interest)):
            if i == j:
                continue
            for a in range(len(vals)):
                for b in range(len(vals)):
                    tmp = np.loadtxt('OUT/test-output_%s.txt' % str(counter))
                    camp[:, counter] = tmp
                    counter += 1
    return camp
print("Done loading files")

camp = load_results()
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
                tmp = likelihood(camp[:, counter])
                image[y + a, x + b] = tmp
                image[x + b, y + a] = tmp
                counter += 1
print("Image file created ")
np.savetxt('sens_%s_matrix.csv' % savename, image)
