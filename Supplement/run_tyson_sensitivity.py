import os
import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pysb
from pysb.bng import generate_equations
from pysb.integrate import odesolve
from pysb_cupsoda import set_cupsoda_path, CupsodaSolver


from pysb.examples.tyson_oscillator import model
tspan = np.linspace(0, 200, 1001)
observable = 'Y3'
savename = 'tyson'

set_cupsoda_path("/home/pinojc/git/cupSODA")
plot = True

ATOL = 1e-6
RTOL = 1e-6
mxstep = 20000
det = 1
vol = 0

generate_equations(model)
proteins_of_interest = []
for i in model.initial_conditions:
    proteins_of_interest.append(i[1].name)
proteins_of_interest.remove('__source_0')
nominal_values = np.array([p.value for p in model.parameters])



def likelihood_one():
    solver = pysb.integrate.Solver(model, tspan, rtol=RTOL, atol=ATOL,
                                   integrator='lsoda', mxstep=mxstep)
    solver.run()
    out = solver.yobs[observable]

    timestep = solver.tspan[:-1]
    y = out[:-1] - out[1:]
    times = []
    prev = y[0]
    for n in range(1, len(y)):
        if y[n] > 0 > prev:
            times.append(timestep[n])
        prev = y[n]
    if plot:
        plt.figure()
        plt.plot(solver.tspan, out)
        plt.xlabel('Time (min)', fontsize=16)
        plt.ylabel('cdc-U:cyclin-P (count)', fontsize=16)
        x1 = np.where(tspan == times[0])
        x2 = np.where(tspan == times[1])
        x = [times[0],times[1]]
        yy = [out[x1],out[x2]]
        plt.axvline(x[0],linestyle='dashed',color='black')
        plt.axvline(x[1], linestyle='dashed',color='black')
        arrow_x_0 = x[1]-x[0]
        y_distance = yy[1]
        plt.arrow(arrow_x_0,y_distance,  x[1]-arrow_x_0-5, 0, head_width=55, head_length=3,color='k')
        plt.arrow(arrow_x_0,y_distance, x[0]-arrow_x_0+5, 0, head_width=55, head_length=3,color='k')
        plt.xlim(0, 60)
        plt.ylim(0,1400)
        plt.savefig(observable + '.png')
        plt.savefig(observable + '.eps')
        plt.close()
    plt.plot(timestep, y)
    plt.savefig('slope.png')
    times = np.array(times)
    period = np.average(times) / len(times) * 2

    return period


objective_value = likelihood_one()
print objective_value

#quit()
def likelihood(ysim_momp):
    out = ysim_momp
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

    return (objective_value - local_freq) / objective_value * 100.



values_to_sample = np.linspace(.8, 1.2, 21)
n_sam = len(values_to_sample)
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
        for a, c in enumerate(values_to_sample):
            for b, d in enumerate(values_to_sample):
                x = index_of_species_of_interest[i]
                y = index_of_species_of_interest[j]
                MX_0[counter, x] *= c
                MX_0[counter, y] *= d
                counter += 1
    done.append(i)

print("Number of simulations to run = %s" % counter)

if not os.path.exists('OUT'):
    os.mkdir('OUT')


def main():
    global c_matrix, MX_0
    num_particles = len(MX_0)
    mem = 2
    solver = CupsodaSolver(model, tspan, atol=ATOL, rtol=RTOL, verbose=False)
    solver.run(c_matrix,
               MX_0,
               outdir=os.path.join('.', 'CUPSODA_%s') % model.name,
               gpu=0,
               max_steps=mxstep,
               load_conc_data=False,
               memory_usage=mem,
               vol = 10e-20)
    os.system('rm -r %s' % os.path.join('.', 'CUPSODA_%s') % model.name)
    plt.plot(solver.tspan,solver.yobs[:][observable].T)
    plt.savefig('test.png')
    for threads_per_block in range(num_particles):
        np.savetxt('OUT/test-output_%s.txt' % str(threads_per_block), solver.yobs[observable][threads_per_block])


def load_results():
    data_matrix = np.zeros((len(tspan), size_of_matrix))
    local_counter = 0
    for i in range(len(proteins_of_interest)):
        for j in range(i, len(proteins_of_interest)):
            if i == j:
                continue
            for a in range(len(values_to_sample)):
                for b in range(len(values_to_sample)):
                    tmp = np.loadtxt('OUT/test-output_%s.txt' % str(local_counter))
                    data_matrix[:, local_counter] = tmp
                    local_counter += 1
    return data_matrix

main()
camp = load_results()
image = np.zeros((len(proteins_of_interest) * len(values_to_sample), len(proteins_of_interest) * len(values_to_sample)))
counter = 0
for i in range(len(proteins_of_interest)):
    y = i * len(values_to_sample)
    for j in range(i, len(proteins_of_interest)):
        x = j * len(values_to_sample)
        if x == y:
            continue
        for a in range(len(values_to_sample)):
            for b in range(len(values_to_sample)):
                tmp = likelihood(camp[:, counter])
                image[y + a, x + b] = tmp
                image[x + b, y + a] = tmp
                counter += 1
print("Number of simulations to ran = %s" % counter)

np.savetxt('sens_%s_matrix.csv' % savename, image)
