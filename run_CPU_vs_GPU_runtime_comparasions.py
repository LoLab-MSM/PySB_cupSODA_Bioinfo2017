import time
import numpy as np
from pysb.bng import generate_equations
from pysb.simulator.cupsoda import CupSodaSimulator
from pysb.simulator.scipyode import ScipyOdeSimulator
import pandas as pd
import os
import logging
from pysb.logging import setup_logger

ATOL = 1e-6
RTOL = 1e-6
mxstep = 20000

card = 'gtx-980-ti'
CPU = 'cpu'
GHZ = 'speed'

mem_dict = {
    0: 'global',
    1: 'shared',
    2: 'sharedconstant'
    }


def run_scipy(tspan, model, simulations, name, ):
    integrator_option = dict(rtol=RTOL, atol=ATOL, mxstep=mxstep)
    scipy_solver = ScipyOdeSimulator(model, tspan,
                                     integrator_options=integrator_option,
                                     integrator='lsoda', )

    cols = ['model', 'nsims', 'scipytime', 'rtol', 'atol', 'mxsteps', 't_end',
            'n_steps', 'cpu', 'GHz', ]
    all_output = []
    for num_particles in simulations:
        out_list = []
        start_time = time.time()
        for i in range(num_particles):
            x = scipy_solver.run()
        total_time = time.time() - start_time
        out_list.append(name)
        out_list.append(num_particles)
        out_list.append(total_time)
        out_list.append(RTOL)
        out_list.append(ATOL)
        out_list.append(mxstep)
        out_list.append(np.max(tspan))
        out_list.append(len(tspan))
        out_list.append(CPU)
        out_list.append(GHZ)
        print(" ".join(str(i) for i in out_list))
        all_output.append(out_list)
    df = pd.DataFrame(all_output, columns=cols)
    df.to_csv('{}_scipy_timings.csv'.format(name), index=False)
    return df


def run_cupsoda(tspan, model, simulations, vol, name, mem):
    tpb = 32

    solver = CupSodaSimulator(model, tspan, atol=ATOL, rtol=RTOL, gpu=0,
                              max_steps=mxstep, verbose=False, vol=vol,
                              obs_species_only=True,
                              memory_usage=mem_dict[mem])

    cols = ['model', 'nsims', 'tpb', 'mem', 'cupsodatime', 'cupsoda_io_time',
            'pythontime', 'rtol', 'atol', 'mxsteps', 't_end', 'n_steps', 'vol',
            'card']
    generate_equations(model)
    nominal_values = np.array([p.value for p in model.parameters])
    all_output = []
    for num_particles in simulations:
        # set the number of blocks to make it 16 threads per block
        n_blocks = int(np.ceil(1. * num_particles / tpb))
        solver.n_blocks = n_blocks

        # create a matrix of initial conditions
        c_matrix = np.zeros((num_particles, len(nominal_values)))
        c_matrix[:, :] = nominal_values
        y0 = np.zeros((num_particles, len(model.species)))
        for ic in model.initial_conditions:
            for j in range(len(model.species)):
                if str(ic[0]) == str(model.species[j]):
                    y0[:, j] = ic[1].value
                    break

        # setup a unique log output file to extract timing from
        log_file = 'logfile_{}.log'.format(num_particles)
        # remove it if it already exists (happens if rerunning simulation)
        if os.path.exists(log_file):
            os.remove(log_file)

        setup_logger(logging.INFO, file_output=log_file, console_output=True)
        start_time = time.time()

        # run the simulations
        x = solver.run(param_values=c_matrix, initials=y0)

        end_time = time.time()

        # create an emtpy list to put all data for this run
        out_list = list()
        out_list.append(name)
        out_list.append(num_particles)
        out_list.append(str(tpb))
        out_list.append(mem)
        cupsoda_time = 'error'
        total_time = 'error'
        with open(log_file, 'r') as f:
            for line in f:
                if 'reported time' in line:
                    good_line = line.split(':')[-1].split()[0]
                    cupsoda_time = float(good_line)
                if 'I/O time' in line:
                    good_line = line.split(':')[-1].split()[0]
                    total_time = float(good_line)
            f.close()

        out_list.append(cupsoda_time)
        out_list.append(total_time)
        out_list.append(end_time - start_time)
        out_list.append(RTOL)
        out_list.append(ATOL)
        out_list.append(len(tspan))
        out_list.append(mxstep)
        out_list.append(np.max(tspan))
        out_list.append(vol)
        out_list.append(card)
        all_output.append(out_list)
        print(" ".join(str(i) for i in out_list))
        print('out==')
        print(x.observables[0][0])
        print(x.observables[0][-1])
        print('==out\n')

    df = pd.DataFrame(all_output, columns=cols)
    print(df)
    df.to_csv('{}_cupsoda_timings_{}.csv'.format(name, mem), index=False)
    return df


def run_tyson(run, mem=None):
    from models.tyson_oscillator_in_situ import model
    name = 'tyson'
    tspan = np.linspace(0, 100, 100)
    simulations = [10, 100, 1000, 10000, 100000]
    # simulations = [10, 100, 1000, 10000]
    vol = 1e-19
    if run == 'cupSODA':
        return run_cupsoda(tspan, model, simulations, vol, name, mem)
    else:
        return run_scipy(tspan, model, simulations, name)


def run_ras(run, mem=None):
    from models.ras_camp_pka import model
    name = 'ras'
    tspan = np.linspace(0, 1500, 100)
    simulations = [10, 100, 1000, 10000, 100000]
    vol = 1e-19
    if run == 'cupSODA':
        return run_cupsoda(tspan, model, simulations, vol, name, mem)
    else:
        return run_scipy(tspan, model, simulations, name)


def run_earm(run, mem=None):
    from models.earm_lopez_embedded_flat import model
    name = 'earm'
    tspan = np.linspace(0, 20000, 100)
    simulations = [10, 100, 1000, 10000]
    # simulations = [10000]
    vol = 1.661e-19
    if run == 'cupSODA':
        return run_cupsoda(tspan, model, simulations, vol, name, mem)
    else:
        return run_scipy(tspan, model, simulations, name)


if __name__ == '__main__':
    # Run all the cupsoda timing
    tyson_cupsoda = []
    for i in range(3):
        tyson_cupsoda.append(run_tyson('cupSODA', i))
    df = pd.concat(tyson_cupsoda)
    df.to_csv('all_tyson.csv')

    earm_cupsoda = []
    for i in range(3):
        earm_cupsoda.append(run_earm('cupSODA', i))
    df = pd.concat(earm_cupsoda)
    df.to_csv('all_earm.csv')

    ras_cupsoda = []
    for i in range(3):
        ras_cupsoda.append(run_ras('cupSODA', i))
    df = pd.concat(ras_cupsoda)
    df.to_csv('all_ras.csv')

    # Run all the scipy timing
    run_tyson("scipy")
    run_ras("scipy")
    run_earm("scipy")
