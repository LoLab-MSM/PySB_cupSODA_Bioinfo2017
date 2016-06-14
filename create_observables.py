import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.interpolate

import pysb
import pysb.integrate as integrate
from pysb.util import update_param_vals, load_params

ATOL = 1e-6
RTOL = 1e-6
mxstep = 20000

def observable_earm():
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
    plt.plot(solver.tspan / 3600, ysim_momp_norm, 'b-', linewidth=2)
    plt.xlabel("Time (hr)", fontsize=16)
    plt.ylabel('cPARP / PARP_0', fontsize=16)
    plt.plot(td/3600,.5,'ok',ms=15,mfc='none',mew=3)
    plt.axvline(td/3600,linestyle='dashed',color='black')
    plt.ylim(-.05, 1.05)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('%s_%s.png' % (savename,observable),dpi=150)
    plt.savefig('%s_%s.eps' % (savename,observable))
    plt.close()

def observable_ras():
    solver = pysb.integrate.Solver(model, tspan, rtol=RTOL, atol=ATOL,
                                   integrator='lsoda', mxstep=mxstep)
    solver.run()
    out = solver.yobs[observable]
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
    plt.savefig('%s_%s.png' % (savename,observable),dpi=150)
    plt.savefig('%s_%s.eps' % (savename,observable))
    plt.close()

def observable_cell_cycle():
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
    plt.savefig('%s_%s.png' % (savename,observable),dpi=150)
    plt.savefig('%s_%s.eps' % (savename,observable))
    plt.close()


tspan = np.linspace(0, 1500, 100)
observable = 'obs_cAMP'
savename = 'ras'
observable_ras()

tspan = np.linspace(0, 300, 1000)
observable = 'Y3'
savename = 'tyson'
observable_cell_cycle()


from earm.lopez_embedded import model
tspan = np.linspace(0, 20000, 100)
simulations = [10, 100, 1000, 10000]
observable = 'cPARP'

new_params = load_params('Params/earm_parameter_set_2.txt')
savename = 'parameters_911_gpu_new'
directory = 'OUT'
update_param_vals(model, new_params)
observable_earm()

new_params = load_params('Params/earm_parameter_set_1.txt')
savename = 'parameters_486_gpu_new'
directory = 'OUT'
update_param_vals(model, new_params)
observable_earm()