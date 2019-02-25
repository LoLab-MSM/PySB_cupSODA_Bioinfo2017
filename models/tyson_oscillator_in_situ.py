from pysb import Parameter, Monomer, Rule, Observable, Initial, Model
from pysb.macros import synthesize, equilibrate
from scipy.constants import N_A
from pysb.simulator.scipyode import ScipyOdeSimulator
import numpy as np

Model()

VOL = 1e-20
NA_V = N_A*VOL

Parameter('k1', 0.015*NA_V)
Parameter('k2', 0)
Parameter('k3', 200/NA_V)
Parameter('k4', 2*180/NA_V/NA_V)
Parameter('kp4', 0.018)
Parameter('k5', 0)
Parameter('k6', 1.0)
Parameter('k7', 0.6)
Parameter('k8', 1e6) #1e12)
Parameter('k9', 1e3) #1e6)

Monomer('cyclin', ['Y', 'b'], {'Y': ['U','P']})
Monomer('cdc2', ['Y', 'b'], {'Y': ['U','P']})

# Rule 1
synthesize(cyclin(Y='U', b=None), k1)

# Rule 2
#degrade(cyclin(Y='U', b=None), k2)

# Rule 3
Rule('BindingAndPhosphoylation', cyclin(Y='U', b=None) + cdc2(Y='P', b=None) >> cyclin(Y='P', b=1) % cdc2(Y='P', b=1), k3)

# Rule 4
Rule('Activation', cyclin(Y='P', b=1) % cdc2(Y='P', b=1) >> cyclin(Y='P', b=1) % cdc2(Y='U', b=1), kp4)

# Rule 4'
Rule('Autocatalytic', cyclin(Y='P', b=1) % cdc2(Y='P', b=1) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2) >>
cyclin(Y='P', b=1) % cdc2(Y='U', b=1) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2) + cyclin(Y='P', b=2) % cdc2(Y='U', b=2), k4)

# Rule 5
#Rule('Opposed', cyclin(Y='P', b=1) % cdc2(Y='U', b=1) >> cyclin(Y='P', b=1) % cdc2(Y='P', b=1), k5)

# Rule 6
#Rule('Dissociation', cyclin(Y='P', b=1) % cdc2(Y='U', b=1) >> cyclin(Y='P', b=None) + cdc2(Y='U', b=None), k6)
Rule('Dissociation', cyclin(Y='P', b=1) % cdc2(Y='U', b=1) >> cdc2(Y='U', b=None), k6)

# Rule 7
#degrade(cyclin(Y='P', b=None), k7)

# Rules 8 and 9
equilibrate(cdc2(Y='U', b=None), cdc2(Y='P', b=None), [k8, k9])

Observable("YT", cyclin()) # Total Cyclin
Observable("CT", cdc2()) # Total CDC2
Observable("M", cyclin(Y='P', b=1) % cdc2(Y='U', b=1) ) # Active Complex

Observable("Y1", cdc2(Y='U', b=None))
Observable("Y2", cdc2(Y='P', b=None))
Observable("Y3", cdc2(Y='U', b=1) % cyclin(Y='P', b=1))
Observable("Y4", cdc2(Y='P', b=1) % cyclin(Y='P', b=1))
Observable("Y5", cyclin(Y='U', b=None))

#Observable("CYCLIN_P", cyclin(Y='P', b=None))
# [C2] in Tyson

Parameter("cdc0", 1*NA_V)
# Parameter("cdc0", 1.0)
Initial(cdc2(Y='P', b=None), cdc0)

# [Y] in Tyson
Parameter('cyc0', 0.25*NA_V)
# Parameter('cyc0', 0.25)
Initial(cyclin(Y='U', b=None), cyc0)

# tspan = tspan = np.linspace(0, 200, 5001)
# sim = ScipyOdeSimulator(model, tspan=tspan)
# result = sim.run()
#
# # # for p in model.parameters:
# # #     print('{},{:e}'.format(p.name,p.value))
#
# # #
# with open('params_tyson_un.txt', 'w') as f:
#     for p, v in zip(model.parameters, result.param_values[0]):
#         f.write('{},{:e}\n'.format(p.name, v))