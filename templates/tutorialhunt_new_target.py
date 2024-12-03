# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import os
# set os.environment variables to ensure that numerical computations
# do not do multiprocessing !! Essential !! Do not change !
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import os.path as op
import matplotlib
matplotlib.use('PDF')

from BayHunter import PlotFromStorage
#from BayHunter import Targets
import Targets
from BayHunter import utils
from BayHunter import MCMC_Optimizer
from BayHunter import ModelMatrix
from BayHunter import SynthObs
import logging


## import my own targets
#from mytarget import MyOwnTarget
#from myfwd import MyForwardModel

#
# console printout formatting
#
formatter = ' %(processName)-12s: %(levelname)-8s |  %(message)s'
logging.basicConfig(format=formatter, level=logging.INFO)
logger = logging.getLogger()


#
# ------------------------------------------------------------  obs SYNTH DATA
#
# Load priors and initparams from config.ini or simply create dictionaries.
initfile = 'config.ini'
priors, initparams = utils.load_params(initfile)

# Load observed data (synthetic test data)
##xsw, _ysw = np.loadtxt('observed/st3_rdispph.dat').T
#xsw, _ysw = np.loadtxt('observed/st2_rdispph.dat').T   # my RF data
##xrf, _yrf = np.loadtxt('observed/st3_prf.dat').T
##xrf, _yrf = np.loadtxt('observed/st1_prf.dat').T     # my RF data
#xrf, _yrf = np.loadtxt('observed/st2_prf.dat').T     # my RF data

# add noise to create observed data
# order of noise values (correlation, amplitude):
# noise = [corr1, sigma1, corr2, sigma2] for 2 targets
#noise = [0.0, 0.012, 0.98, 0.005]
#ysw_err = SynthObs.compute_expnoise(_ysw, corr=noise[0], sigma=noise[1])
#ysw = _ysw + ysw_err
#yrf_err = SynthObs.compute_gaussnoise(_yrf, corr=noise[2], sigma=noise[3])
#yrf = _yrf + yrf_err


#
# -------------------------------------------  get reference model for BayWatch
#
# Create truemodel only if you wish to have reference values in plots
# and BayWatch. You ONLY need to assign the values in truemodel that you
# wish to have visible.
##dep, vs = np.loadtxt('observed/st3_mod.dat', usecols=[0, 2], skiprows=1).T
#dep, vs = np.loadtxt('observed/st2_mod.dat', usecols=[0, 2], skiprows=1).T   # my data
#pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
#pvs = np.repeat(vs, 2)
#
#truenoise = np.concatenate(([noise[0]], [np.std(ysw_err)],   # target 1
#                            [noise[2]], [np.std(yrf_err)]))  # target 2
#
#explike = SynthObs.compute_explike(yobss=[ysw, yrf], ymods=[_ysw, _yrf],
#                                   noise=truenoise, gauss=[False, True],
#                                   rcond=initparams['rcond'])
#truemodel = {'model': (pdep, pvs),
#             'nlays': 3,
#             'noise': truenoise,
#             'explike': explike,
#             }
#
##print truenoise, explike
#print(truenoise, explike)


#
#  -----------------------------------------------------------  DEFINE TARGETS
#
# Only pass x and y observed data to the Targets object which is matching
# the data type. You can chose for SWD any combination of Rayleigh, Love, group
# and phase velocity. Default is the fundamendal mode, but this can be updated.
# For RF chose P or S. You can also use user defined targets or replace the
# forward modeling plugin with your own module.
##target1 = Targets.RayleighDispersionPhase(xsw, ysw, yerr=ysw_err)
#target1 = Targets.RayleighDispersionGroup(xsw, ysw, yerr=ysw_err)  # My data is group velocity
#target2 = Targets.PReceiverFunction(xrf, yrf)
#target2.moddata.plugin.set_modelparams(gauss=1., water=0.01, p=6.4)
# My own target
xhv = np.logspace(np.log10(0.2), np.log10(20), num=50)
yhv = np.logspace(np.log10(0.2), np.log10(20), num=50)
target3 = Targets.MyOwnTarget(xhv, yhv)

# Join the targets. targets must be a list instance with all targets
# you want to use for MCMC Bayesian inversion.
#targets = Targets.JointTarget(targets=[target1, target2])
#targets = Targets.JointTarget(targets=[target2])
#targets = Targets.JointTarget(targets=[target1])
targets = Targets.JointTarget(targets=[target3])

#
#  ---------------------------------------------------  Quick parameter update
#
# "priors" and "initparams" from config.ini are python dictionaries. You could
# also simply define the dictionaries directly in the script, if you don't want
# to use a config.ini file. Or update the dictionaries as follows, e.g. if you
# have station specific values, etc.
# See docs/bayhunter.pdf for explanation of parameters

