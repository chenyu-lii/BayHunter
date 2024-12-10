# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
import quick_routine   # fictive routine
#from quick_routine import quick_routine_gpell
from quick_routine import HVSRForwardModels
from scipy.interpolate import interp1d
import os
import time

class MyForwardModel(object):
    """
    """
    def __init__(self, obsx, ref):
        self.ref = ref
        self.obsx = obsx

        self.modelparams = {}  # Initialize as an empty dictionary

        # default parameters necessary for forward modeling
        # the dictionary can be updated by the user
        self.modelparams.update(
            {'test': 5,
             })

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def compute_data(self, h, vp, vs, rho, **params):
        """
        Method to compute the synthetic data. Here you probably need to
        include your quick e.g. fortran written code.
        """
        test = self.modelparams['test']

        #z = np.cumsum(h)
        #z = np.concatenate(([0], z[:-1]))

        thickness = np.array(h)*1000
        vp_new = np.array(vp)*1000
        vs_new = np.array(vs)*1000
        rho_new = np.array(rho)*1000
        
        print('vp:',vp_new)
        print('vs:',vs_new)
        print('h:',thickness)
        #print('rho:',rho_new)


        L = len(vp)
        # call HVSRForwardModels
        freq1 = 0.1
        freq2 = 50
        #Ds = np.array( [0.05, 0.01, 0.01,0.01])
        #Dp = np.array( [0.05, 0.01, 0.01,0.01])
        Ds = np.ones(vs_new.size)*0.01
        Dp = np.ones(vp_new.size)*0.01
        Ds[0] = 0.05
        Dp[0] = 0.05
        hvsr_freq = self.obsx
        mod1 = HVSRForwardModels(ro=rho_new,Vs=vs_new,Vp=vp_new,fre1=freq1,fre2=freq2, f=hvsr_freq, Ds=Ds,Dp=Dp,h=thickness)
        xmod, ymod = mod1.HV()
    
        return xmod, ymod
                    
        #except Exception as e:
        ## Log the error and return default values
        #    print(f"Error during quick_routine execution: {e}")
        #    # set all values of ymod to be -10
        #    xmod = self.obsx
        #    #ymod = np.zeros(xmod.size)
        #    ymod = np.ones(xmod.size)
        #    ymod = ymod*-1
        #    return xmod, ymod


    def validate(self, xmod, ymod):
        """Some condition that modeled data is valid. """

        #print("Validating sizes:")
        #print('ymod type:', type(ymod))
        #print("ymod.size:", ymod.size)
        #print('obsx type', type(self.obsx))
        #print("self.obsx.size:", self.obsx.size)
    
        if ymod.size == self.obsx.size:
            # xmod == xobs !!!
            return xmod, ymod
        else:
            return np.nan, np.nan


    def run_model(self, h, vp, vs, rho, **params):
        # incoming model is float32
        # xmod, ymod = self.compute_rf(h, vp, vs, rho, **params)
        xmod, ymod = self.compute_data(h, vp, vs, rho, **params)
        #print('ymod type:', type(ymod))
        #print("ymod.size:", ymod.size)

        return self.validate(xmod, ymod)
