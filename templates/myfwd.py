# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
# import quick_routine   # fictive routine
import quick_routine_gpell


class MyForwardModel(object):
    """
    """
    def __init__(self, obsx, ref):
        self.ref = ref
        self.obsx = obsx

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

        z = np.cumsum(h)
        z = np.concatenate(([0], z[:-1]))

        ## original code to compute with quick routine
        # xmod, ymod = quick_routine(test, z, vp, vs, rho)

        ### added lines to compute with quick_routine_gpell
        input_file = "input.txt"
        # Write input data to file
        with open(input_file, "w") as f:
            for depth, vp_val, vs_val, rho_val in zip(z, vp, vs, rho):
                f.write(f"{depth} {vp_val} {vs_val} {rho_val}\n")


        # Specify output file
        output_file = "output.txt"
    
        # Run quick_routine
        exit_code = quick_routine(input_file, output_file)

        if exit_code != 0:
            raise RuntimeError("quick_routine failed to execute.")

        # Read the results from output file
        xmod, ymod = [], []
        with open(output_file, "r") as f:
            for line in f:
                x, y = map(float, line.strip().split())  # Assuming two-column output
                xmod.append(x)
                ymod.append(y)
                
                

    def validate(self, xmod, ymod):
        """Some condition that modeled data is valid. """
        if ymod.size == self.obsx.size:
            # xmod == xobs !!!
            return xmod, ymod
        else:
            return np.nan, np.nan

    def run_model(self, h, vp, vs, rho, **params):
        # incoming model is float32
        # xmod, ymod = self.compute_rf(h, vp, vs, rho, **params)
        xmod, ymod = self.compute_data(h, vp, vs, rho, **params)

        return self.validate(xmod, ymod)
