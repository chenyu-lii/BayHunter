# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np
# import quick_routine   # fictive routine
from quick_routine import quick_routine_gpell
from scipy.interpolate import interp1d

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

        z = np.cumsum(h)
        z = np.concatenate(([0], z[:-1]))

        ## original code to compute with quick routine
        # xmod, ymod = quick_routine(test, z, vp, vs, rho)

        ### added lines to compute with quick_routine_gpell
        input_file = "input.txt"
        # Write input data to file
        #with open(input_file, "w") as f:
        #    for depth, vp_val, vs_val, rho_val in zip(z, vp, vs, rho):
        #        f.write(f"{depth} {vp_val} {vs_val} {rho_val}\n")
        # Write input data to file, specify the last layer
        print("h:",h)
        #print("vp:",vp)
        print("vs:",vs)
        #n_layer = int(len(h)+1)
        n_layer = int(len(h))
        with open(input_file, "w") as f:
            f.write(f"# First line: number of layers\n")
            f.write(str(n_layer)+"\n")
            f.write(f"# One line per layer:\n")
            f.write(f"# Thickness(m), Vp (m/s), Vs (m/s) and density (kg/m3)\n")
            for thickness, vp_val, vs_val, rho_val in zip(h, vp, vs, rho):
                thickness = thickness*1000
                vp_val = vp_val*1000
                vs_val = vs_val*1000
                rho_val = rho_val*1000
                f.write(f"{thickness} {vp_val} {vs_val} {rho_val}\n")
            f.write(f"# Last line is the half-space, its thickness is ignored but the first column is still mandatory\n")
            if thickness != 0:
                f.write(f"0   6000 3000 2900\n")


        # Specify output file
        output_file = "output.txt"
    
        try:
            # Run quick_routine
            exit_code = quick_routine_gpell(input_file, output_file)

            #if exit_code != 0:
            #    raise RuntimeError("quick_routine failed to execute.")

            # Read the results from output file
            xmod, ymod = [], []
            with open(output_file, "r") as f:
                for line in f:
                    # skip comment lines
                    if (line[0]=="#"):
                        continue
                    x, y = map(float, line.strip().split())  # Assuming two-column output
                    xmod.append(x)
                    ymod.append(y)
                    
            xmod = np.array(xmod)
            ymod = np.array(ymod)

            #print('xmod:',xmod)
            #print('ymod:',ymod)

            # Interpolate ymod to match obsx
            #interpolation_function = interp1d(xmod, ymod, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolation_function = interp1d(xmod, ymod, kind='linear', bounds_error=False, fill_value='extrapolate')
            ymod_interpolated = interpolation_function(self.obsx)
            #print('ymode interpolated:', ymod_interpolated)

            return self.obsx, ymod_interpolated
            #return xmod, ymod

        except Exception as e:
        # Log the error and return default values
            print(f"Error during quick_routine execution: {e}")
            # set all values of ymod to be -10
            xmod = self.obsx
            ymod = np.zeros(xmod.size)
            return xmod, ymod


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
