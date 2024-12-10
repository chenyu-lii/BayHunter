# test if quick_routine works

import numpy as np
# import quick_routine   # fictive routine
from quick_routine import quick_routine_gpell

# Input parameters
h = [10, 20, 30]  # Layer thicknesses
vp = [3500, 4000, 4500]  # P-wave velocities
vs = [2000, 2500, 3000]  # S-wave velocities
rho = [2400, 2500, 2600]  # Densities

n_layer = int(len(h)+1)

### added lines to compute with quick_routine_gpell
input_file = "input.txt"
# Write input data to file
#with open(input_file, "w") as f:
#    f.write(f"# First line: number of layers\n")
#    f.write(str(n_layer)+"\n")
#    f.write(f"# One line per layer:\n")
#    f.write(f"# Thickness(m), Vp (m/s), Vs (m/s) and density (kg/m3)\n")
#    for thickness, vp_val, vs_val, rho_val in zip(h, vp, vs, rho):
#        f.write(f"{thickness} {vp_val} {vs_val} {rho_val}\n")
#    f.write(f"# Last line is the half-space, its thickness is ignored but the first column is still mandatory\n")
#    f.write(f"0   2000 1000 2500\n")


# Specify output file
output_file = "output.txt"
    
# Run quick_routine
exit_code = quick_routine_gpell(input_file, output_file)
print('exit_code:',exit_code)

if exit_code != 0:
    raise RuntimeError("quick_routine failed to execute.")
