# test run quick routine in  MyForwardModel

#import MyForwardModel
from myfwd import MyForwardModel
import numpy as np

# Input parameters
##h = [10, 20, 30]  # Layer thicknesses
#h = [0.010, 0.020, 0.030]  # Layer thicknesses
##vp = [500, 1000, 2000]  # P-wave velocities
#vp = [5, 10, 20]  # P-wave velocities
##vs = [250, 500, 1000]  # S-wave velocities
#vs = [2.50, 5.00, 10.00]  # S-wave velocities
#rho = [2.400, 2.500, 2.600]  # Densities

# 5 layers
h = [0.010, 0.020, 0.030, 0.04, 0.05]  # Layer thicknesses
vp = [5, 10, 15, 20, 22]  # P-wave velocities
vs = [2.50, 5.00, 7.0, 10.00, 11]  # S-wave velocities
rho = [2.400, 2.500, 2.6, 2.700, 2.7]  # Densities


print(vp)
print(vs)

# Initialize model
# generate a numpy array with 50 samples ranging from 0.2 to 20, spaced regularly in logarithmic scale as obsx
x_obs = np.logspace(np.log10(0.2), np.log10(20), num=50)
print('obsx:',x_obs)
model = MyForwardModel(obsx=x_obs, ref="test_ref")

## Run the model (includes computation and validation)
xmod, ymod = model.run_model(h, vp, vs, rho)

print("xmod:", xmod)
print("ymod:", ymod)
