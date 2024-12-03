# test run quick routine in  MyForwardModel

#import MyForwardModel
from myfwd import MyForwardModel
import numpy as np

# Input parameters
h = [10, 20, 30]  # Layer thicknesses
vp = [500, 1000, 2000]  # P-wave velocities
vs = [250, 500, 1000]  # S-wave velocities
rho = [2400, 2500, 2600]  # Densities


# Initialize model
# generate a numpy array with 50 samples ranging from 0.2 to 20, spaced regularly in logarithmic scale as obsx
x_obs = np.logspace(np.log10(0.2), np.log10(20), num=50)
model = MyForwardModel(obsx=x_obs, ref="test_ref")

# Run the model (includes computation and validation)
xmod, ymod = model.run_model(h, vp, vs, rho)

print("xmod:", xmod)
print("ymod:", ymod)
