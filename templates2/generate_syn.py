# test run quick routine in  MyForwardModel
# generate synthetic HVSR curve

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
#h = [0.010, 0.020, 0.030, 0.04, 0.05]  # Layer thicknesses
#vp = [5, 10, 15, 20, 22]  # P-wave velocities
#vs = [2.50, 5.00, 7.0, 10.00, 11]  # S-wave velocities
#rho = [2.400, 2.500, 2.6, 2.700, 2.7]  # Densities

# synthetic velocity model
#h = [0.015, 0.05, 0.1]
#vp = [1, 1.6, 4]
#vs = [0.4, 0.8, 2]
#rho = [2.400, 2.500, 2.600]

# synthetic velocity model 2 layers
h = [0.018, 0]
vp = [0.8, 3]
vs = [0.4, 1.5]
rho = [2.40, 2.40]

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

#outfile = 'st1_hvsr.txt'
outfile = 'st2_hvsr.txt'
out = open(outfile,'w') 
for i in range(0,len(xmod)):
    xi = xmod[i]
    yi = ymod[i]
    out.writelines('%f %f\n'%((xi,yi))) 

out.close()

