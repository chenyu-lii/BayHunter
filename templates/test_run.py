# Input parameters
h = [10, 20, 30]  # Layer thicknesses
vp = [3.5, 4.0, 4.5]  # P-wave velocities
vs = [2.0, 2.5, 3.0]  # S-wave velocities
rho = [2.4, 2.5, 2.6]  # Densities

# Initialize model
model = MyForwardModel(obsx=[0, 1, 2], ref="test_ref")

# Run the model (includes computation and validation)
xmod, ymod = model.run_model(h, vp, vs, rho)

print("xmod:", xmod)
print("ymod:", ymod)
