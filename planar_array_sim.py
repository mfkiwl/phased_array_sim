import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils.antenna_utils as au
import utils.plotting_utils as pu
import matplotlib.animation as animation

# Antenna array size
M, N = 8, 8 # M is beamforming in the x-direction, N in the y-direction

# Angular domain in spherical coordinates
theta = np.linspace(-np.pi/2, np.pi/2, 361)        # Elevation (0 to pi)
phi = np.linspace(-np.pi/2, np.pi/2, 361)      # Azimuth (0 to 2pi)
theta_grid, phi_grid = np.meshgrid(theta, phi)

ideal_phase_list = au.ideal_phase_list(M, np.pi/6)
F = au.AF_planar_linear_beamform(M, N, theta_grid, phi_grid, ideal_phase_list)
# Convert to Cartesian for 3D plotting
X = F * np.sin(theta_grid) * np.cos(phi_grid)
Y = F * np.sin(theta_grid) * np.sin(phi_grid)
Z = F * np.cos(theta_grid)


theta_grid /= np.pi/2  # Normalize theta to [0, 1] for plotting
phi_grid /= np.pi/2    # Normalize phi to [0, 1] for plotting

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta_grid, phi_grid, F, facecolors=plt.cm.viridis(F), rstride=1, cstride=1, alpha=0.9, linewidth=0)
ax.set_title('3D Radiation Pattern of 8x8 Uniform Planar Array')
pu.planar_3D_setup(ax)
plt.tight_layout()
plt.show()
