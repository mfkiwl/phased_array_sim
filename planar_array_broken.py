import utils.antenna_utils as au
import numpy as np
import matplotlib.pyplot as plt
import utils.plotting_utils as pu
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Antenna array size
M, N = 8, 8 # M is beamforming in the x-direction, N in the y-direction
n_bits = 4
n_stuck = 4  # Number of bits stuck (0 or 1)
# Angular domain in spherical coordinates
theta = np.linspace(-np.pi/2, np.pi/2, 181)        # Elevation (0 to pi)
phi = np.linspace(-np.pi/2, np.pi/2, 181)      # Azimuth (0 to 2pi)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# select broken bits
broken_elements, broken_bits, broken_values = au.random_select_broken_bits(M, n_bits,n_stuck, mode=1)

# --- Animation setup ---
steering_angles = np.linspace(-np.pi/2, np.pi/2, 61)  # Steering angles from -60° to +60° in radians
n_frames = len(steering_angles)  # Number of frames in the animation

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
pu.planar_3D_setup(ax)
title = fig.suptitle("", fontsize=14)

plt.tight_layout()

# initialise plot surface
surface_ideal = ax.plot_surface(np.zeros_like(theta_grid),
                          np.zeros_like(phi_grid),
                          np.zeros_like(theta_grid),
                          facecolors=plt.cm.viridis(np.zeros_like(theta_grid)),
                          rstride=1, cstride=1, alpha=0.9, linewidth=0)
surface_broke = ax.plot_surface(np.zeros_like(theta_grid),
                                np.zeros_like(phi_grid),
                                np.zeros_like(theta_grid),
                                facecolors=plt.cm.viridis(np.zeros_like(theta_grid)),
                                rstride=1, cstride=1, alpha=0.9, linewidth=0)
surfs = [surface_ideal, surface_broke]
#steers = [-60, -50, -40, -30, -20, -10, 0, 30, 60]  # Steering angles in degrees

def animate(i):
    for surf in surfs:
        surf.remove()
    print(f"Animating Frame {i+1}/{n_frames}")  # Print current frame number

    steering_angle_rad = steering_angles[i]  # Get the current steering angle in radians
    # Calculate all the array factors
    ideal_phase_list = au.ideal_phase_list(M, steering_angle_rad)
    ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
    broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements, broken_bits, broken_values)
    broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
    F_broken = au.AF_planar_linear_beamform(M, N, theta_grid, phi_grid, broken_phase_list)
    # Convert to Cartesian for 3D plotting
    X_broken = F_broken * np.sin(theta_grid) * np.cos(phi_grid)
    Y_broken = F_broken * np.sin(theta_grid) * np.sin(phi_grid)
    Z_broken = F_broken * np.cos(theta_grid)

    F = au.AF_planar_linear_beamform(M, N, theta_grid, phi_grid, ideal_phase_list)
    # Convert to Cartesian for 3D plotting
    X = F * np.sin(theta_grid) * np.cos(phi_grid)
    Y = F * np.sin(theta_grid) * np.sin(phi_grid)
    Z = F * np.cos(theta_grid)
    # Update the surface plot
    surfs[0] = ax.plot_surface(X, Y, Z,
                              facecolors=plt.cm.viridis(F),
                              rstride=1, cstride=1, alpha=0.3, linewidth=0)
    surfs[1] = ax.plot_surface(X_broken, Y_broken, Z_broken,
                              facecolors=plt.cm.viridis(F_broken),
                              rstride=1, cstride=1, alpha=0.9, linewidth=0)
    
    title.set_text(f'Array Factor of 8x8 Uniform Planar Array\nwith {n_stuck} Broken Bits\nSteering Angle: {np.degrees(steering_angle_rad):.2f}°')

    return surfs


ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate,
    frames=n_frames, interval=100, blit=True
)
# To save as a gif: 
ani.save(f"visuals\\sweep\\planar_beamforming_MSB.gif", writer='pillow')
#plt.show()

print("Animation complete. Saved as 'planar_beamforming_MSB.gif'.")