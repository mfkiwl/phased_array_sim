import utils.antenna_utils as au
import numpy as np
import matplotlib.pyplot as plt
import utils.plotting_utils as pu
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Antenna array size
M, N = 8, 8 # M is beamforming in the x-direction, N in the y-direction
# Angular domain in spherical coordinates
theta = np.linspace(-np.pi/2, np.pi/2, 181)        # Elevation (0 to pi)
phi = np.linspace(-np.pi/2, np.pi/2, 181)      # Azimuth (0 to 2pi)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# --- Animation setup ---
steering_angles = np.linspace(-np.pi/2, np.pi/2, 61)  # Steering angles from -60° to +60° in radians
n_frames = len(steering_angles)  # Number of frames in the animation

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
pu.planar_3D_setup(ax)
title = fig.suptitle("", fontsize=14)

plt.tight_layout()

# initialise plot surface
surface = [ax.plot_surface(np.zeros_like(theta_grid),
                          np.zeros_like(phi_grid),
                          np.zeros_like(theta_grid),
                          facecolors=plt.cm.viridis(np.zeros_like(theta_grid)),
                          rstride=1, cstride=1, alpha=0.9, linewidth=0)]

#steers = [-60, -50, -40, -30, -20, -10, 0, 30, 60]  # Steering angles in degrees

def animate(i):
    surface[0].remove()  # Remove previous surface
    print(f"Animating Frame {i+1}/{n_frames}")  # Print current frame number

    steering_angle_rad = steering_angles[i]  # Get the current steering angle in radians
    # Calculate all the array factors
    ideal_phase_list = au.ideal_phase_list(M, steering_angle_rad)
    F = au.AF_planar_linear_beamform(M, N, theta_grid, phi_grid, ideal_phase_list)
    # Convert to Cartesian for 3D plotting
    X = F * np.sin(theta_grid) * np.cos(phi_grid)
    Y = F * np.sin(theta_grid) * np.sin(phi_grid)
    Z = F * np.cos(theta_grid)
    # Update the surface plot
    surface[0] = ax.plot_surface(X, Y, Z,
                              facecolors=plt.cm.viridis(F),
                              rstride=1, cstride=1, alpha=0.9, linewidth=0)
    
    title.set_text(f'Array Factor of 8x8 Uniform Planar Array\nSteering Angle: {np.degrees(steering_angle_rad):.2f}°')

    return surface


ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate,
    frames=n_frames, interval=100, blit=True
)
# To save as a gif: 
ani.save(f"visuals\\sweep\\planar_beamforming_ideal.gif", writer='pillow')
#plt.show()

print("Animation complete. Saved as 'planar_beamforming_ideal.gif'.")