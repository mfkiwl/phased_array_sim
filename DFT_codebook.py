import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Antenna parameters
N = 8  # number of elements
d = 0.5  # spacing in wavelengths (assume λ = 1)

# Angle space for plotting
angles = np.linspace(-np.pi/2, np.pi/2, 1000)  # -90 to 90 degrees
u = np.sin(angles)

# Prepare figure
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
line, = ax.plot([], [], lw=2)

ax.set_ylim(0, 1)
ax.set_title("ULA Beam Steering Pattern", va='bottom')
ax.set_yticklabels([])

# Animation function
def animate(i):
    theta_steer = -np.pi/2 + i * np.pi / 180  # from -90° to 90°
    steering_vector = np.exp(1j * 2 * np.pi * d * np.arange(N) * np.sin(theta_steer))

    # Array response over angles
    array_response = np.exp(1j * 2 * np.pi * d * np.outer(np.arange(N), u))
    af = np.abs(steering_vector.conj().T @ array_response)
    af = af / np.max(af)

    line.set_data(angles, af)
    ax.set_title(f"Steering Angle = {np.degrees(theta_steer):.1f}°", va='bottom')
    return line,

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=180, interval=50, blit=True)

plt.show()
