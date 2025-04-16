import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import antenna_utils as au

# --- Parameters ---
n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
n_stuck = 5  # Number of bits stuck (0 or 1)

broken_elements, broken_bits, broken_values = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=0)

# --- Animation setup ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
scan_deg = np.arange(-180, 181)  # Scan angles from -90° to +90°
scan_rad = np.radians(scan_deg)
line0 = ax.plot([], [], lw=1, color='k', label="ideal array factor")[0]
line1 = ax.plot([], [], lw=1, color='b', label="quantised phases")[0]
line2 = ax.plot([], [], lw=1, color='r', label="broken bit array")[0]
line_optim = ax.plot([], [], lw=1, color='g', label="optimised bit array")[0]
ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)
max_steering_angle = 90  # Maximum steering angle in degrees
n_frames = 90  # Number of frames in the animation

ax.set_ylim(0, 1)
ax.set_ylabel("Array Factor (linear scale)", labelpad=20)
ax.set_theta_offset(np.pi/2)
ax.set_title("Beam Steering with 4-bit 1x8 Linear Array\n", pad = 20)

def init():
    # animate must take a list
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    line_optim.set_data([], [])
    return [line0, line1, line2, line_optim]

def animate(i):
    step = 2 * max_steering_angle / n_frames  # Step size for steering angle
    # i is the frame number
    steering_angle_deg = int(np.round(-max_steering_angle + i*step))  # Sweep from -60° to +60°
    steering_angle_rad = np.deg2rad(steering_angle_deg)

    ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
    quantised_phase_list = au.quantise_phase_list(ideal_phase_list, n_bits)
    af0 = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
    af1 = au.phase_list_to_af_list(quantised_phase_list, scan_rad)

    ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
    broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements, broken_bits, broken_values)
    broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
    optim_bit_array = au.el_by_el_optim(ideal_bit_array, broken_elements, broken_bits, broken_values)
    optim_phase_list = au.bit_array_to_phase_list(optim_bit_array)

    af2 = au.phase_list_to_af_list(broken_phase_list, scan_rad)
    af_optim = au.phase_list_to_af_list(optim_phase_list, scan_rad)
    line0.set_data(scan_rad, af0)
    line1.set_data(scan_rad, af1)
    line2.set_data(scan_rad, af2)
    line_optim.set_data(scan_rad, af_optim)
    ax.set_title(f"{n_stuck} bits stuck, Steering Angle: {steering_angle_deg:.1f}°", va='bottom', pad=20)
    return [line0, line1, line2, line_optim]

ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate, init_func=init,
    frames=n_frames, interval=100, blit=True
)

# To save as a gif: 
ani.save(f"visuals\\beamforming_{n_stuck}_bits_stuck.gif", writer='pillow')
plt.show()
