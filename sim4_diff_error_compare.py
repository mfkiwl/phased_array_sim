import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import antenna_utils as au
from matplotlib.gridspec import GridSpec

# --- Parameters ---
n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
changing_el = 3  # The element that is changing
n_stuck = 3  # Number of bits stuck (0 or 1)
broken_elements0, broken_bits0, broken_values0 = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=0)
broken_elements1, broken_bits1, broken_values1 = np.random.choice(n_elements, n_stuck, replace=False), [n_bits-1]*n_stuck, np.ones((n_stuck,1))
# --- Animation setup ---
max_steering_angle = 90  # Maximum steering angle in degrees
n_frames = 90  # Number of frames in the animation
step = 2 * max_steering_angle / n_frames  # Step size for steering angle
scan_deg = np.arange(-90, 91)  # Scan angles from -90° to +90°
scan_rad = np.radians(scan_deg)

fig = plt.figure(figsize=(6, 9))
gs = GridSpec(2, 1, height_ratios=[6, 3])  # 6 for ax, 3 for ax2
# polar plot
ax = fig.add_subplot(gs[0], polar=True)
# add the plots
line0 = ax.plot([], [], lw=1, color='k', label="ideal array factor")[0]
line1 = ax.plot([], [], lw=1, color='b', label="random break")[0]
line2 = ax.plot([], [], lw=1, color='r', label="break most sig per el")[0]
text = ax.text(0.05, 0.35, '', transform=ax.transAxes, fontsize=12, color='k')
ax.set_ylim(0, 1)
ax.set_ylabel("Array Factor (linear scale)", labelpad=30)
# ax.set_theta_offset(np.pi/2)
ax.set_title("Beam Steering with 4-bit 1x8 Linear Array\n", pad = 20)
# Add a black horizontal line in the polar plot
r = np.linspace(0,1,100)
ax.plot(np.full_like(r, np.pi/2), r, color='k', lw=1, ls='-')
ax.plot(np.full_like(r, -np.pi/2), r, color='k', lw=1, ls='-')
# add a black dotted line at the steering angle
steer_line = ax.plot([], [], color='k', lw=1, ls='--', label="steering angle")[0]
ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)


# cartesian dB plot
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(-90, 90)
ax2.set_ylim(-50, 0)
ax2.set_xlabel("Angle ($^\\circ$)")
ax2.set_ylabel("Relative Gain (dB)")
ax2.grid()
dB0 = ax2.plot([], [], lw=1, color='k', label="ideal array factor")[0]
dB1 = ax2.plot([], [], lw=1, color='b', label="random break")[0]
dB2 = ax2.plot([], [], lw=1, color='r', label="break most sig per el")[0]
steer_line2 = ax2.plot([], [], color='k', lw=1, ls='--')[0]
dB_list = np.linspace(-50, 0, 100)  # dB scale

def init():
    # animate must take a list
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    text.set_text('')
    steer_line.set_data([], [])

    dB0.set_data([], [])
    dB1.set_data([], [])
    dB2.set_data([], [])
    steer_line2.set_data([], [])
    return [line0, line1, line2, text, steer_line, dB0, dB1, dB2, steer_line2]

def animate(i):
    # i is the frame number
    steering_angle_deg = int(np.round(-max_steering_angle + i*step))  # Sweep from -60° to +60°
    steering_angle_rad = np.deg2rad(steering_angle_deg)

    # Calculate all the array factors
    ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
    quantised_phase_list = au.quantise_phase_list(ideal_phase_list, n_bits)
    af0 = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
    ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
    broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements0, broken_bits0, broken_values0)
    broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
    af1 = au.phase_list_to_af_list(broken_phase_list, scan_rad)

    broken_bit_array1 = au.break_bit_array(ideal_bit_array, broken_elements1, broken_bits1, broken_values1)
    broken_phase_list1 = au.bit_array_to_phase_list(broken_bit_array1)
    af2 = au.phase_list_to_af_list(broken_phase_list1, scan_rad)
    line0.set_data(scan_rad, af0)
    line1.set_data(scan_rad, af1)
    line2.set_data(scan_rad, af2)
    # steering line
    steer_line.set_data(np.full_like(r, steering_angle_rad), r)

    # KL divergence
    kl01 = au.kl_divergence(af0, af1)
    kl02 = au.kl_divergence(af0, af2)
    # annotate the value
    #text.set_text(f"KL_quant: {kl01:.2f}\nKL_broken: {kl02:.2f}\nKL_optim: {kl_optim:.2f}")

    # dB scale
    af0_dB = au.amplitude_to_dB_list(af0)
    af1_dB = au.amplitude_to_dB_list(af1)
    af2_dB = au.amplitude_to_dB_list(af2)
    dB0.set_data(scan_deg, af0_dB)
    dB1.set_data(scan_deg, af1_dB)
    dB2.set_data(scan_deg, af2_dB)
    # steering line
    steer_line2.set_data(np.full_like(dB_list, steering_angle_deg), dB_list)

    # losses at the steering angle
    angle_index = np.where(scan_deg == steering_angle_deg)[0][0]
    loss01 = af0_dB[angle_index] - af1_dB[angle_index]
    loss02 = af0_dB[angle_index] - af2_dB[angle_index]
    # annotate the value
    text.set_text(f"KL_rand: {kl01:.2f}\nKL_most_sig: {kl02:.2f}\n"
                  f"L_rand: {-loss01:.2f} dB\nL_most_sig: {-loss02:.2f} dB")

    # title
    ax.set_title(f"{n_stuck} bits stuck, Steering Angle: {steering_angle_deg:.1f}°", va='bottom', pad=30)
    return [line0, line1, line2, text, steer_line, dB0, dB1, dB2, steer_line2]

ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate, init_func=init,
    frames=n_frames, interval=100, blit=True
)

# To save as a gif: 
ani.save(f"visuals\\bit_config_compare\\{n_stuck}.gif", writer='pillow')
plt.show()
