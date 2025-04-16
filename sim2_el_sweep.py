import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import antenna_utils as au
from matplotlib.gridspec import GridSpec

# --- Parameters ---
n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
changing_el = 2  # The element that is changing

# --- Animation setup ---
n_frames = 40  # Number of frames in the animation
scan_deg = np.arange(-90, 91)  # Scan angles from -90° to +90°
scan_rad = np.radians(scan_deg)
steering_angle_deg = 0
steering_angle_rad = np.radians(steering_angle_deg)
ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
af0 = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
af0_dB = au.amplitude_to_dB_list(af0)

fig = plt.figure(figsize=(6, 9))
gs = GridSpec(2, 1, height_ratios=[6, 3])  # 6 for ax, 3 for ax2
# polar plot
ax = fig.add_subplot(gs[0], polar=True)
ax.plot(scan_rad, af0, lw=1, color='k', label="original pattern")
line1 = ax.plot([], [], lw=1, color='r', label="altered pattern")[0]
text =ax.text(0.05, 0.35, '', transform=ax.transAxes, fontsize=12, color='k')
ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Array Factor (linear scale)", labelpad=30)
# ax.set_theta_offset(np.pi/2)
ax.set_title("Beam Steering with 4-bit 1x8 Linear Array\n", pad = 30)
# Add a black horizontal line in the polar plot
r = np.linspace(0,1,100)
ax.plot(np.full_like(r, np.pi/2), r, color='k', lw=1, ls='-')
ax.plot(np.full_like(r, -np.pi/2), r, color='k', lw=1, ls='-')
# add a black dotted line at the steering angle
ax.plot(np.full_like(r, steering_angle_rad), r, color='k', lw=1, ls='--', label="steering angle")

# cartesian dB plot
ax2 = fig.add_subplot(gs[1])
ax2.plot(scan_deg, af0_dB, lw=1, color='k', label="original pattern")
ax2.set_xlim(-90, 90)
ax2.set_ylim(-50, 0)
ax2.set_xlabel("Angle ($^\\circ$)")
ax2.set_ylabel("Relative Gain (dB)")
ax2.grid()
dB1 = ax2.plot([], [], lw=1, color='r', label="altered pattern")[0]
dB_list = np.linspace(-50, 0, 100)  # dB scale
ax2.plot(np.full_like(dB_list, steering_angle_deg), dB_list, color='k', lw=1, ls='--')


def init():
    # animate must take a list
    line1.set_data([], [])
    text.set_text('')
    dB1.set_data([], [])
    return [line1, text, dB1]

def animate(i):
    # i is the frame number
    phase_shift = 2*np.pi / n_frames * i
    #phase_shift = au.quantise_phase(phase_shift, n_bits)

    altered_phase_list = np.copy(ideal_phase_list)
    altered_phase_list[changing_el] += phase_shift
    af1 = au.phase_list_to_af_list(altered_phase_list, scan_rad)
    kl = au.kl_divergence(af0, af1)
    line1.set_data(scan_rad, af1)

    af1_dB = au.amplitude_to_dB_list(af1)
    dB1.set_data(scan_deg, af1_dB)

    # loss at steering angle
    angle_index = np.where(scan_deg == steering_angle_deg)[0][0]
    loss = af0_dB[angle_index] - af1_dB[angle_index]
    # annotate the loss
    text.set_text(f"KL: {kl:.2f}\nLoss: {-loss:.2f} dB")

    
    ax.set_title(f"el {changing_el} at {phase_shift*180.0/np.pi:.1f}°", va='bottom', pad=20)
    return [ line1, text, dB1]

ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate, init_func=init,
    frames=n_frames, interval=100, blit=True
)

# To save as a gif: 
ani.save(f"visuals\\sweep\\changing_el_{changing_el}.gif", writer='pillow')
plt.show()
