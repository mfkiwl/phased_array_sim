import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import utils.antenna_utils as au
from matplotlib.gridspec import GridSpec
import utils.ga as ga
import utils.bpso as bp


# --- Parameters ---
n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
n_stuck = 4  # Number of bits stuck (0 or 1)
beamwidth_rad = np.radians(14.3)  # Beamwidth in radians for the genetic algorithm

broken_elements, broken_bits, broken_values = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=1)



# --- Animation setup ---
steering_angles = np.linspace(-90, 90, 61)  # Steering angles from -90° to +90°
n_frames = len(steering_angles)  # Number of frames in the animation
scan_deg = np.arange(-90, 91)  # Scan angles from -90° to +90°
scan_rad = np.radians(scan_deg)

fig = plt.figure(figsize=(6, 9))
gs = GridSpec(2, 1, height_ratios=[6, 3])  # 6 for ax, 3 for ax2
# polar plot
ax = fig.add_subplot(gs[0], polar=True)

ax.set_ylim(0, 1)
ax.set_ylabel("Array Factor (linear scale)", labelpad=30)
# ax.set_theta_offset(np.pi/2)
ax.set_title("Beam Steering with 4-bit 1x8 Linear Array\n", pad = 20)
# Add a black horizontal line in the polar plot
r = np.linspace(0,1,100)
ax.plot(np.full_like(r, np.pi/2), r, color='k', lw=1, ls='-')
ax.plot(np.full_like(r, -np.pi/2), r, color='k', lw=1, ls='-')
# add a black dotted line at the steering angle
steer_line = ax.plot([], [], color='k', lw=1, ls='--')[0]


# add the plots
line_ideal = ax.plot([], [], lw=1, color='k', label="ideal")[0]
#line_quant = ax.plot([], [], lw=1, color='c', label="quantised")[0]
line_broke = ax.plot([], [], lw=1, color='r', label="broken")[0]
line_elop = ax.plot([], [], lw=1, color='b', label="ELOP")[0]
line_ga = ax.plot([], [], lw=1, color='m', label="GA")[0]
line_bpso = ax.plot([], [], lw=1, color='g', label="BPSO")[0]
text = ax.text(0.05, 0.35, '', transform=ax.transAxes, fontsize=12, color='k')

ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)



# cartesian dB plot
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(-90, 90)
ax2.set_ylim(-50, 0)
ax2.set_xlabel("Angle ($^\\circ$)")
ax2.set_ylabel("Relative Gain (dB)")
ax2.grid()
dB_list = np.linspace(-50, 0, 100)  # dB scale

dB_ideal = ax2.plot([], [], lw=1, color='k', label="ideal")[0]
#dB_quant = ax2.plot([], [], lw=1, color='c', label="quantised")[0]
dB_broke = ax2.plot([], [], lw=1, color='r', label="broken")[0]
dB_elop = ax2.plot([], [], lw=1, color='b', label="ELOP")[0]
dB_ga = ax2.plot([], [], lw=1, color='m', label="GA")[0]
dB_bpso = ax2.plot([], [], lw=1, color='g', label="BPSO")[0]
steer_line2 = ax2.plot([], [], color='k', lw=1, ls='--')[0]

def init():
    # animate must take a list
    line_ideal.set_data([], [])
    #line_quant.set_data([], [])
    line_broke.set_data([], [])
    line_elop.set_data([], [])
    line_ga.set_data([], [])
    line_bpso.set_data([], [])
    text.set_text('')
    steer_line.set_data([], [])

    dB_ideal.set_data([], [])
    #dB_quant.set_data([], [])
    dB_broke.set_data([], [])
    dB_elop.set_data([], [])
    dB_ga.set_data([], [])
    dB_bpso.set_data([], [])
    steer_line2.set_data([], [])
    return [line_ideal, 
            #line_quant, 
            line_broke, line_elop, line_ga, line_bpso, text, steer_line,
            dB_ideal, 
            #dB_quant, 
            dB_broke, dB_elop, dB_ga, dB_bpso, steer_line2]

def animate(i):
    # i is the frame number
    steering_angle_deg = steering_angles[i]
    steering_angle_rad = np.deg2rad(steering_angle_deg)
    print(f"Animating Frame {i+1}/{n_frames}, Steering Angle: {steering_angle_deg:.1f}°")  # Print current frame number

    # Calculate all the array factors
    ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
    #quantised_phase_list = au.quantise_phase_list(ideal_phase_list, n_bits)
    af_ideal = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
    #af_quant = au.phase_list_to_af_list(quantised_phase_list, scan_rad)

    ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
    broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements, broken_bits, broken_values)
    broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
    elop_bit_array = au.el_by_el_optim(ideal_bit_array, broken_elements, broken_bits, broken_values)
    elop_phase_list = au.bit_array_to_phase_list(elop_bit_array)
    af_broke = au.phase_list_to_af_list(broken_phase_list, scan_rad)
    af_elop = au.phase_list_to_af_list(elop_phase_list, scan_rad)

    best_ga = ga.genetic_algorithm(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values)
    ga_bit_array = ga.list_to_bit_array(best_ga)
    ga_broken_bit_array = au.break_bit_array(ga_bit_array, broken_elements, broken_bits, broken_values)
    ga_broken_phase_list = au.bit_array_to_phase_list(ga_broken_bit_array)
    af_ga = au.phase_list_to_af_list(ga_broken_phase_list, scan_rad)

    best_bpso, _ = bp.binary_pso(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values)
    bpso_bit_array = bp.list_to_bit_array(best_bpso)
    bpso_broken_bit_array = au.break_bit_array(bpso_bit_array, broken_elements, broken_bits, broken_values)
    bpso_broken_phase_list = au.bit_array_to_phase_list(bpso_broken_bit_array)
    af_bpso = au.phase_list_to_af_list(bpso_broken_phase_list, scan_rad)

    #loss_quant = au.normalised_SE(af_ideal, af_quant)
    loss_broken = au.normalised_SE(af_ideal, af_broke)
    loss_elop = au.normalised_SE(af_ideal, af_elop)
    loss_ga = au.normalised_SE(af_ideal, af_ga)
    loss_bpso = au.normalised_SE(af_ideal, af_bpso)
    loss_text = f"$MSE_{{broken}}$: {loss_broken:.2f}\n$MSE_{{ELOP}}$: {loss_elop:.2f}\n$MSE_{{GA}}$: {loss_ga:.2f}\n$MSE_{{BPSO}}$: {loss_bpso:.2f}"

    # Update the polar plot
    line_ideal.set_data(scan_rad, af_ideal)
    #line_quant.set_data(scan_rad, af_quant)
    line_broke.set_data(scan_rad, af_broke)
    line_elop.set_data(scan_rad, af_elop)
    line_ga.set_data(scan_rad, af_ga)
    line_bpso.set_data(scan_rad, af_bpso)

    # steering line
    steer_line.set_data(np.full_like(r, steering_angle_rad), r)

    # dB scale
    af_ideal_dB = au.amplitude_to_dB_list(af_ideal)
    #af_quant_dB = au.amplitude_to_dB_list(af_quant)
    af_broke_dB = au.amplitude_to_dB_list(af_broke)
    af_elop_dB = au.amplitude_to_dB_list(af_elop)
    af_ga_dB = au.amplitude_to_dB_list(af_ga)
    af_bpso_dB = au.amplitude_to_dB_list(af_bpso)
    dB_ideal.set_data(scan_deg, af_ideal_dB)
    #dB_quant.set_data(scan_deg, af_quant_dB)
    dB_broke.set_data(scan_deg, af_broke_dB)
    dB_elop.set_data(scan_deg, af_elop_dB)
    dB_ga.set_data(scan_deg, af_ga_dB)
    dB_bpso.set_data(scan_deg, af_bpso_dB)
    # steering line
    steer_line2.set_data(np.full_like(dB_list, steering_angle_deg), dB_list)

    # annotate the value
    text.set_text(
        loss_text
    )

    # title
    ax.set_title(f"{n_stuck} bits stuck, Steering Angle: {steering_angle_deg:.1f}°", va='bottom', pad=30)
    return [line_ideal, 
            #line_quant, 
            line_broke, line_elop, line_ga, line_bpso, text, steer_line,
            dB_ideal, 
            #dB_quant, 
            dB_broke, dB_elop, dB_ga, dB_bpso, steer_line2]

ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate, init_func=init,
    frames=n_frames, interval=100, blit=True
)

# To save as a gif: 
ani.save(f"visuals\\optim_compare_{n_stuck}_bits_stuck.gif", writer='pillow')
#plt.show()
