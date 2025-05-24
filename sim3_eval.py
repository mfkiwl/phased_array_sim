import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils.antenna_utils as au

'''
Evaluates the performance of various bit patterns / optimisation techniques.
'''

# --- Parameters ---
n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
#n_stuck = 3  # Number of bits stuck (0 or 1)
scan_deg = np.arange(-90, 91)  # Scan angles from -90° to +90° in degrees
scan_rad = np.radians(scan_deg)  # Scan angles from -90° to +90° in radians
beamwidth = 14.3 # in degrees

def average_losses(n_elements=8, n_bits=4, n_stuck=3, trials=100):
    # for every trial, randomly select n_broken_bits bits to be stuck, then calculate the average nse_ over all steering angles
    #total_nse_quantised = 0
    #total_mb_nse_quantised = 0
    total_nse_broken = 0
    total_mb_nse_broken = 0
    total_nse_optim = 0
    total_mb_nse_optim = 0
    for i in range(trials):
        broken_elements, broken_bits, broken_values = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=0)
        #nse_quant_list = []
        nse_broken_list = []
        nse_optim_list = []
        #mb_nse_quant_list = []
        mb_nse_broken_list = []
        mb_nse_optim_list = []
        for steering_angle_deg in range(-60, 61):
            steering_angle_rad = np.radians(steering_angle_deg)
            # Calculate all the array factors and nse_s
            ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
            #quantised_phase_list = au.quantise_phase_list(ideal_phase_list, n_bits)
            af_ideal = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
            #af_quant = au.phase_list_to_af_list(quantised_phase_list, scan_rad)
            ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
            broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements, broken_bits, broken_values)
            broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
            optim_bit_array = au.el_by_el_optim(ideal_bit_array, broken_elements, broken_bits, broken_values)
            optim_phase_list = au.bit_array_to_phase_list(optim_bit_array)
            af_broken = au.phase_list_to_af_list(broken_phase_list, scan_rad)
            af_optim = au.phase_list_to_af_list(optim_phase_list, scan_rad)
            #nse_quant = au.nse_(af_ideal, af_quant)
            #nse_broken = au.nse_(af_ideal, af_broken)
            #nse_optim = au.nse_(af_ideal, af_optim)
            #nse_quant = au.normalised_SE(af_ideal, af_quant)
            nse_broken = au.normalised_SE(af_ideal, af_broken)
            nse_optim = au.normalised_SE(af_ideal, af_optim)

            # dB scale
            #af_ideal_dB = au.amplitude_to_dB_list(af_ideal)
            #af_quant_dB = au.amplitude_to_dB_list(af_quant)
            #af_broken_dB = au.amplitude_to_dB_list(af_broken)
            #af_optim_dB = au.amplitude_to_dB_list(af_optim)

            # losses at the steering angle
            angle_index = np.where((scan_deg >= steering_angle_deg - beamwidth/2.0) & (scan_deg <= steering_angle_deg + beamwidth/2.0))[0]
            #mb_nse_quant = af_quant_dB[angle_index] - af_ideal_dB[angle_index]
            #mb_nse_broken = af_broken_dB[angle_index] - af_ideal_dB[angle_index]
            #mb_nse_optim = af_optim_dB[angle_index] - af_ideal_dB[angle_index]
            #mb_nse_quant = au.normalised_SE(af_ideal[angle_index], af_quant[angle_index])
            mb_nse_broken = au.normalised_SE(af_ideal[angle_index], af_broken[angle_index])
            mb_nse_optim = au.normalised_SE(af_ideal[angle_index], af_optim[angle_index])

            #nse_quant_list.append(nse_quant)
            nse_broken_list.append(nse_broken)
            nse_optim_list.append(nse_optim)
            #mb_nse_quant_list.append(mb_nse_quant)
            mb_nse_broken_list.append(mb_nse_broken)
            mb_nse_optim_list.append(mb_nse_optim)
        #total_nse_quantised += np.mean(nse_quant_list)
        #total_mb_nse_quantised += np.mean(mb_nse_quant_list)
        total_nse_broken += np.mean(nse_broken_list)
        total_mb_nse_broken += np.mean(mb_nse_broken_list)
        total_nse_optim += np.mean(nse_optim_list)
        total_mb_nse_optim += np.mean(mb_nse_optim_list)

        print(f"Trial {i+1}/{trials} complete")

    # Calculate the average KL  and loss
    #avg_nse_quantised = total_nse_quantised / trials
    #avg_mb_nse_quantised = total_mb_nse_quantised / trials
    avg_nse_broken = total_nse_broken / trials
    avg_mb_nse_broken = total_mb_nse_broken / trials
    avg_nse_optim = total_nse_optim / trials
    avg_mb_nse_optim = total_mb_nse_optim / trials

    return avg_nse_broken, avg_mb_nse_broken, avg_nse_optim, avg_mb_nse_optim


# --- Main function ---
n_bits_broken_list = np.arange(1, 32)
n_trials = 25
#avg_nse_quantised_list = []
#avg_mb_nse_quantised_list = []
avg_nse_broken_list = []
avg_mb_nse_broken_list = []
avg_nse_optim_list = []
avg_mb_nse_optim_list = []
for n_broken_bits in n_bits_broken_list:
    avg_nse_broken, avg_mb_nse_broken, avg_nse_optim, avg_mb_nse_optim = average_losses(n_elements, n_bits, n_broken_bits, n_trials)
    #avg_nse_quantised_list.append(avg_nse_quantised)
    #avg_mb_nse_quantised_list.append(avg_mb_nse_quantised)
    avg_nse_broken_list.append(avg_nse_broken)
    avg_mb_nse_broken_list.append(avg_mb_nse_broken)
    avg_nse_optim_list.append(avg_nse_optim)
    avg_mb_nse_optim_list.append(avg_mb_nse_optim)

# --- Plotting ---
# 2 subplots, one for KL and one for Relative Gain (loss)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
fig.suptitle(f"Expected Relative Energy Loss for {n_elements} Elements and {n_bits} Bits")
fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
#axs[0].plot(n_bits_broken_list, avg_nse_quantised_list, label='Quantised', marker='x', color='blue')
axs[0].plot(n_bits_broken_list, avg_nse_broken_list, label='Broken', marker='x', color='red')
axs[0].plot(n_bits_broken_list, avg_nse_optim_list, label='Optimised', marker='x', color='green')
axs[0].set_title('Total normalised_SE')
axs[0].set_xlabel('Number of Stuck Bits')
axs[0].set_ylabel('Relative Energy')
axs[0].legend()
axs[0].grid()
#axs[1].plot(n_bits_broken_list, avg_mb_nse_quantised_list, label='Quantised', marker='x', color='blue')
axs[1].plot(n_bits_broken_list, avg_mb_nse_broken_list, label='Broken', marker='x', color='red')
axs[1].plot(n_bits_broken_list, avg_mb_nse_optim_list, label='Optimised', marker='x', color='green')
axs[1].set_title('Main Beam normalised_SE')
axs[1].set_xlabel('Number of Stuck Bits')
axs[1].set_ylabel('Relative Energy')
axs[1].legend()
axs[1].grid()
# Save the figure
fig.savefig(f"visuals\\evals\\sim3_eval_{n_elements}elements_{n_bits}bits.png", dpi=300, bbox_inches='tight')
plt.show()