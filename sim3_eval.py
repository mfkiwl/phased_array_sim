import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import antenna_utils as au

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
    # for every trial, randomly select n_broken_bits bits to be stuck, then calculate the average KL_divergence over all steering angles
    total_KL_quantised = 0
    total_loss_quantised = 0
    total_KL_broken = 0
    total_loss_broken = 0
    total_KL_optim = 0
    total_loss_optim = 0
    for i in range(trials):
        broken_elements, broken_bits, broken_values = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=0)
        KL_quant_list = []
        KL_broken_list = []
        KL_optim_list = []
        loss_quant_list = []
        loss_broken_list = []
        loss_optim_list = []
        for steering_angle_deg in range(-60, 61):
            steering_angle_rad = np.radians(steering_angle_deg)
            # Calculate all the array factors and KL_divergences
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
            #kl01 = au.kl_divergence(af0, af1)
            #kl02 = au.kl_divergence(af0, af2)
            #kl_optim = au.kl_divergence(af0, af_optim)
            kl01 = au.MSE(af0, af1)
            kl02 = au.MSE(af0, af2)
            kl_optim = au.MSE(af0, af_optim)

            # dB scale
            af0_dB = au.amplitude_to_dB_list(af0)
            af1_dB = au.amplitude_to_dB_list(af1)
            af2_dB = au.amplitude_to_dB_list(af2)
            af_optim_dB = au.amplitude_to_dB_list(af_optim)

            # losses at the steering angle
            angle_index = np.where((scan_deg >= steering_angle_deg - beamwidth/2.0) & (scan_deg <= steering_angle_deg + beamwidth/2.0))[0]
            #loss01 = af1_dB[angle_index] - af0_dB[angle_index]
            #loss02 = af2_dB[angle_index] - af0_dB[angle_index]
            #loss_optim = af_optim_dB[angle_index] - af0_dB[angle_index]
            loss01 = au.MSE(af0[angle_index], af1[angle_index])
            loss02 = au.MSE(af0[angle_index], af2[angle_index])
            loss_optim = au.MSE(af0[angle_index], af_optim[angle_index])

            KL_quant_list.append(kl01)
            KL_broken_list.append(kl02)
            KL_optim_list.append(kl_optim)
            loss_quant_list.append(loss01)
            loss_broken_list.append(loss02)
            loss_optim_list.append(loss_optim)
        total_KL_quantised += np.mean(KL_quant_list)
        total_loss_quantised += np.mean(loss_quant_list)
        total_KL_broken += np.mean(KL_broken_list)
        total_loss_broken += np.mean(loss_broken_list)
        total_KL_optim += np.mean(KL_optim_list)
        total_loss_optim += np.mean(loss_optim_list)

        print(f"Trial {i+1}/{trials} complete")

    # Calculate the average KL divergence and loss
    avg_KL_quantised = total_KL_quantised / trials
    avg_loss_quantised = total_loss_quantised / trials
    avg_KL_broken = total_KL_broken / trials
    avg_loss_broken = total_loss_broken / trials
    avg_KL_optim = total_KL_optim / trials
    avg_loss_optim = total_loss_optim / trials

    return avg_KL_quantised, avg_loss_quantised, avg_KL_broken, avg_loss_broken, avg_KL_optim, avg_loss_optim


# --- Main function ---
n_bits_broken_list = np.arange(1, 32)
n_trials = 50
avg_KL_quantised_list = []
avg_loss_quantised_list = []
avg_KL_broken_list = []
avg_loss_broken_list = []
avg_KL_optim_list = []
avg_loss_optim_list = []
for n_broken_bits in n_bits_broken_list:
    avg_KL_quantised, avg_loss_quantised, avg_KL_broken, avg_loss_broken, avg_KL_optim, avg_loss_optim = average_losses(n_elements, n_bits, n_broken_bits, n_trials)
    avg_KL_quantised_list.append(avg_KL_quantised)
    avg_loss_quantised_list.append(avg_loss_quantised)
    avg_KL_broken_list.append(avg_KL_broken)
    avg_loss_broken_list.append(avg_loss_broken)
    avg_KL_optim_list.append(avg_KL_optim)
    avg_loss_optim_list.append(avg_loss_optim)

# --- Plotting ---
# 2 subplots, one for KL and one for Relative Gain (loss)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
fig.suptitle(f"Average KL Divergence and Loss for {n_elements} Elements and {n_bits} Bits")
fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
axs[0].plot(n_bits_broken_list, avg_KL_quantised_list, label='Quantised', marker='x', color='blue')
axs[0].plot(n_bits_broken_list, avg_KL_broken_list, label='Broken', marker='x', color='red')
axs[0].plot(n_bits_broken_list, avg_KL_optim_list, label='Optimised', marker='x', color='green')
axs[0].set_title('Average MSE')
axs[0].set_xlabel('Number of Stuck Bits')
axs[0].set_ylabel('Energy')
axs[0].legend()
axs[0].grid()
axs[1].plot(n_bits_broken_list, avg_loss_quantised_list, label='Quantised', marker='x', color='blue')
axs[1].plot(n_bits_broken_list, avg_loss_broken_list, label='Broken', marker='x', color='red')
axs[1].plot(n_bits_broken_list, avg_loss_optim_list, label='Optimised', marker='x', color='green')
axs[1].set_title('Main Beam MSE')
axs[1].set_xlabel('Number of Stuck Bits')
axs[1].set_ylabel('Energy')
axs[1].legend()
axs[1].grid()
# Save the figure
fig.savefig(f"visuals\\evals\\sim3_eval_{n_elements}elements_{n_bits}bits.png", dpi=300, bbox_inches='tight')
plt.show()