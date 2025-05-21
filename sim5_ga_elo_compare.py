import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import antenna_utils as au
import ga
import pandas as pd


# --- Parameters ---
n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
#n_stuck = 3  # Number of bits stuck (0 or 1)
scan_deg = np.arange(-90, 91)  # Scan angles from -90° to +90° in degrees
scan_rad = np.radians(scan_deg)  # Scan angles from -90° to +90° in radians
beamwidth_deg = 14.5 # in degrees # This is arcsin(2/N)
beamwidth_rad = np.radians(beamwidth_deg)  # Beamwidth in radians

def average_losses(n_elements=8, n_bits=4, n_stuck=3, trials=100):
    # for every trial, randomly select n_broken_bits bits to be stuck, then calculate the average nse_ over all steering angles
    #total_nse_quantised = 0
    #total_mb_nse_quantised = 0
    total_nse_broken = 0
    total_pssl_broken = 0
    total_pbp_broken = 0
    #total_mb_nse_broken = 0
    total_nse_optim = 0
    total_pssl_optim = 0
    total_pbp_optim = 0
    #total_mb_nse_optim = 0
    total_nse_ga = 0
    total_pssl_ga = 0
    total_pbp_ga = 0
    #total_mb_nse_ga = 0
    for i in range(trials):
        broken_elements, broken_bits, broken_values = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=0)
        #nse_quant_list = [] # total nse
        nse_broken_list = []
        nse_optim_list = []
        nse_ga_list = []
        pssl_broken_list = []
        pssl_optim_list = []
        pssl_ga_list = []
        pbp_broken_list = []
        pbp_optim_list = []
        pbp_ga_list = []
        #mb_nse_quant_list = []
        #mb_nse_broken_list = [] # main beam nse
        #mb_nse_optim_list = []
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

            # GA
            best = ga.genetic_algorithm(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad)
            best_bit_array = ga.string_to_bit_array(best)
            ga_bit_array = au.break_bit_array(best_bit_array, broken_elements, broken_bits, broken_values)
            ga_phase_list = au.bit_array_to_phase_list(ga_bit_array)
            af_ga = au.phase_list_to_af_list(ga_phase_list, scan_rad)


            #nse_quant = au.nse_(af_ideal, af_quant)
            #nse_broken = au.nse_(af_ideal, af_broken)
            #nse_optim = au.nse_(af_ideal, af_optim)
            #nse_quant = au.normalised_SE(af_ideal, af_quant)
            nse_broken = au.normalised_SE(af_ideal, af_broken)
            nse_optim = au.normalised_SE(af_ideal, af_optim)
            nse_ga = au.normalised_SE(af_ideal, af_ga)
            pssl_broken = au.PSSL(af_broken, scan_rad, steering_angle_rad, beamwidth_rad)
            pssl_optim = au.PSSL(af_optim, scan_rad, steering_angle_rad, beamwidth_rad)
            pssl_ga = au.PSSL(af_ga, scan_rad, steering_angle_rad, beamwidth_rad)
            pbp_broken = au.PBP(af_broken, scan_rad, steering_angle_rad)
            pbp_optim = au.PBP(af_optim, scan_rad, steering_angle_rad)
            pbp_ga = au.PBP(af_ga, scan_rad, steering_angle_rad)

            #nse_quant_list.append(nse_quant)
            nse_broken_list.append(nse_broken)
            nse_optim_list.append(nse_optim)
            nse_ga_list.append(nse_ga)
            pssl_broken_list.append(pssl_broken)
            pssl_optim_list.append(pssl_optim)
            pssl_ga_list.append(pssl_ga)
            pbp_broken_list.append(pbp_broken)
            pbp_optim_list.append(pbp_optim)
            pbp_ga_list.append(pbp_ga)

        total_nse_broken += np.mean(nse_broken_list)
        total_nse_optim += np.mean(nse_optim_list)
        total_nse_ga += np.mean(nse_ga_list)
        total_pssl_broken += np.mean(pssl_broken_list)
        total_pssl_optim += np.mean(pssl_optim_list)
        total_pssl_ga += np.mean(pssl_ga_list)
        total_pbp_broken += np.mean(pbp_broken_list)
        total_pbp_optim += np.mean(pbp_optim_list)
        total_pbp_ga += np.mean(pbp_ga_list)

        print(f"Trial {i+1}/{trials} complete")

    # Calculate the average KL  and loss
    #avg_nse_quantised = total_nse_quantised / trials
    #avg_mb_nse_quantised = total_mb_nse_quantised / trials
    avg_nse_broken = total_nse_broken / trials
    avg_nse_optim = total_nse_optim / trials
    avg_nse_ga = total_nse_ga / trials
    avg_pssl_broken = total_pssl_broken / trials
    avg_pssl_optim = total_pssl_optim / trials
    avg_pssl_ga = total_pssl_ga / trials
    avg_pbp_broken = total_pbp_broken / trials
    avg_pbp_optim = total_pbp_optim / trials
    avg_pbp_ga = total_pbp_ga / trials

    return avg_nse_broken, avg_nse_optim, avg_nse_ga, avg_pssl_broken, avg_pssl_optim, avg_pssl_ga, avg_pbp_broken, avg_pbp_optim, avg_pbp_ga


# --- Main function ---
n_bits_broken_list = np.arange(1, 32)
n_trials = 25
#avg_nse_quantised_list = []
#avg_mb_nse_quantised_list = []
avg_nse_broken_list = []
avg_nse_optim_list = []
avg_nse_ga_list = []
avg_pssl_broken_list = []
avg_pssl_optim_list = []
avg_pssl_ga_list = []
avg_pbp_broken_list = []
avg_pbp_optim_list = []
avg_pbp_ga_list = []
for n_broken_bits in n_bits_broken_list:
    avg_nse_broken, avg_nse_optim, avg_nse_ga, avg_pssl_broken, avg_pssl_optim, avg_pssl_ga, avg_pbp_broken, avg_pbp_optim, avg_pbp_ga = average_losses(
        n_elements=n_elements, n_bits=n_bits, n_stuck=n_broken_bits, trials=n_trials)
    #avg_nse_quantised_list.append(avg_nse_quantised)
    #avg_mb_nse_quantised_list.append(avg_mb_nse_quantised)
    avg_nse_broken_list.append(avg_nse_broken)
    avg_nse_optim_list.append(avg_nse_optim)
    avg_nse_ga_list.append(avg_nse_ga)
    avg_pssl_broken_list.append(avg_pssl_broken)
    avg_pssl_optim_list.append(avg_pssl_optim)
    avg_pssl_ga_list.append(avg_pssl_ga)
    avg_pbp_broken_list.append(avg_pbp_broken)
    avg_pbp_optim_list.append(avg_pbp_optim)
    avg_pbp_ga_list.append(avg_pbp_ga)

# save the results to a CSV file
df = pd.DataFrame({
    'avg_nse_broken': avg_nse_broken_list,
    'avg_nse_optim': avg_nse_optim_list,
    'avg_nse_ga': avg_nse_ga_list,
    'avg_pssl_broken': avg_pssl_broken_list,
    'avg_pssl_optim': avg_pssl_optim_list,
    'avg_pssl_ga': avg_pssl_ga_list,
    'avg_pbp_broken': avg_pbp_broken_list,
    'avg_pbp_optim': avg_pbp_optim_list,
    'avg_pbp_ga': avg_pbp_ga_list
})
df.to_csv("ga_elo_compare.csv")

# --- Plotting ---
# 3 subplots, one for NSe, one for PSSL and one for PBP
fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
fig.suptitle(f"Comparing GA and ELOP for {n_elements} Elements and {n_bits} Bits")
fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
#axs[0].plot(n_bits_broken_list, avg_nse_quantised_list, label='Quantised', marker='x', color='blue')
axs[0].plot(n_bits_broken_list, avg_nse_broken_list, label='Broken', marker='x', color='red')
axs[0].plot(n_bits_broken_list, avg_nse_optim_list, label='ELOP', marker='x', color='green')
axs[0].plot(n_bits_broken_list, avg_nse_ga_list, label='GA', marker='x', color='purple')
axs[0].set_title('Total normalised_SE')
axs[0].set_xlabel('Number of Stuck Bits')
axs[0].set_ylabel('Relative Energy Distribution Difference')
axs[0].legend()
axs[0].grid()
#axs[1].plot(n_bits_broken_list, avg_mb_nse_quantised_list, label='Quantised', marker='x', color='blue')
axs[1].plot(n_bits_broken_list, avg_pssl_broken_list, label='Broken', marker='x', color='red')
axs[1].plot(n_bits_broken_list, avg_pssl_optim_list, label='ELOP', marker='x', color='green')
axs[1].plot(n_bits_broken_list, avg_pssl_ga_list, label='GA', marker='x', color='purple')
axs[1].set_title('Main Beam PSSL')
axs[1].set_xlabel('Number of Stuck Bits')
axs[1].set_ylabel('PSSL [dB]')
axs[1].legend()
axs[1].grid()
axs[2].plot(n_bits_broken_list, avg_pbp_broken_list, label='Broken', marker='x', color='red')
axs[2].plot(n_bits_broken_list, avg_pbp_optim_list, label='ELOP', marker='x', color='green')
axs[2].plot(n_bits_broken_list, avg_pbp_ga_list, label='GA', marker='x', color='purple')
axs[2].set_title('PBP')
axs[2].set_xlabel('Number of Stuck Bits')
axs[2].set_ylabel('PBP [dB]')
axs[2].legend()
axs[2].grid()
# Save the figure
fig.savefig(f"visuals\\evals\\sim5_GAELOP_{n_elements}elements_{n_bits}bits.png", dpi=300, bbox_inches='tight')
plt.show()