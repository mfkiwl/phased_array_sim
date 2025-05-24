import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils.antenna_utils as au
import utils.ga as ga
import pandas as pd
import utils.bpso as bpso
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process an input file and save results."
    )

    # ── positional arguments ──────────────────────────────────────
    parser.add_argument("-id",  type=str, default="", help="ID for the Path to the output files.")
    parser.add_argument(
        "-n_trials", type=int, default=100, help="Number of trials per n_stuck_bits."
    )
    parser.add_argument(
        "-upper_bound", type=int, default=32, help="Maximum number of stuck bits."
    )
    parser.add_argument(
        "-lower_bound", type=int, default=0, help="Minimum number of stuck bits."
    )
    '''
    parser.add_argument("output_data",  help="Path to the output data file.")
    parser.add_argument("output_plot", help="Path to the output plot.")

    # ── optional flags & switches ─────────────────────────────────
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5,
        help="Detection threshold (default: 0.5)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print progress messages."
    )
    '''
    return parser.parse_args()

# parse the command line arguments
args = parse_args()
id = args.id
n_trials = args.n_trials
upper_bound = args.upper_bound
lower_bound = args.lower_bound
# --- Path to the output files ---
output_data = f"ult_eval_data\\ga_elo_bpso_compare_{id}.csv"
output_plot = f"visuals\\evals\\ga_elo_bpso_compare_{id}.png"


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
    total_isll_broken = 0
    #total_mb_nse_broken = 0
    total_nse_elo = 0
    total_pssl_elo = 0
    total_pbp_elo = 0
    total_isll_elo = 0
    #total_mb_nse_elo = 0
    total_nse_ga = 0
    total_pssl_ga = 0
    total_pbp_ga = 0
    total_isll_ga = 0
    #total_mb_nse_ga = 0
    total_nse_bpso = 0
    total_pssl_bpso = 0
    total_pbp_bpso = 0
    total_isll_bpso = 0
    #total_mb_nse_bpso = 0
    for i in range(trials):
        broken_elements, broken_bits, broken_values = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=0)
        #nse_quant_list = [] # total nse
        '''
        nse_broken_list = []
        nse_elo_list = []
        nse_ga_list = []
        nse_bpso_list = []
        pssl_broken_list = []
        pssl_elo_list = []
        pssl_ga_list = []
        pssl_bpso_list = []
        pbp_broken_list = []
        pbp_elo_list = []
        pbp_ga_list = []
        pbp_bpso_list = []
        '''
        #mb_nse_quant_list = []
        #mb_nse_broken_list = [] # main beam nse
        #mb_nse_elo_list = []
    #for steering_angle_deg in np.arange(-60, 61, 30):
        steering_angle_deg = np.random.randint(-60, 61)
        steering_angle_rad = np.radians(steering_angle_deg)
        # Calculate all the array factors and nse_s
        ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
        #quantised_phase_list = au.quantise_phase_list(ideal_phase_list, n_bits)
        af_ideal = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
        #af_quant = au.phase_list_to_af_list(quantised_phase_list, scan_rad)
        ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
        broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements, broken_bits, broken_values)
        broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
        elo_bit_array = au.el_by_el_optim(ideal_bit_array, broken_elements, broken_bits, broken_values)
        elo_phase_list = au.bit_array_to_phase_list(elo_bit_array)
        af_broken = au.phase_list_to_af_list(broken_phase_list, scan_rad)
        af_elo = au.phase_list_to_af_list(elo_phase_list, scan_rad)

        # GA
        best = ga.genetic_algorithm(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values)
        best_bit_array = ga.list_to_bit_array(best)
        ga_bit_array = au.break_bit_array(best_bit_array, broken_elements, broken_bits, broken_values)
        ga_phase_list = au.bit_array_to_phase_list(ga_bit_array)
        af_ga = au.phase_list_to_af_list(ga_phase_list, scan_rad)

        #BPSO
        best_bpso, _ = bpso.binary_pso(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values)
        best_bpso_bit_array = bpso.list_to_bit_array(best_bpso)
        bpso_bit_array = au.break_bit_array(best_bpso_bit_array, broken_elements, broken_bits, broken_values)
        bpso_phase_list = au.bit_array_to_phase_list(bpso_bit_array)
        af_bpso = au.phase_list_to_af_list(bpso_phase_list, scan_rad)


        #nse_quant = au.nse_(af_ideal, af_quant)
        #nse_broken = au.nse_(af_ideal, af_broken)
        #nse_elo = au.nse_(af_ideal, af_elo)
        #nse_quant = au.normalised_SE(af_ideal, af_quant)
        nse_broken = au.normalised_SE(af_ideal, af_broken)
        nse_elo = au.normalised_SE(af_ideal, af_elo)
        nse_ga = au.normalised_SE(af_ideal, af_ga)
        nse_bpso = au.normalised_SE(af_ideal, af_bpso)
        pssl_broken = au.PSSL(af_broken, scan_rad, steering_angle_rad, beamwidth_rad)
        pssl_elo = au.PSSL(af_elo, scan_rad, steering_angle_rad, beamwidth_rad)
        pssl_ga = au.PSSL(af_ga, scan_rad, steering_angle_rad, beamwidth_rad)
        pssl_bpso = au.PSSL(af_bpso, scan_rad, steering_angle_rad, beamwidth_rad)
        pbp_broken = au.PBP(af_broken, scan_rad, steering_angle_rad)
        pbp_elo = au.PBP(af_elo, scan_rad, steering_angle_rad)
        pbp_ga = au.PBP(af_ga, scan_rad, steering_angle_rad)
        pbp_bpso = au.PBP(af_bpso, scan_rad, steering_angle_rad)
        isll_broken = au.ISLL(af_broken, scan_rad, steering_angle_rad, beamwidth_rad)
        isll_elo = au.ISLL(af_elo, scan_rad, steering_angle_rad, beamwidth_rad)
        isll_ga = au.ISLL(af_ga, scan_rad, steering_angle_rad, beamwidth_rad)
        isll_bpso = au.ISLL(af_bpso, scan_rad, steering_angle_rad, beamwidth_rad)

        '''
        #nse_quant_list.append(nse_quant)
        nse_broken_list.append(nse_broken)
        nse_elo_list.append(nse_elo)
        nse_ga_list.append(nse_ga)
        nse_bpso_list.append(nse_bpso)
        pssl_broken_list.append(pssl_broken)
        pssl_elo_list.append(pssl_elo)
        pssl_ga_list.append(pssl_ga)
        pssl_bpso_list.append(pssl_bpso)
        pbp_broken_list.append(pbp_broken)
        pbp_elo_list.append(pbp_elo)
        pbp_ga_list.append(pbp_ga)
        pbp_bpso_list.append(pbp_bpso)

        #print(f"Trial {i+1}/{trials}, Steering Angle {steering_angle_deg}° complete")

        total_nse_broken += np.mean(nse_broken_list)
        total_nse_elo += np.mean(nse_elo_list)
        total_nse_ga += np.mean(nse_ga_list)
        total_nse_bpso += np.mean(nse_bpso_list)
        total_pssl_broken += np.mean(pssl_broken_list)
        total_pssl_elo += np.mean(pssl_elo_list)
        total_pssl_ga += np.mean(pssl_ga_list)
        total_pssl_bpso += np.mean(pssl_bpso_list)
        total_pbp_broken += np.mean(pbp_broken_list)
        total_pbp_elo += np.mean(pbp_elo_list)
        total_pbp_ga += np.mean(pbp_ga_list)
        total_pbp_bpso += np.mean(pbp_bpso_list)
        '''
        total_nse_broken += nse_broken
        total_nse_elo += nse_elo
        total_nse_ga += nse_ga
        total_nse_bpso += nse_bpso
        total_pssl_broken += pssl_broken
        total_pssl_elo += pssl_elo
        total_pssl_ga += pssl_ga
        total_pssl_bpso += pssl_bpso
        total_pbp_broken += pbp_broken
        total_pbp_elo += pbp_elo
        total_pbp_ga += pbp_ga
        total_pbp_bpso += pbp_bpso
        total_isll_broken += isll_broken
        total_isll_elo += isll_elo
        total_isll_ga += isll_ga
        total_isll_bpso += isll_bpso

        print(f"Trial {i+1}/{trials} complete")

    # Calculate the average KL  and loss
    #avg_nse_quantised = total_nse_quantised / trials
    #avg_mb_nse_quantised = total_mb_nse_quantised / trials
    avg_nse_broken = total_nse_broken / trials
    avg_nse_elo = total_nse_elo / trials
    avg_nse_ga = total_nse_ga / trials
    avg_nse_bpso = total_nse_bpso / trials
    avg_pssl_broken = total_pssl_broken / trials
    avg_pssl_elo = total_pssl_elo / trials
    avg_pssl_ga = total_pssl_ga / trials
    avg_pssl_bpso = total_pssl_bpso / trials
    avg_pbp_broken = total_pbp_broken / trials
    avg_pbp_elo = total_pbp_elo / trials
    avg_pbp_ga = total_pbp_ga / trials
    avg_pbp_bpso = total_pbp_bpso / trials
    avg_isll_broken = total_isll_broken / trials
    avg_isll_elo = total_isll_elo / trials
    avg_isll_ga = total_isll_ga / trials
    avg_isll_bpso = total_isll_bpso / trials

    return avg_nse_broken, avg_nse_elo, avg_nse_ga, avg_nse_bpso, avg_pssl_broken, avg_pssl_elo, avg_pssl_ga, avg_pssl_bpso, avg_pbp_broken, avg_pbp_elo, avg_pbp_ga, avg_pbp_bpso, avg_isll_broken, avg_isll_elo, avg_isll_ga, avg_isll_bpso


# --- Main function ---
n_bits_broken_list = np.arange(lower_bound, upper_bound+1)
n_trials = n_trials
#avg_nse_quantised_list = []
#avg_mb_nse_quantised_list = []
avg_nse_broken_list = []
avg_nse_elo_list = []
avg_nse_ga_list = []
avg_nse_bpso_list = []
avg_pssl_broken_list = []
avg_pssl_elo_list = []
avg_pssl_ga_list = []
avg_pssl_bpso_list = []
avg_pbp_broken_list = []
avg_pbp_elo_list = []
avg_pbp_ga_list = []
avg_pbp_bpso_list = []
avg_isll_broken_list = []
avg_isll_elo_list = []
avg_isll_ga_list = []
avg_isll_bpso_list = []
for n_broken_bits in n_bits_broken_list:
    print(f"Number of stuck bits: {n_broken_bits}")
    avg_nse_broken, avg_nse_elo, avg_nse_ga, avg_nse_bpso, avg_pssl_broken, avg_pssl_elo, avg_pssl_ga, avg_pssl_bpso, avg_pbp_broken, avg_pbp_elo, avg_pbp_ga, avg_pbp_bpso, avg_isll_broken, avg_isll_elo, avg_isll_ga, avg_isll_bpso = average_losses(
        n_elements=n_elements, n_bits=n_bits, n_stuck=n_broken_bits, trials=n_trials)
    #avg_nse_quantised_list.append(avg_nse_quantised)
    #avg_mb_nse_quantised_list.append(avg_mb_nse_quantised)
    avg_nse_broken_list.append(avg_nse_broken)
    avg_nse_elo_list.append(avg_nse_elo)
    avg_nse_ga_list.append(avg_nse_ga)
    avg_nse_bpso_list.append(avg_nse_bpso)
    avg_pssl_broken_list.append(avg_pssl_broken)
    avg_pssl_elo_list.append(avg_pssl_elo)
    avg_pssl_ga_list.append(avg_pssl_ga)
    avg_pssl_bpso_list.append(avg_pssl_bpso)
    avg_pbp_broken_list.append(avg_pbp_broken)
    avg_pbp_elo_list.append(avg_pbp_elo)
    avg_pbp_ga_list.append(avg_pbp_ga)
    avg_pbp_bpso_list.append(avg_pbp_bpso)
    avg_isll_broken_list.append(avg_isll_broken)
    avg_isll_elo_list.append(avg_isll_elo)
    avg_isll_ga_list.append(avg_isll_ga)
    avg_isll_bpso_list.append(avg_isll_bpso)

# save the results to a CSV file
df = pd.DataFrame({
    'avg_nse_broken': avg_nse_broken_list,
    'avg_nse_elo': avg_nse_elo_list,
    'avg_nse_ga': avg_nse_ga_list,
    'avg_nse_bpso': avg_nse_bpso_list,
    'avg_pssl_broken': avg_pssl_broken_list,
    'avg_pssl_elo': avg_pssl_elo_list,
    'avg_pssl_ga': avg_pssl_ga_list,
    'avg_pssl_bpso': avg_pssl_bpso_list,
    'avg_pbp_broken': avg_pbp_broken_list,
    'avg_pbp_elo': avg_pbp_elo_list,
    'avg_pbp_ga': avg_pbp_ga_list,
    'avg_pbp_bpso': avg_pbp_bpso_list,
    'avg_isll_broken': avg_isll_broken_list,
    'avg_isll_elo': avg_isll_elo_list,
    'avg_isll_ga': avg_isll_ga_list,
    'avg_isll_bpso': avg_isll_bpso_list,
})
df.to_csv(output_data)
'''
# --- Plotting ---
n_bits_broken_list = n_bits_broken_list / (n_elements * n_bits)  # normalise the number of stuck bits
# 3 subplots, one for NSe, one for PSSL and one for PBP
fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
fig.suptitle(f"Comparing GA and ELO for {n_elements} Elements and {n_bits} Bits")
fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
#axs[0].plot(n_bits_broken_list, avg_nse_quantised_list, label='Quantised', marker='x', color='blue')
axs[0].plot(n_bits_broken_list, avg_nse_broken_list, label='Broken', marker='x', color='red')
axs[0].plot(n_bits_broken_list, avg_nse_elo_list, label='ELO', marker='x', color='green')
axs[0].plot(n_bits_broken_list, avg_nse_ga_list, label='GA', marker='x', color='purple')
axs[0].plot(n_bits_broken_list, avg_nse_bpso_list, label='BPSO', marker='x', color='cyan')
axs[0].set_title('Total normalised_SE')
axs[0].set_xlabel('Proportion of Stuck Bits')
axs[0].set_ylabel('Relative Energy Distribution Difference')
axs[0].legend()
axs[0].grid()
#axs[1].plot(n_bits_broken_list, avg_mb_nse_quantised_list, label='Quantised', marker='x', color='blue')
axs[1].plot(n_bits_broken_list, avg_pssl_broken_list, label='Broken', marker='x', color='red')
axs[1].plot(n_bits_broken_list, avg_pssl_elo_list, label='ELO', marker='x', color='green')
axs[1].plot(n_bits_broken_list, avg_pssl_ga_list, label='GA', marker='x', color='purple')
axs[1].plot(n_bits_broken_list, avg_pssl_bpso_list, label='BPSO', marker='x', color='cyan')
axs[1].set_title('Main Beam PSSL')
axs[1].set_xlabel('Proportion of Stuck Bits')
axs[1].set_ylabel('PSSL [dB]')
axs[1].legend()
axs[1].grid()
axs[2].plot(n_bits_broken_list, avg_pbp_broken_list, label='Broken', marker='x', color='red')
axs[2].plot(n_bits_broken_list, avg_pbp_elo_list, label='ELO', marker='x', color='green')
axs[2].plot(n_bits_broken_list, avg_pbp_ga_list, label='GA', marker='x', color='purple')
axs[2].plot(n_bits_broken_list, avg_pbp_bpso_list, label='BPSO', marker='x', color='cyan')
axs[2].set_title('PBP')
axs[2].set_xlabel('Proportion of Stuck Bits')
axs[2].set_ylabel('PBP [dB]')
axs[2].legend()
axs[2].grid()
# Save the figure
fig.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.show()
'''