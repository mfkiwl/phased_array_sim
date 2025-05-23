import antenna_utils as au
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.gridspec import GridSpec
import ga
import bpso
import random
import pandas as pd

# to measure the time taken to run each optimization algorithm. 
# --- Parameters ---
n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
#n_stuck = 3  # Number of bits stuck (0 or 1)
scan_deg = np.arange(-90, 91)  # Scan angles from -90° to +90° in degrees
scan_rad = np.radians(scan_deg)  # Scan angles from -90° to +90° in radians
beamwidth_deg = 14.5 # in degrees # This is arcsin(2/N)
beamwidth_rad = np.radians(beamwidth_deg)  # Beamwidth in radians

def average_times(n_elements=8, n_bits=4, n_stuck=3, trials=100):
    # for every trial, randomly select n_broken_bits bits to be stuck, then calculate the average nse_ over all steering angles
    #total_nse_quantised = 0
    #total_mb_nse_quantised = 0
    total_time_elo = 0
    total_time_ga = 0
    total_time_bpso = 0
    #total_mb_nse_bpso = 0
    for i in range(trials):
        broken_elements, broken_bits, broken_values = au.random_select_broken_bits(n_elements, n_bits, n_stuck, mode=0)
        #nse_quant_list = [] # total nse
        time_elo_list = []
        time_ga_list = []
        time_bpso_list = []
        #mb_nse_quant_list = []
        #mb_nse_broken_list = [] # main beam nse
        #mb_nse_elo_list = []
        for steering_angle_deg in np.arange(-60, 61, 30):
            steering_angle_rad = np.radians(steering_angle_deg)
            # Calculate all the array factors and nse_s
            ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
            #quantised_phase_list = au.quantise_phase_list(ideal_phase_list, n_bits)
            af_ideal = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
            #af_quant = au.phase_list_to_af_list(quantised_phase_list, scan_rad)
            ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
            #broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements, broken_bits, broken_values)
            #broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)

            # ELO 
            # start measuring time
            start_time = time.time()
            # run the ELO algorithm
            elo_bit_array = au.el_by_el_optim(ideal_bit_array, broken_elements, broken_bits, broken_values)
            # end measuring time
            end_time = time.time()
            # calculate the time taken
            time_elo = end_time - start_time
            # convert time to ms
            time_elo = time_elo * 1000

            #elo_phase_list = au.bit_array_to_phase_list(elo_bit_array)
            #af_broken = au.phase_list_to_af_list(broken_phase_list, scan_rad)
            #af_elo = au.phase_list_to_af_list(elo_phase_list, scan_rad)

            # GA
            # start measuring time
            start_time = time.time()
            # run the GA algorithm
            best = ga.genetic_algorithm(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values)
            # end measuring time
            end_time = time.time()
            # calculate the time taken
            time_ga = end_time - start_time
            # convert time to ms
            time_ga = time_ga * 1000

            #best_bit_array = ga.string_to_bit_array(best)
            #ga_bit_array = au.break_bit_array(best_bit_array, broken_elements, broken_bits, broken_values)
            #ga_phase_list = au.bit_array_to_phase_list(ga_bit_array)
            #af_ga = au.phase_list_to_af_list(ga_phase_list, scan_rad)

            #BPSO
            # start measuring time
            start_time = time.time()
            # run the BPSO algorithm
            best_bpso, _ = bpso.binary_pso(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values)
            # end measuring time
            end_time = time.time()
            # calculate the time taken
            time_bpso = end_time - start_time
            # convert time to ms
            time_bpso = time_bpso * 1000

            #best_bpso_bit_array = bpso.list_to_bit_array(best_bpso)
            #bpso_bit_array = au.break_bit_array(best_bpso_bit_array, broken_elements, broken_bits, broken_values)
            #bpso_phase_list = au.bit_array_to_phase_list(bpso_bit_array)
            #af_bpso = au.phase_list_to_af_list(bpso_phase_list, scan_rad)

            #nse_quant_list.append(nse_quant)
            
            time_elo_list.append(time_elo)
            time_ga_list.append(time_ga)
            time_bpso_list.append(time_bpso)

            print(f"Trial {i+1}/{trials}, Steering Angle {steering_angle_deg}° complete")

        total_time_elo += np.mean(time_elo_list)
        total_time_ga += np.mean(time_ga_list)
        total_time_bpso += np.mean(time_bpso_list)

        print(f"Trial {i+1}/{trials} complete")

    # Calculate the average KL  and loss
    #avg_nse_quantised = total_nse_quantised / trials
    #avg_mb_nse_quantised = total_mb_nse_quantised / trials
    avg_time_elo = total_time_elo / trials
    avg_time_ga = total_time_ga / trials
    avg_time_bpso = total_time_bpso / trials

    return avg_time_elo, avg_time_ga, avg_time_bpso

if __name__ == "__main__":
    # Run the average_times function and print the results
    avg_time_elo, avg_time_ga, avg_time_bpso = average_times(trials=4)
    print(f"Average time taken by ELO: {avg_time_elo:.2f} ms")
    print(f"Average time taken by GA: {avg_time_ga:.2f} ms")
    print(f"Average time taken by BPSO: {avg_time_bpso:.2f} ms")
    # Save the results to a CSV file
    data = {
        'Algorithm': ['ELO', 'GA', 'BPSO'],
        'Average Time (ms)': [avg_time_elo, avg_time_ga, avg_time_bpso]
    }
    df = pd.DataFrame(data)
    df.to_csv('average_times.csv', index=False)
    print("Average times saved to average_times.csv")