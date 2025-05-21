import random
import antenna_utils as au
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# === CONFIGURATION ===
POPULATION_SIZE = 100
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
GENERATIONS = 200
ELITE_FRACTION = 0.10  # Top 10%
STOP_CRITERION = 0.001  # Stop if fitness improvement is less than this
STAGNATION_LIMIT = 20  # Number of generations to wait before stopping
GENE_LENGTH = 32  # Length of binary string

n_elements = 8  # Number of elements in the array
n_bits = 4  # Number of bits per element
n_stuck = 3  # Number of bits stuck (0 or 1)
broken_elements, broken_bits, broken_values = np.array([0, 3, 5]), np.array([2, 1, 3]), np.array([1, 1, 1])
steering_angle_deg = 0  # Steering angle in degrees
steering_angle_rad = np.radians(steering_angle_deg)  # Convert to radians
scan_deg = np.arange(-90, 91)  # Scan angles from -90° to +90°
scan_rad = np.radians(scan_deg)

ideal_phase_list = au.ideal_phase_list(n_elements, steering_angle_rad)
quantised_phase_list = au.quantise_phase_list(ideal_phase_list, n_bits)
af_ideal = au.phase_list_to_af_list(ideal_phase_list, scan_rad)
af_quant = au.phase_list_to_af_list(quantised_phase_list, scan_rad)
ideal_bit_array = au.phase_list_to_bit_array(ideal_phase_list, n_bits)
broken_bit_array = au.break_bit_array(ideal_bit_array, broken_elements, broken_bits, broken_values)
broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
optim_bit_array = au.el_by_el_optim(ideal_bit_array, broken_elements, broken_bits, broken_values)
optim_phase_list = au.bit_array_to_phase_list(optim_bit_array)
af_broke = au.phase_list_to_af_list(broken_phase_list, scan_rad)
af_optim = au.phase_list_to_af_list(optim_phase_list, scan_rad)

# === CONVERT STRING TO BIT ARRAY ===
def string_to_bit_array(binary_string, n_elements=n_elements, n_bits=n_bits):
    """Convert an M length binary string to an N by k bit array."""
    if len(binary_string) != n_elements * n_bits:
        raise ValueError(f"Binary string length must be {n_elements * n_bits}.")
    return np.array([int(bit) for bit in binary_string]).reshape((n_elements, n_bits))

def bit_array_to_string(bit_array):
    """Convert an N by k bit array to a binary string."""
    return ''.join(str(bit) for row in bit_array for bit in row)

# === FITNESS FUNCTION ===
def fitness_function(binary_string):
    """minimize the difference between the ideal and broken array factors."""
    bit_array = string_to_bit_array(binary_string)
    broken_bit_array = au.break_bit_array(bit_array, broken_elements, broken_bits, broken_values)
    broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
    af_broke = au.phase_list_to_af_list(broken_phase_list, scan_rad)
    # normalised MSE fitness function
    norm_se = au.normalised_SE(af_ideal, af_broke)
    return norm_se

# === GENETIC OPERATORS ===
def generate_individual():
    return ''.join(random.choice('01') for _ in range(GENE_LENGTH))

def mutate(individual):
    return ''.join(
        bit if random.random() > MUTATION_RATE else '1' if bit == '0' else '0'
        for bit in individual
    )

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        # select indices to swap the bits
        swap_locations = random.sample(range(GENE_LENGTH), 10)
        child1 = ''.join(
            parent1[i] if i not in swap_locations else parent2[i]
            for i in range(GENE_LENGTH)
        )
        child2 = ''.join(
            parent2[i] if i not in swap_locations else parent1[i]
            for i in range(GENE_LENGTH)
        )
        return child1, child2
    return parent1, parent2

# === MAIN GA LOOP ===
def genetic_algorithm():
    population = [generate_individual() for _ in range(POPULATION_SIZE)]

    stag_count = 0
    best_fitness = float('inf')
    for gen in range(GENERATIONS):
        fitnesses = [fitness_function(ind) for ind in population]
        current_best_fitness = min(fitnesses)
        
        # check stop criterion
        if abs(current_best_fitness - best_fitness) < STOP_CRITERION:
            stag_count += 1
        else:
            stag_count = 0
            best_fitness = current_best_fitness

        # check stagnation
        if stag_count >= STAGNATION_LIMIT:
            print(f"Stopping early at generation {gen} due to stagnation.")
            break

        # Select top 10% elite
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1])
        elite_size = max(1, int(ELITE_FRACTION * POPULATION_SIZE))
        elite = [ind for ind, _ in sorted_pop[:elite_size]]
        
        next_generation = elite.copy()
        
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(mutate(child2))

        population = next_generation

        # Optional: print best result each generation
        best = min(population, key=fitness_function)
        print(f"Generation {gen}: Best = {best} (Fitness = {fitness_function(best)})")

    # Final best
    best = min(population, key=fitness_function)
    print(f"\nBest solution: {best} (Fitness = {fitness_function(best)})")
    return best

# Run the GA
if __name__ == "__main__":
    best = genetic_algorithm()
    best_bit_array = string_to_bit_array(best)
    best_broken_bit_array = au.break_bit_array(best_bit_array, broken_elements, broken_bits, broken_values)
    best_broken_phase_list = au.bit_array_to_phase_list(best_broken_bit_array)
    best_af_broke = au.phase_list_to_af_list(best_broken_phase_list, scan_rad)
    best_af_broke_dB = au.amplitude_to_dB_list(best_af_broke)

    af_ideal_dB = au.amplitude_to_dB_list(af_ideal)
    af_quant_dB = au.amplitude_to_dB_list(af_quant)
    af_broke_dB = au.amplitude_to_dB_list(af_broke)
    af_optim_dB = au.amplitude_to_dB_list(af_optim)
    best_af_broke_dB = au.amplitude_to_dB_list(best_af_broke)

    loss_quant = au.normalised_SE(af_ideal, af_quant)
    loss_broken = au.normalised_SE(af_ideal, af_broke)
    loss_optim = au.normalised_SE(af_ideal, af_optim)
    loss_ga = au.normalised_SE(af_ideal, best_af_broke)
    loss_text = f"L_quant: {-loss_quant:.2f} dB\nL_broken: {-loss_broken:.2f} dB\nL_optim: {-loss_optim:.2f} dB\nL_ga: {-loss_ga:.2f} dB"

    # Plot the best result
    # === PLOTTING ===
    fig = plt.figure(figsize=(6, 9))
    gs = GridSpec(2, 1, height_ratios=[6, 3])  # 6 for ax, 3 for ax2
    # polar plot
    ax = fig.add_subplot(gs[0], polar=True)
    # add the plots
    line0 = ax.plot(scan_rad, af_ideal, lw=1, color='k', label="ideal array factor")[0]
    line1 = ax.plot(scan_rad, af_quant, lw=1, color='b', label="quantised phases")[0]
    line2 = ax.plot(scan_rad, af_broke, lw=1, color='r', label="broken bit array")[0]
    line_optim = ax.plot(scan_rad, af_optim, lw=1, color='g', label="optimised bit array")[0]
    line_ga = ax.plot(scan_rad, best_af_broke, lw=1, color='m', label="GA optimised bit array")[0]
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
    steer_line = ax.plot(np.full_like(r, steering_angle_rad), r, color='k', lw=1, ls='--', label="steering angle")[0]
    ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)
    ax.text(0.05, 0.35, loss_text, transform=ax.transAxes, fontsize=12, color='k')
    

    # cartesian dB plot
    dB_list = np.linspace(-50, 0, 100)  # dB scale
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-90, 90)
    ax2.set_ylim(-50, 0)
    ax2.set_xlabel("Angle ($^\\circ$)")
    ax2.set_ylabel("Relative Gain (dB)")
    ax2.grid()
    dB0 = ax2.plot(scan_deg, af_ideal_dB, lw=1, color='k', label="ideal array factor")[0]
    dB1 = ax2.plot(scan_deg, af_quant_dB, lw=1, color='b', label="quantised phases")[0]
    dB2 = ax2.plot(scan_deg, af_broke_dB, lw=1, color='r', label="broken bit array")[0]
    dB_optim = ax2.plot(scan_deg, af_optim_dB, lw=1, color='g', label="optimised bit array")[0]
    dB_ga = ax2.plot(scan_deg, best_af_broke_dB, lw=1, color='m', label="GA optimised bit array")[0]
    steer_line2 = ax2.plot(np.full_like(dB_list, steering_angle_deg), dB_list, color='k', lw=1, ls='--')[0]

    # use the broken array elements and bit positions to name the plot
    broken_elements_str = ', '.join(map(str, broken_elements))
    broken_bits_str = ', '.join(map(str, broken_bits))
    plot_name = f"{broken_elements_str} {broken_bits_str}"
    plt.savefig("visuals\\ga_optim\\"+plot_name, dpi=300, bbox_inches='tight')
    plt.show()