import random
import utils.antenna_utils as au
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# === CONFIGURATION ===
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 200
ELITE_FRACTION = 0.10  # Top 10%
STOP_CRITERION = 0.001  # Stop if fitness improvement is less than this
STAGNATION_LIMIT = 12  # Number of generations to wait before stopping
GENE_LENGTH = 32  # Length of binary string

n_elements = 8  # Number of elements in the array
beamwidth_deg = 14.5  # in degrees
beamwidth_rad = np.radians(beamwidth_deg)  # Convert to radians
n_bits = 4  # Number of bits per element


# === CONVERT STRING TO BIT ARRAY ===

def bit_array_to_list(bit_array):
    """Flatten an N by k bit array to a np array of bits."""
    return bit_array.flatten()

def broken_indices(broken_elements, broken_bits):
    """Convert broken elements and bits to a list of indices."""
    return [broken_elements[i] * n_bits + broken_bits[i] for i in range(len(broken_elements))]

def inject_broken_bits(bit_list, broken_indices_list, broken_values):
    """Inject broken bits into the bit list."""
    output_list = bit_list.copy()
    for i in range(len(broken_indices_list)):
        output_list = np.insert(output_list, broken_indices_list[i], broken_values[i])
    return output_list

def list_to_bit_array(bit_list):
    """Convert a list of bits to an N by k bit array."""
    return bit_list.reshape((n_elements, n_bits))

# === FITNESS FUNCTION ===
def fitness_function(binary_string, af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_indices_list, broken_values):
    """minimize the difference between the ideal and broken array factors."""
    broken_bit_list = inject_broken_bits(binary_string, broken_indices_list, broken_values)
    #print(broken_bit_list)
    broken_bit_array = list_to_bit_array(broken_bit_list)
    #broken_bit_array = au.break_bit_array(bit_array, broken_elements, broken_bits, broken_values)
    broken_phase_list = au.bit_array_to_phase_list(broken_bit_array)
    af_broke = au.phase_list_to_af_list(broken_phase_list, scan_rad)
    # normalised MSE fitness function
    norm_se = au.normalised_SE(af_ideal, af_broke)
    #peak_beam_power = au.PBP(af_broke, scan_rad, steering_angle_rad)
    #peak_side_lobe_level = au.PSSL(af_broke, scan_rad, steering_angle_rad, beamwidth_rad)
    #return  - 0.5* peak_beam_power + 2*norm_se + 0.1* peak_side_lobe_level
    return norm_se

# === GENETIC OPERATORS ===
def generate_individual(gene_length):
    #return ''.join(random.choice('01') for _ in range(GENE_LENGTH))
    #print(np.random.randint(0, 2, GENE_LENGTH))
    return np.random.randint(0, 2, gene_length)

def mutate(individual, gene_length):
    #return ''.join(
    #    bit if random.random() > MUTATION_RATE else '1' if bit == '0' else '0'
    #    for bit in individual
    #)
    new_individual = individual.copy()
    #print(new_individual)
    for i in range(gene_length):
        if random.random() < MUTATION_RATE:
            new_individual[i] = 1 - new_individual[i]
    return new_individual

def crossover(parent1, parent2, gene_length):
    n_swap = int(gene_length / 3 )
    if random.random() < CROSSOVER_RATE:
        # select indices to swap the bits
        swap_locations = random.sample(range(gene_length), n_swap)
        child1 = np.array([
            parent1[i] if i not in swap_locations else parent2[i]
            for i in range(gene_length)
        ])
        child2 = np.array([
            parent2[i] if i not in swap_locations else parent1[i]
            for i in range(gene_length)
        ])
        #print(child1)
        return child1, child2
    return parent1, parent2

# === MAIN GA LOOP ===
def genetic_algorithm(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values):
    gene_length = n_elements * n_bits - len(broken_elements)
    broken_indices_list = broken_indices(broken_elements, broken_bits)
    population = [generate_individual(gene_length) for _ in range(POPULATION_SIZE)]

    stag_count = 0
    best_fitness = float('inf')
    for gen in range(GENERATIONS):
        fitnesses = [fitness_function(ind, af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_indices_list, broken_values) for ind in population]
        current_best_fitness = min(fitnesses)
        
        # check stop criterion
        if abs(current_best_fitness - best_fitness) < STOP_CRITERION:
            stag_count += 1
        else:
            stag_count = 0
            best_fitness = current_best_fitness

        # check stagnation
        if stag_count >= STAGNATION_LIMIT:
            #print(f"Stopping early at generation {gen} due to stagnation.")
            break

        # Select top 10% elite
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1])
        elite_size = max(1, int(ELITE_FRACTION * POPULATION_SIZE))
        elite = [ind for ind, _ in sorted_pop[:elite_size]]
        
        next_generation = elite.copy()
        
        while len(next_generation) < POPULATION_SIZE:
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child1, child2 = crossover(parent1, parent2, gene_length)
            next_generation.append(mutate(child1, gene_length))
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(mutate(child2, gene_length))

        population = next_generation

        # Optional: print best result each generation
        #best = min(population, key=lambda val: fitness_function(val, af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values))
        #print(f"Generation {gen}: Best = {best} (Fitness = {fitness_function(best, af_ideal, scan_rad, steering_angle_rad, beamwidth_rad)})")

    # Final best
    best = min(population, key=lambda val: fitness_function(val, af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_indices_list, broken_values))
    #print(f"\nBest solution: {best} (Fitness = {fitness_function(best, af_ideal, scan_rad, steering_angle_rad, beamwidth_rad)})")

    return inject_broken_bits(best, broken_indices_list, broken_values)

# Run the GA
if __name__ == "__main__":
    n_stuck = 5  # Number of bits stuck (0 or 1)
    #broken_elements, broken_bits, broken_values = np.array([0, 3, 5]), np.array([2, 1, 3]), np.array([1, 1, 1])
    broken_elements, broken_bits, broken_values = np.array([0, 2, 3, 5, 6]), np.array([2, 1, 3, 3, 1]), np.array([1, 1, 1, 1, 1])
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

    best = genetic_algorithm(af_ideal, scan_rad, steering_angle_rad, beamwidth_rad, broken_elements, broken_bits, broken_values)
    best_bit_array = list_to_bit_array(best)
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