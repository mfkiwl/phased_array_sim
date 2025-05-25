import numpy as np
from utils.closest_int import closest_integer

def amplitude_to_dB(amplitude):
    if amplitude == 0:
        amplitude = 1e-10
    return 20 * np.log10(amplitude)

def amplitude_to_dB_list(amplitude_list):
    amplitude_list = np.where(amplitude_list == 0, 1e-10, amplitude_list)  # avoid log(0)
    dB_list = 20 * np.log10(amplitude_list)
    return dB_list

def dB_to_amplitude(dB):
    return 10 ** (dB / 20)
 
def power_to_dB(power):
    # if power is 0, change it to 1e-10 to avoid log(0)
    if power == 0:
        power = 1e-10
    return 10 * np.log10(power)

def power_to_dB_list(power_list):
    power_list = np.where(power_list == 0, 1e-10, power_list)  # avoid log(0)
    dB_list = 10 * np.log10(power_list)
    return dB_list

def dB_to_power(dB):
    return 10 ** (dB / 10)

def reduce_phase(phase):
    '''
    Reduce phase to the range [0, 2pi]
    input: phase - phase in radians
    output: reduced_phase - reduced phase in radians
    '''
    reduced_phase = phase % (2 * np.pi)
    return reduced_phase

def quantise_phase(phase, n_bits):
    '''
    Quantise phase to n_bits.
    input: phase - phase in radians
           n_bits - number of bits
    output: quantised_phase - quantised phase in radians
    '''
    phase = reduce_phase(phase)  # Reduce phase to [0, 2pi]
    # Convert phase to integer
    integer = int(round(phase / (2 * np.pi) * (2 ** n_bits)))
    # Convert integer to phase
    quantised_phase = integer * (2 * np.pi / (2 ** n_bits))
    return quantised_phase

def quantise_phase_list(phase_list, n_bits):
    '''
    Quantise a list of phases to n_bits.
    input: phase_list - list of phases in radians
           n_bits - number of bits
    output: quantised_phase_list - list of quantised phases in radians
    '''
    quantised_phase_list = []
    for phase in phase_list:
        quantised_phase = quantise_phase(phase, n_bits)
        quantised_phase_list.append(quantised_phase)
    return quantised_phase_list

def bit_list_to_integer(bits):
    '''
    Convert bits to integer.
    input: bits - list of bits (0 or 1), higher index means higher significance.
    output: integer - integer value
    '''
    integer = 0
    for i, bit in enumerate(bits):
        integer += bit * (2 ** i)
    return integer

def integer_to_bit_list(integer, n_bits):
    '''
    Convert integer to bits.
    input: integer - integer value
           n_bits - number of bits
    output: bits - list of bits (0 or 1), higher index means higher significance.
    '''
    bits = [(integer >> i) & 1 for i in range(n_bits)]
    return bits

def bit_list_to_phase(bits):
    '''
    Convert bits to phase.
    input: bits - list of bits (0 or 1), higher index means higher significance.
    output: phase - phase in radians
    '''
    # Convert bits to integer
    integer = bit_list_to_integer(bits)
    n = len(bits)
    phase = integer * (2 * np.pi / (2 ** n))
    return phase

def phase_to_bit_list(phase, n_bits):
    '''
    Convert phase to bits.
    input: phase - phase in radians
           n_bits - number of bits
    output: bits - list of bits (0 or 1), higher index means higher significance.
    '''
    phase = reduce_phase(phase)  # Reduce phase to [0, 2pi]
    integer = int(round(phase / (2 * np.pi) * (2 ** n_bits)))
    # convert integer to bits
    bits = integer_to_bit_list(integer, n_bits)
    return bits

def phase_list_to_bit_array(phase_list, n_bits):
    '''
    Convert a list of phases to a bit array.
    input: phase_list - list of phases in radians
           n_bits - number of bits
    output: bit_array - numpy array of bits (0 or 1), higher index means higher significance.
    '''
    bit_array = np.zeros((len(phase_list), n_bits), dtype=int)
    for i, phase in enumerate(phase_list):
        bit_array[i] = phase_to_bit_list(phase, n_bits)
    return bit_array

def bit_array_to_phase_list(bit_array):
    '''
    Convert a bit array to a list of phases.
    input: bit_array - numpy array of bits (0 or 1), higher index means higher significance.
    output: phase_list - list of phases in radians
    '''
    phase_list = []
    for bits in bit_array:
        phase = bit_list_to_phase(bits)
        phase_list.append(phase)
    return phase_list

def ideal_phase_list(n_elements, steering_angle=0):
    '''
    Generate ideal phase list for a given scan angle.
    input: n_elements - number of elements in the array
           steering_angle - scan angle in radians
    output: phase_list - list of phases in radians
    '''
    # Calculate ideal phase list
    phase_list = [-np.pi * i * np.sin(steering_angle) for i in range(n_elements)]
    return phase_list

def phase_list_to_af_list(phase_list, scan_angles=np.deg2rad(np.arange(0, 361, 1))):
    '''
    Convert a list of phases to a list of antenna factors.
    input: phase_list - list of phases in radians
           n_bits - number of bits
           scan_angles - list of scan angles in degrees
    output: AF_list - list of antenna factors in dB
    '''
    af_list = np.zeros_like(scan_angles, dtype=complex)
    for k, th in enumerate(scan_angles):
        # Calculate antenna factor
        af = 0j
        for i, phase in enumerate(phase_list):
            af += np.exp(1j * (phase + np.pi * i * np.sin(th)))
        af_list[k] = af
    return np.abs(af_list) / len(phase_list)  # Normalise the antenna factor

def AF_planar_linear_beamform(M, N, theta_grid, phi_grid, phase_list):
    # beamforming in the x direction, on theta??
    # Calculate Array Factor (AF)
    AF = np.zeros_like(theta_grid, dtype=complex)

    for m in range(M):
        for n in range(N):
            # Phase shift for element (m, n)
            phase = (
                m * np.pi * np.sin(theta_grid) * np.cos(phi_grid) + 
                n * np.pi * np.sin(theta_grid) * np.sin(phi_grid) +
                phase_list[m]  # Add the ideal phase for this element
            )
            AF += np.exp(1j * phase)

    # Magnitude and normalization
    AF_mag = np.abs(AF) / (N * M)  # Normalize by number of elements
    return AF_mag


## Breaking and fixing the array
def random_select_broken_bits(n_elements, n_bits, n_broken_bits, mode=0):
    '''
    Randomly select broken bits. 
    The output will be 3 lists of length n_broken_bits:
    broken_elements - the element index of the broken bits
    broken_bits - the bit index of the broken bits
    broken_values - the value of the broken bits (0 or 1)
    '''
    if mode == 0:
        broken_ids = np.random.choice(n_elements * n_bits, n_broken_bits, replace=False)
        # sort the broken ids
        broken_ids = np.sort(broken_ids)
        broken_elements = broken_ids // n_bits  # Element index
        broken_bits = broken_ids % n_bits
        broken_values = [np.random.choice([0, 1]) for _ in range(n_broken_bits)]
    elif mode == 1:
        # select 1 bit from each element
        broken_elements = np.random.choice(n_elements, n_broken_bits, replace=False)
        broken_elements = np.sort(broken_elements)  # sort the elements
        #broken_bits = np.random.choice(n_bits, n_broken_bits, replace=True)
        # choose the most significant bits
        broken_bits = np.full(n_broken_bits, n_bits - 1)
        broken_values = [np.random.choice([0, 1]) for _ in range(n_broken_bits)]
    return broken_elements, broken_bits, broken_values

def break_bit_array(bit_array, broken_elements, broken_bits, broken_values):
    '''
    Break the bits in the bit array according to the broken elements and bits.
    input: bit_array - numpy array of bits (0 or 1), higher index means higher significance.
           broken_elements - list of broken element indices
           broken_bits - list of broken bit indices
           broken_values - list of broken values (0 or 1)
    output: bit_array - numpy array of bits (0 or 1), higher index means higher significance.
    '''
    broken_bit_array = np.copy(bit_array)
    n = len(broken_elements)
    for el, bit, val in zip(broken_elements, broken_bits, broken_values):
        broken_bit_array[el][bit] = val
    return broken_bit_array

def el_by_el_optim(unbroken_bit_array, broken_elements, broken_bits, broken_values):
    '''
    Use the "closest integer" method to optimise the bit array element by element.
    '''
    # sort the broken elements
    broken_elements = np.array(broken_elements)
    broken_bits = np.array(broken_bits)
    broken_values = np.array(broken_values)
    #sorted_indices = np.argsort(broken_elements)
    #broken_elements = broken_elements[sorted_indices]
    #broken_bits = broken_bits[sorted_indices]
    #broken_values = broken_values[sorted_indices]

    n_bits = len(unbroken_bit_array[0])
    optim_bit_array = np.copy(unbroken_bit_array)
    for i in range(len(unbroken_bit_array)):
        # collect all the broken bits and values
        broken_bits_i = broken_bits[broken_elements == i]
        broken_values_i = broken_values[broken_elements == i]
        # if there are no broken bits, continue
        if broken_bits_i.size == 0:
            continue
        # if there are broken bits:
        # use the closest_integer function to find the closest integer to the desired value
        # convert the current row into a decimal.
        decimal_value = bit_list_to_integer(unbroken_bit_array[i])
        closest = closest_integer(decimal_value, len(unbroken_bit_array[0]), broken_bits_i, broken_values_i)
        # convert the closest integer into a binary number
        binary_word = integer_to_bit_list(closest, n_bits)
        # update the bit array with the new value
        optim_bit_array[i] = binary_word
    return optim_bit_array


## Evaluation utils
def af_to_distribution(af_list):
    n = len(af_list)
    # convert the af_list such that the total sum is 1
    output = np.array(af_list) ** 2
    output = output / np.sum(output)
    return output

def kl_divergence(af_ideal, af_actual):
    ideal_dist = af_to_distribution(af_ideal)
    actual_dist = af_to_distribution(af_actual)
    # avoid division by zero
    ideal_dist = np.where(ideal_dist == 0, 1e-10, ideal_dist)
    actual_dist = np.where(actual_dist == 0, 1e-10, actual_dist)
    # calculate the Kullback-Leibler divergence
    kl = np.sum(ideal_dist * np.log(ideal_dist / actual_dist))
    return kl

def MSE(af_ideal, af_actual):
    # Calculate the Mean Squared Error between the ideal and actual antenna factors
    mse = np.mean((af_ideal - af_actual) ** 2)
    return mse

def normalised_SE(af_ideal, af_actual):
    # Calculate the normalised Mean Squared Error between the ideal and actual antenna factors
    se = np.sum((af_ideal - af_actual) ** 2)
    norm_se = se / np.sum(af_ideal ** 2)
    return norm_se

def PBP(af_list, scan_rads, steering_angle_rad):
    mbp = af_list[np.argmin(np.abs(scan_rads - steering_angle_rad))]
    return 20* np.log10(mbp)

def PSSL(af_list, scan_rads, steering_angle_rad, beamwidth_rad):
    # Calculate the peak side lobe level
    psll_linear = np.max(af_list[np.abs(scan_rads - steering_angle_rad) > beamwidth_rad])
    psll_dB = 20 * np.log10(psll_linear)
    main_beam_dB = PBP(af_list, scan_rads, steering_angle_rad)
    return psll_dB - main_beam_dB

def ISLL(af_list, scan_rads, steering_angle_rad, beamwidth_rad):
    """
    Calculate the Integrated Side Lobe Level (ISLL) for a given antenna factor list.
    
    Parameters:
    af_list (np.ndarray): Antenna factor values.
    scan_rads (np.ndarray): Scan angles in radians.
    steering_angle_rad (float): Steering angle in radians.
    beamwidth_rad (float): Beamwidth in radians.
    
    Returns:
    float: ISLL in dB.
    """
    # Calculate the step size in radians
    rad_step = scan_rads[1] - scan_rads[0]
    
    # Calculate the integrated side lobe level
    isl = np.sum(af_list[np.abs(scan_rads - steering_angle_rad) > beamwidth_rad]**2) * rad_step

    # calculate the integrated main beam level
    integrated_main_beam = np.sum(af_list[np.abs(scan_rads - steering_angle_rad) <= beamwidth_rad]**2) * rad_step
    
    return 10 * np.log10(isl / integrated_main_beam)
