'''
Created by Alexander Li, 2025-4-9
This script simulates the beamforming of a uniform linear array (ULA) with 4-bit quantized phase shifters.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from correction import valid_integers, modular_distance, closest_integer

# --- Array and beamforming setup ---
N = 8                         # Number of elements in ULA
n_bits = 4                    # Phase shifter resolution (4 bits)
phase_levels = 2 ** n_bits   # 16 levels
d = 0.5                       # Element spacing in wavelengths
wavelength = 1.0            # unit wavelength
k = 2 * np.pi / wavelength    # Wavenumber
n_stuck = 5               # Number of stuck bits

# --- Bit breaking setup ---
# Randomly select 3 elements to break
#broken_elements = [1, 4, 5]
#broken_bits = [0, 1, 2]
#broken_values = [0, 1, 1]

def param_break_n_bits_random(n_broken_bits):
    broken_ids = np.random.choice(N * n_bits, n_broken_bits, replace=False)
    # sort the broken ids
    broken_ids = np.sort(broken_ids)
    broken_elements = broken_ids // n_bits  # Element index
    broken_bits = broken_ids % n_bits
    broken_values = [np.random.choice([0, 1]) for _ in range(n_broken_bits)]
    return broken_elements, broken_bits, broken_values

broken_elements, broken_bits, broken_values = param_break_n_bits_random(n_stuck)
'''
broken_elements = np.array(
    [1]
)
broken_bits = np.array(
    [n_bits - 1]
)
broken_values = np.array(
    [0]
)
'''

def amplitude_to_dB(amplitude):
    """
    Convert amplitude to decibels (dB).
    Inputs: 
        amplitude: Amplitude value
    Outputs:
        Amplitude in dB
    """
    return 20 * np.log10(amplitude)

def quantize_phase(phase):
    """
    Quantize a phase value to nearest 4-bit level.
    Inputs: 
        phase: Phase in radians
    Outputs:
        Quantized phase in radians
    """
    step = 2 * np.pi / phase_levels
    return np.round(phase / step) * step

def steering_bit_array(steering_angle_deg):
    """
    Compute the bit array for a given steering angle.
    Inputs:
        steering_angle_deg: Steering angle in degrees
    Outputs:
        N x n_bits array of 1s and 0s representing a 4-bit binary number for each element
    """
    steering_angle_rad = np.radians(steering_angle_deg)
    # Ideal phase shift for steering
    phase_shifts_rad = -k * d * np.arange(N) * np.sin(steering_angle_rad)
    # Quantize phases
    quantized_phases_rad = quantize_phase(phase_shifts_rad)
    # Convert to binary representation
    bit_array = np.zeros((N, n_bits), dtype=int)
    for n in range(N):
        decimal_value = int(quantized_phases_rad[n] / (2 * np.pi) * phase_levels) % phase_levels
        #print(f"Element {n}: Phase shift (rad) = {quantized_phases_rad[n]:.4f}, Decimal value = {decimal_value}")
        bit_array[n] = np.array(list(np.binary_repr(decimal_value, width=n_bits)), dtype=int)
    return bit_array

def break_bit_array(bit_array, broken_elements=broken_elements, broken_bits=broken_bits, broken_values=broken_values):
    """
    Apply specific fixed bits to several locations in the array.
    Inputs:
        bit_array: N x n_bits array of 1s and 0s representing a 4-bit binary number for each element
    Outputs:
        N x n_bits array of 1s and 0s representing a 4-bit binary number for each element, with specific bits broken
    """
    # break 3 random bits in the array
    broken_bit_array = np.copy(bit_array)
    
    for element, bit, value in zip(broken_elements, broken_bits, broken_values):
        # Randomly select 3 bits to break
        broken_bit_array[element, bit] = value
    return broken_bit_array

def optimise_bit_array(bit_array, broken_elements=broken_elements, broken_bits=broken_bits, broken_values=broken_values):
    # sort the broken elements
    broken_elements = np.array(broken_elements)
    broken_bits = np.array(broken_bits)
    broken_values = np.array(broken_values)
    sorted_indices = np.argsort(broken_elements)
    broken_elements = broken_elements[sorted_indices]
    broken_bits = broken_bits[sorted_indices]
    broken_values = broken_values[sorted_indices]
    # for every element
    for i in range(len(bit_array)):
        # collect all the broken bits and values
        broken_bits_i = broken_bits[broken_elements == i]
        broken_values_i = broken_values[broken_elements == i]
        # if there are no broken bits, continue
        if broken_bits_i.size == 0:
            continue
        # if there are broken bits:
        # use the closest_integer function to find the closest integer to the desired value
        # convert the current row into a decimal.
        decimal_value = int(''.join(map(str, bit_array[i]))[::-1], 2)
        closest = closest_integer(decimal_value, len(bit_array[0]), broken_bits_i, broken_values_i)
        # convert the closest integer into a binary number
        binary_value = np.array(list(np.binary_repr(closest, width=n_bits)), dtype=int)
        # update the bit array with the new value
        bit_array[i] = binary_value
    return bit_array


def array_factor_from_bits(scan_deg, bit_array):
    """
    Compute array factor for a given steering angle and quantized phase bits.
    Inputs:
        scan_deg: Scan angles in degrees
        bit_array: N x n_bits array of 1s and 0s representing a 4-bit binary number for each element
    Outputs:
        Array factor in linear scale
    """
    scan_rad = np.radians(scan_deg)
    # Convert bit array to phase shifts
    phase_shifts_rad = np.zeros(N)
    for n in range(N):
        # Convert binary to decimal
        decimal_value = int(''.join(map(str, bit_array[n])), 2)
        # Map to phase shift
        phase_shifts_rad[n] = (decimal_value / phase_levels) * 2 * np.pi
    # Compute AF
    af = np.zeros_like(scan_rad, dtype=complex)
    for n in range(N):
        af += np.exp(1j * (k * d * n * np.sin(scan_rad) + phase_shifts_rad[n]))
    return np.abs(af) / N

def array_factor_fully_formed(scan_deg, steering_angle_deg, quantise=True):
    """
    Compute array factor for a given steering angle and scan angle.
    Inputs:
        scan_deg: Scan angles in degrees
        steering_angle_deg: Steering angle in degrees
    Outputs:
        Array factor in linear scale
    """
    scan_rad = np.radians(scan_deg)
    steer_rad = np.radians(steering_angle_deg)
    # Ideal phase shift for steering
    phase_shifts_rad = -k * d * np.arange(N) * np.sin(steer_rad)
    # Quantize phases
    if quantise:
        phase_shifts_rad = quantize_phase(phase_shifts_rad)
    # Compute AF
    af = np.zeros_like(scan_rad, dtype=complex)
    for n in range(N):
        af += np.exp(1j * (k * d * n * np.sin(scan_rad) + phase_shifts_rad[n]))
    return np.abs(af) / N

def array_factor_fully_formed_list(steering_angle_deg, scan_degs, quantise=True):
    """
    Compute array factor for a given steering angle and scan angle.
    Inputs:
        scan_deg: Scan angles in degrees
        steering_angle_deg: Steering angle in degrees
    Outputs:
        Array factor in linear scale
    """
    # Compute the array factor for the given steering angle and scan angle
    output = []
    for scan_deg in scan_degs:
        # Compute the ideal array factor for the given scan angle
        af = array_factor_fully_formed(scan_deg, steering_angle_deg, quantise=quantise)
        output.append(af)
    return np.array(output)

ideal_af_lookup_dict = {}
for steering_angle_deg in range(-90, 91):
    ideal_af_lookup_dict[steering_angle_deg] = {}
    for scan_deg in range(-180, 181):
        # Compute the ideal array factor for the given scan angle
        ideal_af = array_factor_fully_formed(scan_deg, steering_angle_deg, quantise=False)
        # Store the ideal array factor in a dictionary with the scan angle as the key
        ideal_af_lookup_dict[steering_angle_deg][scan_deg] = float(ideal_af)


def ideal_af_lookup(steering_angle_deg, scan_deg_list):
    """
    Retrieve the ideal array factor from the lookup table.
    Inputs:
        steering_angle_deg: Steering angle in degrees
        scan_deg: numpy array of Scan angles in degrees
    Outputs:
        Ideal array factor in linear scale
    """
    # Retrieve the ideal array factor from the lookup table
    ideal_af = np.array([ideal_af_lookup_dict[steering_angle_deg][scan_deg] for scan_deg in scan_deg_list])
    return ideal_af

def pattern_cost_function(ideal_af, actual_af):
    """
    Compute the cost function for the pattern matching.
    Inputs:
        ideal_af: Ideal array factor
        actual_af: Actual array factor
    Outputs:
        Cost function value
    """
    # Compute the cost function as the mean squared error between ideal and actual AF
    return np.mean((ideal_af - actual_af) ** 2)

def break_n_bits_random(bit_array, n_broken_bits):
    """
    Break n_broken_bits in the bit array.
    Inputs:
        bit_array: N x n_bits array of 1s and 0s representing a 4-bit binary number for each element
        n_broken_bits: Number of bits to break
    Outputs:
        N x n_bits array of 1s and 0s representing a 4-bit binary number for each element, with specific bits broken
    """
    broken_ids = np.random.choice(N * n_bits, n_broken_bits, replace=False)
    broken_elements = broken_ids // n_bits  # Element index
    broken_bits = broken_ids % n_bits
    broken_values = [np.random.choice([0, 1]) for _ in range(n_broken_bits)]
    # Break n_broken_bits in the array
    broken_bit_array = np.copy(bit_array)
    for element, bit, value in zip(broken_elements, broken_bits, broken_values):
        broken_bit_array[element, bit] = value
    return broken_bit_array

def avg_cost_n_broken_bits(n_broken_bits, scan_range = [-90, 90], max_iter = 200):
    """
    Compute the average cost function for a given number of broken bits.
    Inputs:
        bit_array: N x n_bits array of 1s and 0s representing a 4-bit binary number for each element
        n_broken_bits: Number of broken bits
    Outputs:
        Average cost function value
    """
    costs = []
    for i in range(max_iter):
        # for every scanning angle, compute the cost function
        for steering_angle_deg in np.arange(scan_range[0], scan_range[1] + 1, 1):
            # Compute the ideal array factor
            ideal_af = array_factor_fully_formed(scan_deg, steering_angle_deg, quantise=False)
            # Compute the actual array factor with broken bits
            bit_array = steering_bit_array(steering_angle_deg)
            # Break some bits in the array
            bit_array = break_n_bits_random(bit_array, n_broken_bits)
            actual_af = array_factor_from_bits(scan_deg, bit_array)
            # Compute the cost function
            cost = pattern_cost_function(ideal_af, actual_af)
            costs.append(cost)
    # Return the average cost function value
    return np.mean(costs)
'''
n_broken_bits_array = np.arange(0, N*n_bits + 1)  # Number of broken bits from 0 to N*n_bits
avg_cost_array = np.zeros(N*n_bits + 1)  # Initialize the average cost array
for n_broken_bits in n_broken_bits_array:
    avg_cost_array[n_broken_bits] = avg_cost_n_broken_bits(n_broken_bits, scan_range=[-90, 90], max_iter=10)
#plot
plt.figure(figsize=(6, 4))
plt.plot(n_broken_bits_array, avg_cost_array, marker='x', color='k', linestyle='-')
plt.xlabel("Number of Broken Bits")
plt.ylabel("Average Cost Function")
plt.title("Average Cost Function vs Number of Broken Bits")
plt.grid()
plt.xlim(0, N*n_bits+1)
plt.ylim(0, np.max(avg_cost_array) * 1.1)
plt.savefig("avg_cost_function.png", dpi=300, bbox_inches='tight')
plt.show()
'''

'''
# plot a specific beam pattern, steering angle = 0
steering_angle_deg = 0
scan_deg = np.arange(-180, 181)  # Scan angles from -90° to +90°
scan_rad = np.radians(scan_deg)
af0 = ideal_af_lookup(steering_angle_deg, scan_deg)
af1 = array_factor_fully_formed(scan_deg, steering_angle_deg)
bit_array = steering_bit_array(steering_angle_deg)
# Break some bits in the array
bit_array = break_bit_array(bit_array)
af2 = array_factor_from_bits(scan_deg, bit_array)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.plot(scan_rad, amplitude_to_dB(af0), lw=1, color='k', label="ideal array factor")
ax.plot(scan_rad, amplitude_to_dB(af1), lw=1, color='b', label="quantised phases")
ax.plot(scan_rad, amplitude_to_dB(af2), lw=1, color='r', label="broken bit array")
# plot the fourier transform of the array factor
af0_FT = np.fft.fft(af0, n=1024)
af1_FT = np.fft.fft(af1, n=1024)
af2_FT = np.fft.fft(af2, n=1024)
# normalise the fourier transform to the maximum value
af0_FT = af0_FT / np.max(np.abs(af0_FT))
af1_FT = af1_FT / np.max(np.abs(af1_FT))
af2_FT = af2_FT / np.max(np.abs(af2_FT))
# convert everything to dB
af0_FT_dB = amplitude_to_dB(np.abs(af0_FT))
af1_FT_dB = amplitude_to_dB(np.abs(af1_FT))
af2_FT_dB = amplitude_to_dB(np.abs(af2_FT))
# plot the fourier transform in dB
ax.plot(np.linspace(-np.pi, np.pi, len(af0_FT)), af0_FT_dB, lw=1, color='k', label="ideal array factor FT")
ax.plot(np.linspace(-np.pi, np.pi, len(af1_FT)), af1_FT_dB, lw=1, color='b', label="quantised phases FT")
ax.plot(np.linspace(-np.pi, np.pi, len(af2_FT)), af2_FT_dB, lw=1, color='r', label="broken bit array FT")
ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)
ax.set_ylim(-50, 0)
ax.set_ylabel("Array Factor (linear scale)", labelpad=20)
#ax.set_theta_offset(np.pi/2)
ax.set_title(f"Beam Steering with 4-bit 1x8 Linear Array\n{n_stuck} bits stuck, Steering Angle: {steering_angle_deg:.1f}°", pad = 20)
plt.savefig(f"beamforming_{n_stuck}_bits_stuck.png", dpi=300, bbox_inches='tight')
plt.show()
'''
    

# --- Animation setup ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
scan_deg = np.arange(-180, 181)  # Scan angles from -90° to +90°
scan_rad = np.radians(scan_deg)
line0 = ax.plot([], [], lw=1, color='k', label="ideal array factor")[0]
line1 = ax.plot([], [], lw=1, color='b', label="quantised phases")[0]
line2 = ax.plot([], [], lw=1, color='r', label="broken bit array")[0]
line_optim = ax.plot([], [], lw=1, color='g', label="optimised bit array")[0]
ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)
max_steering_angle = 90  # Maximum steering angle in degrees
n_frames = 90  # Number of frames in the animation

ax.set_ylim(0, 1)
ax.set_ylabel("Array Factor (linear scale)", labelpad=20)
ax.set_theta_offset(np.pi/2)
ax.set_title("Beam Steering with 4-bit 1x8 Linear Array\n", pad = 20)

def init():
    # animate must take a list
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    line_optim.set_data([], [])
    return [line0, line1, line2, line_optim]

def animate(i):
    step = 2 * max_steering_angle / n_frames  # Step size for steering angle
    # i is the frame number
    steering_angle_deg = int(np.round(-max_steering_angle + i*step))  # Sweep from -60° to +60°
    #steering_angle_deg = -1*max_steering_angle + i*steering_step  # Sweep from -max_steering_angle to +max_steering_angle
    #print(f"Frame {i}: Steering Angle: {steering_angle_deg:.1f}°")
    #af0 = array_factor_fully_formed(scan_deg, steering_angle_deg, quantise=False)
    af0 = ideal_af_lookup(steering_angle_deg, scan_deg)
    af1 = array_factor_fully_formed(scan_deg, steering_angle_deg)
    bit_array = steering_bit_array(steering_angle_deg)
    bit_array_optim = optimise_bit_array(bit_array)
    # Break some bits in the array
    bit_array = break_bit_array(bit_array)
    af2 = array_factor_from_bits(scan_deg, bit_array)
    af_optim = array_factor_from_bits(scan_deg, bit_array_optim)
    line0.set_data(scan_rad, af0)
    line1.set_data(scan_rad, af1)
    line2.set_data(scan_rad, af2)
    line_optim.set_data(scan_rad, af_optim)
    ax.set_title(f"{n_stuck} bits stuck, Steering Angle: {steering_angle_deg:.1f}°", va='bottom', pad=20)
    return [line0, line1, line2, line_optim]

ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate, init_func=init,
    frames=n_frames, interval=100, blit=True
)

# To save as a gif: 
ani.save(f"beamforming_{n_stuck}_bits_stuck.gif", writer='pillow')
plt.show()
