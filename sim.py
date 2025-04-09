'''
Created by Alexander Li, 2025-4-9
This script simulates the beamforming of a uniform linear array (ULA) with 4-bit quantized phase shifters.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Array and beamforming setup ---
N = 8                         # Number of elements in ULA
n_bits = 4                    # Phase shifter resolution (4 bits)
phase_levels = 2 ** n_bits   # 16 levels
d = 0.5                       # Element spacing in wavelengths
wavelength = 1.0            # unit wavelength
k = 2 * np.pi / wavelength    # Wavenumber

# --- Bit breaking setup ---
# Randomly select 3 elements to break
broken_elements = [1, 4, 5]
broken_bits = [0, 1, 2]
broken_values = [0, 1, 1]

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

def break_bit_array(bit_array):
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

# --- Animation setup ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
scan_deg = np.linspace(-180, 180, 1000)
scan_rad = np.radians(scan_deg)
line0 = ax.plot([], [], lw=1, color='k', label="ideal array factor")[0]
line1 = ax.plot([], [], lw=1, color='b', label="quantised phases")[0]
line2 = ax.plot([], [], lw=1, color='r', label="broken bit array")[0]
ax.legend(loc='upper right', fontsize=8, frameon=False, bbox_to_anchor=(1.1, 1.1), handlelength=1.5, handleheight=0.5, borderpad=0.5)
n_frames = 30

ax.set_ylim(0, 1)
ax.set_ylabel("Array Factor (linear scale)", labelpad=20)
ax.set_theta_offset(np.pi/2)
ax.set_title("Beam Steering with 4-bit 1x8 Linear Array\n", pad = 20)

def init():
    # animate must take a list
    line0.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    return [line0, line1, line2]

def animate(i):
    max_steering_angle = 90  # Maximum steering angle in degrees
    # angle step
    steering_step = 2 * max_steering_angle / n_frames
    # i is the frame number
    steering_angle_deg = -max_steering_angle + i * steering_step  # Sweep from -60° to +60°
    #steering_angle_deg = -1*max_steering_angle + i*steering_step  # Sweep from -max_steering_angle to +max_steering_angle
    #print(f"Frame {i}: Steering Angle: {steering_angle_deg:.1f}°")
    af0 = array_factor_fully_formed(scan_deg, steering_angle_deg, quantise=False)
    af1 = array_factor_fully_formed(scan_deg, steering_angle_deg)
    bit_array = steering_bit_array(steering_angle_deg)
    # Break some bits in the array
    bit_array = break_bit_array(bit_array)
    af2 = array_factor_from_bits(scan_deg, bit_array)
    line0.set_data(scan_rad, af0)
    line1.set_data(scan_rad, af1)
    line2.set_data(scan_rad, af2)
    ax.set_title(f"Steering Angle: {steering_angle_deg:.1f}°", va='bottom', pad=20)
    return [line0, line1, line2]

ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate, init_func=init,
    frames=n_frames, interval=100, blit=True
)

# To save as a gif: 
ani.save("beamforming.gif", writer='pillow')
plt.show()
