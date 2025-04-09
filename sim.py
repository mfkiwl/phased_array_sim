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
max_steering_angle = np.degrees(np.asin(wavelength / (N * d)))  # Maximum steering angle in radians
print(f"Max steering angle: {max_steering_angle:.2f}°")

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

def array_factor_fully_formed(scan_deg, steering_angle_deg):
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
    quantized_phases_rad = quantize_phase(phase_shifts_rad)
    # Compute AF
    af = np.zeros_like(scan_rad, dtype=complex)
    for n in range(N):
        af += np.exp(1j * (k * d * n * np.sin(scan_rad) + quantized_phases_rad[n]))
    return np.abs(af) / N

# --- Animation setup ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
scan_deg = np.linspace(-90, 90, 1000)
scan_rad = np.radians(scan_deg)
line = ax.plot([], [], lw=1, color='k')[0]
n_frames = 30

ax.set_ylim(0, 1)
ax.set_title("Beam Steering with 4-bit Quantized Phase\n", va='bottom')

def init():
    # animate must take a list
    line.set_data([], [])
    return [line]

def animate(i):
    # angle step
    steering_step = 2 * max_steering_angle / n_frames
    # i is the frame number
    steering_angle_deg = -1*max_steering_angle + i*steering_step  # Sweep from -60° to +60°
    #print(f"Frame {i}: Steering Angle: {steering_angle_deg:.1f}°")
    #af = array_factor_fully_formed(scan_deg, steering_angle_deg)
    bit_array = steering_bit_array(steering_angle_deg)
    af = array_factor_from_bits(scan_deg, bit_array)
    line.set_data(scan_rad, af)
    ax.set_title(f"Steering Angle: {steering_angle_deg:.1f}°", va='bottom')
    return [line]

ani = animation.FuncAnimation(
    # interval is the time between frames in milliseconds
    fig, animate, init_func=init,
    frames=n_frames, interval=100, blit=True
)

# To save as a gif: ani.save("beamforming.gif", writer='pillow')
plt.show()
