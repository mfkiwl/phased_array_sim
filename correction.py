'''
Created by Alexander Li on 2025-4-14
Consider the ring of integers modulo 2^n.
In this ring, the elements are represented as n-bit binary numbers.
If a number is desired, but certain bits are fixed to 0 or 1, we wish to find the number that is closest to the desired number, while respecting the fixed bits.
There are two approachse considered:
1. Greedy: start from the desired number and move towards the closest number that respects the fixed bits.
2. Dynamic programming: use a table to store the closest number for each possible fixed bits configuration.
'''

import numpy as np
import matplotlib.pyplot as plt

def valid_integers(n_bits_total, fixed_bit_locations, fixed_bit_values):
    '''
    Generate all valid integers in a ring of integers modulo 2^n.
    The function takes the total number of bits, the locations of fixed bits, and their values.
    Input:
        n_bits_total: the total number of bits
        fixed_bit_locations: a list of indices where bits are fixed
        fixed_bit_values: a list of values (0 or 1) for the fixed bits
    Output:
        valid_integers: a list of valid integers
    '''
    # sort the fixed bits by their locations, with the fixed bit valus also sorted.
    fixed_bit_locations = np.array(fixed_bit_locations)
    fixed_bit_values = np.array(fixed_bit_values)
    sorted_indices = np.argsort(fixed_bit_locations)
    fixed_bit_locations = fixed_bit_locations[sorted_indices]
    fixed_bit_values = fixed_bit_values[sorted_indices]
    # Create a list of all possible integers in the ring
    if fixed_bit_locations.size == 0 or fixed_bit_locations is None:
        # If there are no fixed bits, return all integers in the ring
        return np.arange(2**n_bits_total)

    if fixed_bit_locations[0] == 0:
        # If the first fixed bit is not at index 0, add the range of integers from 0 to the first fixed bit
        valid_integers = np.array([[fixed_bit_values[0]]])
        fixed_bit_locations = fixed_bit_locations[1:]
        fixed_bit_values = fixed_bit_values[1:]
    else:
        # Create a list of all possible integers in the ring
        valid_integers = np.array(
            [
                [0],
                [1]
            ]
        )
    for i in range(1, n_bits_total):
        if fixed_bit_locations.size == 0 or fixed_bit_locations[0] != i:
            zeroes = np.zeros((valid_integers.shape[0], 1), dtype=int)
            ones = np.ones((valid_integers.shape[0], 1), dtype=int)
            array0 = np.concatenate((valid_integers, zeroes), axis=1)
            array1 = np.concatenate((valid_integers, ones), axis=1)
            valid_integers = np.concatenate((array0, array1), axis=0)
        # remove the first element of the fixed bits
        else:
            if fixed_bit_values[0] == 0:
                zeroes = np.zeros((valid_integers.shape[0], 1), dtype=int)
                valid_integers = np.concatenate((valid_integers, zeroes), axis=1)
            elif fixed_bit_values[0] == 1:
                ones = np.ones((valid_integers.shape[0], 1), dtype=int)
                valid_integers = np.concatenate((valid_integers, ones), axis=1)
            fixed_bit_locations = fixed_bit_locations[1:]
            fixed_bit_values = fixed_bit_values[1:]
        
    # Convert the binary numbers to integers
    valid_integers = np.array([int(''.join(map(str, row))[::-1], 2) for row in valid_integers])
    return valid_integers

def modular_distance(a, b, mod):
    '''
    Calculate the modular distance between two integers in a ring of integers modulo 2^n.
    The function takes two integers and the total number of bits.
    Input:
        a: first integer
        b: second integer
        n_bits_total: the total number of bits
    Output:
        distance: the modular distance between the two integers
    '''
    dist1 = (a - b) % mod
    dist2 = (b - a) % mod
    # add the mod to the distance to avoid negative values
    if dist1 < 0:
        dist1 += mod
    if dist2 < 0:
        dist2 += mod
    # Calculate the modular distance
    return min((a - b) % mod, (b - a) % mod)

def closest_integer(a, n, fixed_bit_locations, fixed_bit_values):
    '''
    Find the closest integer to a in a ring of integers modulo 2^n.
    The function takes an integer, the total number of bits, the locations of fixed bits, and their values.
    Input:
        a: the integer to find the closest integer to
        n_bits_total: the total number of bits
        fixed_bit_locations: a list of indices where bits are fixed
        fixed_bit_values: a list of values (0 or 1) for the fixed bits
    Output:
        closest_integer: the closest integer to a that respects the fixed bits
    '''
    # Generate all valid integers in the ring
    valid_ints = valid_integers(n, fixed_bit_locations, fixed_bit_values)
    #print(f"Valid integers: {valid_ints}")
    # Calculate the modular distance between a and each valid integer
    distances = np.array([modular_distance(a, valid_int, 2**n) for valid_int in valid_ints])
    #print(f"Distances: {distances}")
    # Find the index of the closest integer
    closest_index = np.argmin(distances)
    # if there are 
    #print(f"Closest index: {closest_index}")
    # Return the closest integer
    return valid_ints[closest_index]
