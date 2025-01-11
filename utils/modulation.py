import numpy as np

def bpsk_modulation(bits):
    return 2 * bits - 1

def qpsk_modulation(bits):
    bits = bits.reshape(-1, 2)
    mapping = {
        (0, 0): (1 + 1j), (0, 1): (-1 + 1j),
        (1, 0): (1 - 1j), (1, 1): (-1 - 1j)
    }
    return np.array([mapping[tuple(b)] for b in bits])

def qam16_modulation(bits):
    bits = bits.reshape(-1, 4)
    mapping = {
        (0, 0, 0, 0): (-3 - 3j), (0, 0, 0, 1): (-3 - 1j),
        (0, 0, 1, 0): (-3 + 3j), (0, 0, 1, 1): (-3 + 1j),
        (1, 0, 0, 0): (3 - 3j),  (1, 0, 0, 1): (3 - 1j),
        (1, 0, 1, 0): (3 + 3j),  (1, 0, 1, 1): (3 + 1j),
        (0, 1, 0, 0): (-1 - 3j), (0, 1, 0, 1): (-1 - 1j),
        (0, 1, 1, 0): (-1 + 3j), (0, 1, 1, 1): (-1 + 1j),
        (1, 1, 0, 0): (1 - 3j),  (1, 1, 0, 1): (1 - 1j),
        (1, 1, 1, 0): (1 + 3j),  (1, 1, 1, 1): (1 + 1j)
    }
    return np.array([mapping[tuple(b)] for b in bits])
