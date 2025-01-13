import numpy as np

def calculate_ber(original_bits, received_bits):
    """
    obliczanie wskaźnika BER
    """
    errors = np.sum(original_bits != received_bits)
    ber = errors / len(original_bits)

    return max(ber, 1e-8)
