import numpy as np

def calculate_ber(original_bits, received_bits):
    """
    Obliczanie wskaźnika BER.
    """
    errors = np.sum(original_bits != received_bits)
    ber = errors / len(original_bits)
    # Zwracamy minimalnie 1e-10, żeby uniknąć log(0)
    return max(ber, 1e-10)
