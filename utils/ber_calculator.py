import numpy as np
def calculate_ber(original_bits, received_bits):
    """
    Obliczanie wskaźnika BER.
    """
    errors = np.sum(original_bits != received_bits)
    return errors / len(original_bits)
