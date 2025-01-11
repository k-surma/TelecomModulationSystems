import numpy as np
def calculate_ber(original_bits, received_bits):
    """
    Obliczanie wskaźnika BER.
    """
    errors = np.sum(original_bits != received_bits)
    ber = errors / len(original_bits)
    return max(ber, 1e-10)  # Zapewnia, że BER nie będzie 0

