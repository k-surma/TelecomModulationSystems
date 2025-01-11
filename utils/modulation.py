import numpy as np

# def bpsk_modulation(bits):
#     return 2 * bits - 1

def qpsk_modulation(bits):
    """
    Modulacja QPSK: Grupowanie bitów po dwa i przypisanie faz.
    """
    if len(bits) % 2 != 0:
        raise ValueError("Długość bitów wejściowych musi być wielokrotnością 2.")

    bits = bits.reshape(-1, 2)
    mapping = {
        (0, 0): complex(1, 1),
        (0, 1): complex(-1, 1),
        (1, 0): complex(1, -1),
        (1, 1): complex(-1, -1)
    }
    return np.array([mapping[tuple(pair)] for pair in bits])

def qam16_modulation(bits):
    """
    Modulacja 16-QAM: Grupowanie bitów po cztery i przypisanie współrzędnych.
    """
    if len(bits) % 4 != 0:
        raise ValueError("Długość bitów wejściowych musi być wielokrotnością 4.")

    bits = bits.reshape(-1, 4)
    mapping = {
        (0, 0, 0, 0): complex(-3, -3), (0, 0, 0, 1): complex(-3, -1),
        (0, 0, 1, 0): complex(-3, 3),  (0, 0, 1, 1): complex(-3, 1),
        (1, 0, 0, 0): complex(3, -3),  (1, 0, 0, 1): complex(3, -1),
        (1, 0, 1, 0): complex(3, 3),   (1, 0, 1, 1): complex(3, 1),
        (0, 1, 0, 0): complex(-1, -3), (0, 1, 0, 1): complex(-1, -1),
        (0, 1, 1, 0): complex(-1, 3),  (0, 1, 1, 1): complex(-1, 1),
        (1, 1, 0, 0): complex(1, -3),  (1, 1, 0, 1): complex(1, -1),
        (1, 1, 1, 0): complex(1, 3),   (1, 1, 1, 1): complex(1, 1)
    }
    return np.array([mapping[tuple(group)] for group in bits])
