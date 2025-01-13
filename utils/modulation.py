import numpy as np

def bpsk_modulation(bits):
    """
    Modulacja BPSK: bity 0,1 mapowane na symbole -1, +1
    """
    return 2 * bits - 1

def qpsk_modulation(bits):
    """
    modulacja QPSK: grupowanie bitów po 2 i przypisanie faz
    (0,0) ->  1 + j
    (0,1) -> -1 + j
    (1,0) ->  1 - j
    (1,1) -> -1 - j
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
    return np.array([mapping[tuple(pair)] for pair in bits])    # dla kadej pary bitów
    # funkcja przypisuje odpowiedni symbol z mapowania i tworzy nową tablicę symboli QPSK
