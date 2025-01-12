from utils.channel import add_awgn_noise

def simulate_transmission(modulation_func, bits, snr_db):
    """
    Symuluje transmisję danych przez kanał z szumem AWGN.

    :param modulation_func: Funkcja modulacji (np. BPSK, QPSK).
    :param bits: Dane wejściowe (ciąg bitów).
    :param snr_db: Wartość SNR w dB.
    :return: Zaszumiony sygnał po modulacji.
    """
    # Modulacja danych
    modulated_signal = modulation_func(bits)

    # Dodanie szumu AWGN
    noisy_signal = add_awgn_noise(modulated_signal, snr_db)

    return noisy_signal
