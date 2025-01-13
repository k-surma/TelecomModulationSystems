from utils.channel import add_awgn_noise

def simulate_transmission(modulation_func, bits, snr_db):
    """
    symuluje transmisję danych przez kanał z szumem AWGN

    :param modulation_func: funkcja modulacji ( BPSK, QPSK)
    :param bits: dane wejściowe (ciąg bitów)
    :param snr_db: wartość SNR w dB
    :return: zaszumiony sygnał po modulacji
    """
    # modulacja danych
    modulated_signal = modulation_func(bits)

    # dodanie szumu AWGN
    noisy_signal = add_awgn_noise(modulated_signal, snr_db)

    return noisy_signal
