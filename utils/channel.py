import numpy as np

def add_awgn_noise(signal, snr_db):
    """
    Dodaje szum AWGN do sygnału, uwzględniając SNR w decybelach.
    """
    if len(signal) == 0:
        raise ValueError("Sygnał wejściowy jest pusty!")

    snr_linear = 10 ** (snr_db / 10)  # Przeliczenie SNR na skalę liniową
    signal_power = np.mean(np.abs(signal)**2)  # Moc sygnału
    noise_power = signal_power / snr_linear    # Moc szumu
    noise = np.sqrt(noise_power / 2) * (np.random.normal(size=signal.shape) +
                                        1j * np.random.normal(size=signal.shape))  # Szum Gaussowski zespolony
    return signal + noise


