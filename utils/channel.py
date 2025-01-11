import numpy as np

def add_awgn_noise(signal, snr_db):
    """
    Dodaje szum AWGN do sygnału.
    """
    if len(signal) == 0:
        raise ValueError("Sygnał wejściowy jest pusty!")

    snr_linear = 10 ** (snr_db / 10)
    power_signal = np.mean(signal**2)
    if power_signal == 0:
        raise ValueError("Moc sygnału jest zerowa, nie można dodać szumu.")

    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    return signal + noise
