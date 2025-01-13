import numpy as np

def add_awgn_noise(signal, snr_db):
    """
    AWGN (Additive White Gaussian noise) - szum symulujący zakłócenia z otoczenia
    """
    if len(signal) == 0:
        raise ValueError("Sygnał wejściowy jest pusty!")

    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(signal)**2)  # moc sygnału - srednia wartosc kwaddratu amplitudy sygnalu
    noise_power = signal_power / snr_linear    # moc szumu - tutaj wynika z SNR=Ps/Pn

    # szum Gaussowski zespolony - generujemy zestaw liczb zespolonych
    # o takiej samej długosci jak nasz sygnał,
    # gdzie każda część (rzeczywista i urojona) jest losowana z N(0,1),
    # a następnie skalujemy je tak, żeby końcowa moc wynosiła noise_power
    noise = np.sqrt(noise_power / 2) * (
        np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape)
    )
    return signal + noise
