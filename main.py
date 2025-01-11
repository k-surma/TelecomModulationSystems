from models.neural_network import create_neural_network, train_neural_network
import numpy as np
import matplotlib.pyplot as plt
from utils.prbs_generator import generate_prbs
from utils.channel import add_awgn_noise
from utils.modulation import bpsk_modulation, qpsk_modulation, qam16_modulation
from utils.ber_calculator import calculate_ber

# Parametry projektu
snr_values = [5, 10, 15, 20]
prbs = generate_prbs(1024)

# Symulacja dla BPSK
ber_bpsk = []
for snr in snr_values:
    modulated_signal = bpsk_modulation(prbs)
    noisy_signal = add_awgn_noise(modulated_signal, snr)
    received_bits = (noisy_signal > 0).astype(int)
    ber = calculate_ber(prbs, received_bits)
    ber_bpsk.append(ber)

# Wizualizacja BER
plt.plot(snr_values, ber_bpsk, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER dla BPSK")
plt.grid()
plt.show()


# Przygotowanie danych wejściowych
features = np.array([[snr, 0] for snr in snr_values])  # 0: BPSK
labels = np.array(ber_bpsk)

# Tworzenie i trenowanie modelu
model = create_neural_network(input_dim=2)
history = train_neural_network(model, features, labels, epochs=100, batch_size=4)

# Predykcja nowych wartości
new_features = np.array([[12, 0], [18, 0]])  # Testowe SNR dla BPSK
predicted_ber = model.predict(new_features)
print("Przewidywane BER dla nowych danych:", predicted_ber)
