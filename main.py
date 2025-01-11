import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Ustawienie kompatybilnego backendu
matplotlib.use('TkAgg')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import własnych modułów
from utils.prbs_generator import generate_prbs
from utils.channel import add_awgn_noise
from utils.modulation import bpsk_modulation, qpsk_modulation, qam16_modulation
from utils.transmission import simulate_transmission
from utils.ber_calculator import calculate_ber
from models.neural_network import create_neural_network, train_neural_network, create_classification_network


def find_best_modulation(ber_results, snr_values):
    """
    Znajduje najlepszy typ modulacji dla każdego SNR na podstawie minimalnego BER.
    """
    best_modulations = []
    for i, snr in enumerate(snr_values):
        min_ber = float('inf')
        best_modulation = None
        for mod_type, ber_list in ber_results.items():
            if ber_list[i] < min_ber:
                min_ber = ber_list[i]
                best_modulation = mod_type
        best_modulations.append((snr, best_modulation, min_ber))
    return best_modulations


def main():
    """
    Główna funkcja projektu.
    """
    # Parametry projektu
    snr_values = [5, 10, 15, 20]  # Wartości SNR (dB)
    prbs_length = 1024           # Długość sekwencji PRBS
    prbs = generate_prbs(prbs_length)  # Generowanie danych wejściowych

    # Symulacja transmisji dla różnych technik modulacji
    ber_results = {
        "BPSK": [],
        "QPSK": [],
        "16-QAM": []
    }

    # Symulacja dla BPSK
    for snr in snr_values:
        noisy_signal = simulate_transmission(bpsk_modulation, prbs, snr)
        received_bits = (noisy_signal > 0).astype(int)
        ber = calculate_ber(prbs, received_bits)
        ber_results["BPSK"].append(ber)

    # Symulacja dla QPSK
    for snr in snr_values:
        qpsk_bits = prbs[:len(prbs) // 2 * 2]  # Liczba bitów musi być parzysta
        noisy_signal = simulate_transmission(qpsk_modulation, qpsk_bits, snr)
        received_bits = np.array([
            [1 if bit.real > 0 else 0, 1 if bit.imag > 0 else 0]
            for bit in noisy_signal
        ]).flatten()
        ber = calculate_ber(qpsk_bits.flatten(), received_bits)
        ber_results["QPSK"].append(ber)

    # Symulacja dla 16-QAM
    for snr in snr_values:
        qam16_bits = prbs[:len(prbs) // 4 * 4]  # Liczba bitów musi być wielokrotnością 4
        noisy_signal = simulate_transmission(qam16_modulation, qam16_bits, snr)
        received_bits = np.array([
            [
                1 if bit.real > 0 else 0,
                1 if abs(bit.real) > 2 else 0,
                1 if bit.imag > 0 else 0,
                1 if abs(bit.imag) > 2 else 0
            ]
            for bit in noisy_signal
        ]).flatten()
        ber = calculate_ber(qam16_bits.flatten(), received_bits)
        ber_results["16-QAM"].append(ber)

    # Wizualizacja wyników BER
    for mod_type, ber in ber_results.items():
        plt.plot(snr_values, ber, marker='o', label=mod_type)

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Porównanie BER dla różnych technik modulacji")
    plt.legend()
    plt.grid()
    plt.savefig("results/plots/ber_comparison.png")
    print("Wykres zapisano w: results/plots/ber_comparison.png")

    # Znajdowanie najlepszej modulacji
    best_modulations = find_best_modulation(ber_results, snr_values)
    print("\nNajlepsze modulacje dla każdego SNR:")
    for snr, mod, ber in best_modulations:
        print(f"SNR={snr} dB -> Najlepsza modulacja: {mod}, BER={ber:.6f}")

    # Przygotowanie danych do uczenia maszynowego
    classification_features = []
    classification_labels = []
    for snr_idx, snr in enumerate(snr_values):
        # Znalezienie najlepszej modulacji
        best_mod = None
        min_ber = float('inf')
        for mod_idx, mod_type in enumerate(["BPSK", "QPSK", "16-QAM"]):
            if ber_results[mod_type][snr_idx] < min_ber:
                min_ber = ber_results[mod_type][snr_idx]
                best_mod = mod_idx  # Klasa: 0, 1 lub 2
        classification_features.append([snr])
        classification_labels.append(best_mod)

    classification_features = np.array(classification_features, dtype=np.float32)
    classification_labels = np.array(classification_labels, dtype=np.int32)

    # Trening sieci neuronowej do wyboru modulacji
    classification_model = create_classification_network(input_dim=1)
    classification_model.fit(classification_features, classification_labels, epochs=50, batch_size=8, verbose=1)

    # Predykcja optymalnej modulacji dla nowych danych
    new_snr_values = np.array([[12], [15], [18]], dtype=np.float32)  # Nowe wartości SNR
    predicted_classes = classification_model.predict(new_snr_values)
    predicted_modulations = np.argmax(predicted_classes, axis=1)

    print("\nPredykcja najlepszych modulacji dla nowych wartości SNR:")
    for snr, mod in zip(new_snr_values.flatten(), predicted_modulations):
        mod_type = ["BPSK", "QPSK", "16-QAM"][mod]
        print(f"SNR={snr:.1f} -> Najlepsza modulacja: {mod_type}")

if __name__ == "__main__":
    main()
