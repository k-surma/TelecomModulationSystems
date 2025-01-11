import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import random

# Ustawienie kompatybilnego backendu
matplotlib.use('TkAgg')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import własnych modułów
from utils.prbs_generator import generate_prbs
#from utils.modulation import bpsk_modulation
from utils.modulation import  qpsk_modulation, qam16_modulation
from utils.transmission import simulate_transmission
from utils.ber_calculator import calculate_ber
from models.neural_network import create_classification_network


def normalize_ber(ber_results):
    """
    Normalizuje BER w skali logarytmicznej.
    """
    normalized_results = {}
    for mod_type, ber_list in ber_results.items():
        normalized_results[mod_type] = [np.log10(ber + 1e-10) for ber in ber_list]
    return normalized_results


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
    # Wartości SNR, których nie chcemy wylosować
    excluded_snr_values = [12, 15, 18]

    # Generowanie 5 unikalnych losowych liczb z zakresu (0-40), które nie znajdują się w excluded_snr_values
    snr_values = []
    while len(snr_values) < 5:
        random_value = random.randint(0, 40)
        if random_value not in snr_values and random_value not in excluded_snr_values:
            snr_values.append(random_value)

    prbs_length = 1024           # Długość sekwencji PRBS
    prbs = generate_prbs(prbs_length)  # Generowanie danych wejściowych

    # Symulacja transmisji dla różnych technik modulacji
    ber_results = {
        #"BPSK": [],
        "QPSK": [],
        "16-QAM": []
    }

    # # Symulacja dla BPSK
    # for snr in snr_values:
    #     noisy_signal = simulate_transmission(bpsk_modulation, prbs, snr)
    #     received_bits = (noisy_signal > 0).astype(int)
    #     ber = calculate_ber(prbs, received_bits)
    #     ber_results["BPSK"].append(ber)

    # Symulacja dla QPSK
    for snr in snr_values:
        qpsk_bits = prbs[:len(prbs) // 2 * 2]  # Liczba bitów musi być parzysta
        noisy_signal = simulate_transmission(qpsk_modulation, qpsk_bits, snr)

        # Detekcja bitów w QPSK
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

        # Detekcja bitów w 16-QAM
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
        plt.semilogy(snr_values, ber, marker='o', label=mod_type)  # Skala logarytmiczna

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (log10 scale)")
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

    # Normalizacja BER
    normalized_ber_results= normalize_ber(ber_results)

    # Przygotowanie danych do uczenia maszynowego
    classification_features = []
    classification_labels = []
    for snr_idx, snr in enumerate(snr_values):
        feature_vector = [snr]
        # for mod_type in ["BPSK", "QPSK", "16-QAM"]:
        for mod_type in ["QPSK", "16-QAM"]:
            feature_vector.append(normalized_ber_results[mod_type][snr_idx])
        classification_features.append(feature_vector)
        # Etykieta: Indeks modulacji o najniższym BER
        classification_labels.append(np.argmin([#ber_results["BPSK"][snr_idx],
                                                ber_results["QPSK"][snr_idx],
                                                ber_results["16-QAM"][snr_idx]]))

    classification_features = np.array(classification_features, dtype=np.float32)
    classification_labels = np.array(classification_labels, dtype=np.int32)

    # Trening sieci neuronowej do wyboru modulacji
    classification_model = create_classification_network(input_dim=len(classification_features[0]))
    classification_model.fit(classification_features, classification_labels, epochs=100, batch_size=8, verbose=1)

    # Predykcja optymalnej modulacji dla nowych danych
    #new_snr_values = np.array([[12, 0.2, 0.4, 0.6], [15, 0.1, 0.3, 0.5], [18, 0.05, 0.2, 0.3]], dtype=np.float32)
    new_snr_values = np.array([[12, 0.4, 0.6], [15, 0.3, 0.5], [18, 0.2, 0.3]], dtype=np.float32)

    predicted_classes = classification_model.predict(new_snr_values)
    print("\nPredykcja najlepszych modulacji dla nowych danych:")
    for snr, pred in zip(new_snr_values, np.argmax(predicted_classes, axis=1)):
        # print(f"SNR={snr[0]:.1f} -> Najlepsza modulacja: {['BPSK', 'QPSK', '16-QAM'][pred]}")
        print(f"SNR={snr[0]:.1f} -> Najlepsza modulacja: {['QPSK', '16-QAM'][pred]}")

if __name__ == "__main__":
    main()
