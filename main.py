import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('TkAgg')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from utils.prbs_generator import generate_prbs
from utils.modulation import bpsk_modulation, qpsk_modulation
from utils.transmission import simulate_transmission
from utils.ber_calculator import calculate_ber
from models.neural_network import create_classification_network

def main():
    # seria SNR do testów
    snr_values = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    # generujemy dłuższą sekwencję bitów, by rzadziej mieć idealne 0 błędów
    prbs_length = 50_000
    prbs = generate_prbs(prbs_length)

    np.savetxt("prbs_sequence.txt", prbs)

    # słowniki na wyniki
    ber_results = {"BPSK": [], "QPSK": []}
    throughput_results = {"BPSK": [], "QPSK": []}

    # dla BPSK mamy 1 bit/symbol, dla QPSK – 2 bity/symbol
    bpsk_bits_per_symbol = 1
    qpsk_bits_per_symbol = 2

    # symulacja BPSK
    for snr_db in snr_values:
        noisy_signal = simulate_transmission(
            modulation_func=bpsk_modulation,
            bits=prbs,
            snr_db=snr_db
        )
        # detekcja BPSK
        received_bits = (noisy_signal.real > 0).astype(int) # demodulacja - zamiana otrzymanego sygnalu z powrotem na ciag bitow
        ber_bpsk = calculate_ber(prbs, received_bits)
        ber_results["BPSK"].append(ber_bpsk)

        # Throughput - przepływność - ilosc poprawnie otrzymanych bitów na symbol transmisyjny
        thr_bpsk = bpsk_bits_per_symbol * (1.0 - ber_bpsk)
        throughput_results["BPSK"].append(thr_bpsk)

    # symulacja QPSK
    # QPSK wymaga wielokrotności 2 bitów
    qpsk_bits = prbs[: (len(prbs) // 2) * 2] # zapobieganie sytuacji w ktr ciąg nie jest wielokrotnością 2 i usuniecie w razie czego ostatniego bitu
    for snr_db in snr_values:
        noisy_signal = simulate_transmission(
            modulation_func=qpsk_modulation,
            bits=qpsk_bits,
            snr_db=snr_db
        )
        # detekcja QPSK
        received_bits = np.array([
            [1 if symbol.real > 0 else 0,
             1 if symbol.imag > 0 else 0]
            for symbol in noisy_signal
        ]).flatten()
        ber_qpsk = calculate_ber(qpsk_bits, received_bits)
        ber_results["QPSK"].append(ber_qpsk)

        # kara implementacyjna w celach eksperymentalnych
        norm = 0.93 if snr_db<12 else 1.0
        thr_qpsk = norm*qpsk_bits_per_symbol * (1.0 - ber_qpsk)
        throughput_results["QPSK"].append(thr_qpsk)

    # rysowanie BER
    plt.figure(figsize=(7, 5))
    for mod_type in ["BPSK", "QPSK"]:
        plt.semilogy(snr_values, ber_results[mod_type], marker='o', label=mod_type)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (log10 scale)")
    plt.title("Porównanie BER: BPSK vs. QPSK")
    plt.grid(True)
    plt.legend()
    os.makedirs("results/plots", exist_ok=True)
    print("Zapisano wykres BER w: results/plots/ber_bpsk_qpsk.png")

    # rysowanie Throughput
    plt.figure(figsize=(7, 5))
    for mod_type in ["BPSK", "QPSK"]:
        plt.plot(snr_values, throughput_results[mod_type], marker='s', label=mod_type)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Throughput [bit/symbol]")
    plt.title("Efektywna przepływność: BPSK vs. QPSK")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plots/throughput_bpsk_qpsk.png")
    print("Zapisano wykres Throughput w: results/plots/throughput_bpsk_qpsk.png")

    # Porównanie modulacji na podstawie Throughput
    print("\nPorównanie modulacji na podstawie Throughput:")
    for i, snr_db in enumerate(snr_values):
        bpsk_thr = throughput_results["BPSK"][i]
        qpsk_thr = throughput_results["QPSK"][i]
        best_mod = "BPSK" if bpsk_thr > qpsk_thr else "QPSK"
        print(f"SNR={snr_db} dB -> BPSK_thr={bpsk_thr:.4f}, QPSK_thr={qpsk_thr:.4f}, najlepsza={best_mod}")

    # Sieć neuronowa do wyboru modulacji

    # cecha: [SNR, ber_BPSK, ber_QPSK, throughput_BPSK, throughput_QPSK]
    # etykieta: 0 (BPSK) lub 1 (QPSK), w zależności od wyższego Throughput

    features = []
    labels = []
    for i, snr_db in enumerate(snr_values):
        features.append([
            snr_db,
            ber_results["BPSK"][i],
            ber_results["QPSK"][i],
            throughput_results["BPSK"][i],
            throughput_results["QPSK"][i]
        ])
        label = 0 if throughput_results["BPSK"][i] > throughput_results["QPSK"][i] else 1
        labels.append(label)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # tworzymy i trenujemy model
    model = create_classification_network(input_dim=5)
    model.fit(features, labels, epochs=50, batch_size=4, verbose=1)
    # cechy, etykiety, epochs - iteracje, batch_size - ilosc probek na iteracje, verbose - wyswietla nam proces nauki

    # przykładowa predykcja
    new_samples = np.array([
        [5,   0.2,   0.35,  0.8,   0.5   ],  # SNR=5
        [15,  0.01,  0.03,  0.99,  1.4   ],  # SNR=15
        [25,  1e-4,  5e-4,  0.9999,1.99  ]   # SNR=25
    ], dtype=np.float32)
    predictions = model.predict(new_samples)
    print("\nPredykcja NN: (0->BPSK, 1->QPSK)")
    for sample, pred_prob in zip(new_samples, predictions):
        pred_label = np.argmax(pred_prob)
        chosen_mod = "BPSK" if pred_label == 0 else "QPSK"
        print(f"SNR={sample[0]} dB -> wybrano: {chosen_mod}")


if __name__ == "__main__":
    main()
