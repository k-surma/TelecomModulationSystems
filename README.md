# Symulacja optymalnej modulacji w systemach telekomunikacyjnych z użyciem PRBS i sieci neuronowych

## Opis
Projekt symuluje transmisję danych w telekomunikacji z użyciem trzech technik modulacji: **BPSK**, **QPSK** oraz **16-QAM**. Analizuje jakość transmisji, obliczając wskaźnik **BER** (Bit Error Rate) w różnych warunkach zakłóceń (**SNR** - Signal-to-Noise Ratio). Projekt wykorzystuje sieci neuronowe TensorFlow/Keras do przewidywania optymalnej techniki modulacji na podstawie parametrów kanału transmisyjnego.

Projekt jest modularny i zawiera następujące komponenty:
1. **Generowanie PRBS**: Sekwencje pseudolosowe jako dane wejściowe.
2. **Modelowanie kanału transmisyjnego**: Dodanie szumu AWGN.
3. **Implementacja technik modulacji**: BPSK, QPSK, 16-QAM.
4. **Symulacja transmisji**: Przesyłanie danych zmodulowanych przez kanał z zakłóceniami.
5. **Uczenie maszynowe**: Sieć neuronowa do analizy parametrów transmisji i wyboru optymalnej modulacji.

## Instalacja
1. **Sklonuj repozytorium:**
   ```bash
   git clone https://github.com/k-surma/TelecomModulationSystems.git
   cd TelecomModulationSystems
   ```
2. **Utwórz wirtualne środowisko i aktywuj je:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
    ```
3. **Zainstaluj wymagane biblioteki:**
    ```bash
    pip install -r requirements.txt
    ```

## Uruchamianie
1. **Uruchom główny skrypt:**
    ```bash
    python main.py
    ```
2. **Sprawdź wyniki:**
Wygenerowane wykresy jakości transmisji znajdziesz w katalogu 
`results/plots/`


## Struktura projektu
- `utils/` - Moduły pomocnicze (PRBS, modulacje, szum).
- `models/` - Sieci neuronowe do przewidywania optymalnej modulacji.
- `results/` - Wygenerowane wyniki i wykresy.

## Wymagane biblioteki
- TensorFlow
- NumPy
- Matplotlib
