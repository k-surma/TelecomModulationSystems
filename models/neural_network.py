import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_classification_network(input_dim):
    """
    tworzy model klasyfikacyjny dla 2 klas: BPSK i QPSK

    MLP (Multilayer Perceptron) / „Fully-connected network” – to sieć, gdzie każda warstwa jest gęsto połączona (Dense)
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.1),  # żeby nie dopuścić do przeuczenia porzucamy 10% losowych neuronów
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(2, activation='softmax')  # output layer - BPSK (id 0) i QPSK (id 1), softmax zwraca prawdopodob. przynalezności do klas
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# NOTATKI

# relu - "funkcja aktywacji" wprowadza nieliniowość, dzięki czemu model może uczyć się bardziej skomplikowanych wzorców,
# wybiera max(0,x)

# batch_size - liczba próbek

# optimizer adam - algorytm optymalizacji używany do aktualizacji wag sieci, minimalizując funkcję straty,
# automatycznie dostosowuje tempo uczenia się

# sparse_categorical_crossentropy - mierzy różnicę między rzeczywistymi etykietami a przewidywaniami sieci, stosowana
# w problemach klasyfikacji wieloklasowej

# metrics accuracy monitoruje dokladnosc podczas trenowania modelu,co pozwala ocenić,jak dobrze model klasyfikuje dane
