import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def create_neural_network(input_dim):
    """
    Tworzy prostą sieć neuronową do przewidywania optymalnej techniki modulacji.
    """
    model = Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Wyjście regresji dla wartości BER
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_neural_network(model, features, labels, epochs=100, batch_size=16):
    """
    Trenuje model sieci neuronowej.
    """
    history = model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=1)
    return history

def create_classification_network(input_dim):
    """
    Tworzy sieć neuronową do klasyfikacji optymalnej techniki modulacji.
    """
    from tensorflow.keras.layers import Dropout
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')  # Trzy klasy: BPSK, QPSK, 16-QAM
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

