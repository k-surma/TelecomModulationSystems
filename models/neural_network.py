import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_classification_network(input_dim):
    """
    Tworzy model klasyfikacyjny dla 2 klas: BPSK i QPSK.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(2, activation='softmax')  # 2 klasy: BPSK (indeks 0) i QPSK (indeks 1)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
