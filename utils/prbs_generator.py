import numpy as np

def generate_prbs(length, polynomial=None, seed=1):
    """
    generuje sekwencję PRBS przy użyciu rejestrów przesuwających
    """
    if polynomial is None:
        polynomial = [5, 3]

    max_index = max(polynomial)
    # inicjujemy rejestr na podstawie wartości 'seed' - ziarna
    register = [int(x) for x in f"{seed:0{max_index}b}"]

    # wykonujemy xora na bitach rejestru okreslonym w wielomianie charakterystycznym i wpisujemy do prbs
    prbs = []
    for _ in range(length):
        output = register[-1]
        feedback = np.bitwise_xor.reduce([register[i - 1] for i in polynomial])
        prbs.append(output)
        register = [feedback] + register[:-1]

    return np.array(prbs)
