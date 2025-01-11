import numpy as np

def generate_prbs(length, polynomial=[5, 2], seed=1):
    """
    Generuje sekwencję PRBS przy użyciu rejestrów przesuwających.
    """
    register = [int(x) for x in f"{seed:05b}"]
    prbs = []

    for _ in range(length):
        output = register[-1]
        feedback = np.bitwise_xor.reduce([register[i - 1] for i in polynomial])
        prbs.append(output)
        register = [feedback] + register[:-1]

    return np.array(prbs)
