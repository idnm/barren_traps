import numpy as np


def random_local_two_body_pauli(num_qubits):
    p1, p2 = np.random.choice(['X', 'Y', 'Z'], size=2, replace=True)
    i = np.random.choice(range(num_qubits - 1))
    return 'I' * i + p1 + p2 + 'I' * (num_qubits - i - 2)


def all_local_two_body_pauli(num_qubits):
    """Generate all two-body, local, nearest neighbour Pauli strings.

     Example: num_qubits=3, res = ['XXI', 'IXX', 'XYI', 'IXY', ...]
     """
    res = []
    for n in range(1, num_qubits):
        for p1 in ['X', 'Y', 'Z']:
            for p2 in ['X', 'Y', 'Z']:
                pauli_string = 'I' * (n - 1) + p1 + p2 + 'I' * (num_qubits - n - 1)
                res.append(pauli_string)
    return res


def all_two_body_pauli(num_qubits):
    """Generate all two-body Pauli strings, not necessarily local.
     """
    res = []
    for i in range(num_qubits - 1):
        for j in range(i + 1, num_qubits):
            for pi in ['X', 'Y', 'Z']:
                for pj in ['X', 'Y', 'Z']:
                    p = ['I'] * num_qubits
                    p[i] = pi
                    p[j] = pj
                    res.append(''.join(p))
    return res