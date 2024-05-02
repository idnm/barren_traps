from ast import Str
from typing import List

import numpy as np
import qiskit.quantum_info


def random_local_two_body_pauli(num_qubits: int) -> Str:
    """Random Pauli operator of weight 2, supported on adjacent qubits."""

    p1, p2 = np.random.choice(['X', 'Y', 'Z'], size=2, replace=True)
    i = np.random.choice(range(num_qubits - 1))
    return 'I' * i + p1 + p2 + 'I' * (num_qubits - i - 2)


def all_local_two_body_pauli(num_qubits: int) -> List[Str]:
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


def pauli_batch(num_qubits: int, batch_size: int, max_weight: int, seed: int | np.random.Generator = 42) -> list[str]:
    """Generate a batch of Pauli operators of max weight `max_weight`, not necessarily local.
        batch_size: how many to generate
        max_weight: maximal weight of each Pauli (can be lees, but weight zero (identity) is excluded)
        seed: either an int seed or a Generator.
     """

    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    batch = set()
    while len(batch) < batch_size:
        pauli = ['I'] * num_qubits

        weight = min(num_qubits, max_weight)
        letters = rng.choice(['I', 'X', 'Y', 'Z'], replace=True, size=weight)
        idx = rng.choice(range(num_qubits), replace=False, size=weight)

        # Do not include identity
        if set(letters) == set('I'):
            continue

        for i, p in zip(idx, letters):
            pauli[i] = p
        batch.add(''.join(pauli))

    return list(batch)