from ast import Str
from typing import List, Sequence

import numpy as np
import qiskit.quantum_info
from qiskit.quantum_info import Pauli


def random_local_two_body_pauli(num_qubits: int) -> str:
    """Random Pauli operator of weight 2, supported on adjacent qubits."""

    p1, p2 = np.random.choice(['X', 'Y', 'Z'], size=2, replace=True)
    i = np.random.choice(range(num_qubits - 1))
    return 'I' * i + p1 + p2 + 'I' * (num_qubits - i - 2)


def all_one_body_pauli(num_qubits: int) -> List[str]:
    """All weight=1 Pauli operators"""
    res = []
    for n in range(num_qubits):
        for p in ['X', 'Y', 'Z']:
            s = ['I'] * num_qubits
            s[n] = p
            res.append(''.join(s))

    return res


def all_local_two_body_pauli(num_qubits: int) -> List[str]:
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


def all_independent_local_two_body_pauli(num_qubits: int) -> List[str]:
    """Generate all two-body, local, nearest neighbour Pauli strings, without 'Y'.
     """

    res = []
    for n in range(1, num_qubits):
        for p1 in ['X', 'Z']:
            for p2 in ['X', 'Z']:
                pauli_string = 'I' * (n - 1) + p1 + p2 + 'I' * (num_qubits - n - 1)
                res.append(pauli_string)
    return res


def all_independent_two_body_pauli(num_qubits: int) -> List[str]:
    """Generate all two-body, local, nearest neighbour Pauli strings, without 'Y'.
     """
    res = []
    for n1 in range(num_qubits-1):
        for n2 in range(n1+1, num_qubits):
            for p1 in ['X', 'Z']:
                for p2 in ['X', 'Z']:
                    s = ['I'] * num_qubits
                    s[n1] = p1
                    s[n2] = p2
                    res.append(''.join(s))
    return res


def all_two_body_pauli(num_qubits: int) -> List[str]:
    """Generate all two-body Pauli strings.

     """
    res = []
    for n1 in range(num_qubits-1):
        for n2 in range(n1+1, num_qubits):
            for p1 in ['X', 'Y', 'Z']:
                for p2 in ['X', 'Y', 'Z']:
                    s = ['I'] * num_qubits
                    s[n1] = p1
                    s[n2] = p2
                    res.append(''.join(s))
    return res


class PauliTerms:
    def __init__(self, paulis: List[str]):
        self.paulis = paulis.copy()
        self.remaining_paulis = paulis.copy()
        self.fixed_paulis = []
        self.dependent_paulis = {'I'*len(paulis[0])}

    def add_fixed_pauli(self, pauli):
        self.fixed_paulis.append(pauli)
        self.update_dependent_paulis(pauli)
        self.remaining_paulis = [p for p in self.remaining_paulis if p not in self.dependent_paulis]

    def update_dependent_paulis(self, pauli):
        new_dependent = []
        for p in self.dependent_paulis:
            p_prod = Pauli(p).compose(Pauli(pauli))
            p_prod.phase = 0
            new_dependent.append(p_prod.to_label())

        self.dependent_paulis = self.dependent_paulis.union(set(new_dependent))


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