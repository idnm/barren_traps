from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Union, List, Tuple

import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.quantum_info import Clifford, Pauli, StabilizerState, random_pauli
from qiskit.circuit import Parameter

from tqdm.auto import tqdm


class LocalVQA:
    def __init__(self, num_qubits: int, num_layers: int):

        assert num_qubits % 2 == 0
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Parameters for qiskit circuit
        self._x0 = [Parameter(f'x{n}') for n in range(self.num_qubits)]
        self._z0 = [Parameter(f'z{n}') for n in range(self.num_qubits)]
        self._x = [[Parameter(f'x{n}{l}') for n in range(self.num_qubits)] for l in range(self.num_layers)]
        self._z = [[Parameter(f'z{n}{l}') for n in range(self.num_qubits)] for l in range(self.num_layers)]

    @property
    def num_initial_parameters(self) -> int:
        return 2 * self.num_qubits

    @property
    def num_entangling_parameters(self) -> int:
        return 2 * self.num_qubits * self.num_layers

    @property
    def num_parameters(self) -> int:
        return self.num_initial_parameters + self.num_entangling_parameters

    @property
    def _flat_params(self) -> List[Parameter]:
        return self._x0 + self._z0 + [xnl for xn in self._x for xnl in xn] + [znl for zn in self._z for znl in zn]

    def _params_dict(self, params: np.ndarray) -> dict:
        return dict(zip(self._flat_params, params))

    @property
    def initial_circuit(self):
        qc = QuantumCircuit(self.num_qubits)
        for n in range(self.num_qubits):
            qc.rx(self._x0[n], n)
            qc.rz(self._z0[n], n)

        return qc

    def entangling_layer(self, x, z, start=0):
        qc = QuantumCircuit(self.num_qubits)
        for n in range(start, self.num_qubits + start, 2):
            i = n % self.num_qubits
            j = (n + 1) % self.num_qubits
            qc.cz(i, j)

        for n in range(self.num_qubits):
            qc.rx(x[n], n)
            qc.rz(z[n], n)

        return qc

    @property
    def all_entangling_layers(self):
        qc = QuantumCircuit(self.num_qubits)

        s = 0
        for xi, zi in zip(self._x, self._z):
            qc.compose(self.entangling_layer(xi, zi, start=s), inplace=True)
            s = 1 - s

        return qc

    def split_params(self, params: np.ndarray) -> Sequence[np.ndarray]:
        n = self.num_qubits
        x0 = params[:n]
        z0 = params[n: 2 * n]

        num_x = len(params[2 * n:]) // 2
        params = params[2 * n: 2 * n + num_x]
        z = params[2 * n + num_x:]

        return x0, z0, params, z

    @property
    def qiskit_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(self.initial_circuit, inplace=True)
        qc.compose(self.all_entangling_layers, inplace=True)
        return qc

    @property
    def penny_circuit(self):
        return qml.from_qiskit(self.qiskit_circuit)

    def expval(self, pauli_string: Union[str, Sequence[str]], x: np.ndarray, interface='jax') -> np.ndarray:
        if isinstance(pauli_string, str):
            pauli_strings = [pauli_string]
        else:
            pauli_strings = pauli_string

        observables = [qml.pauli.string_to_pauli_word(ps) for ps in pauli_strings]
        dev = qml.device('default.qubit', wires=self.num_qubits)

        @qml.qnode(dev, interface=interface)
        def expectation():
            self.penny_circuit(self._params_dict(x))
            return [qml.expval(obs) for obs in observables]

        if isinstance(pauli_string, str):
            return expectation()[0]
        else:
            return jnp.asarray(expectation())

    def expval_vmap(self, pauli_string: Union[str, Sequence[str]], x: np.ndarray) -> np.ndarray:
        return jax.vmap(lambda xi: self.expval(pauli_string, xi))(x)

    @staticmethod
    def _init_rng(rng: Union[np.random.Generator, int, None]) -> np.random.Generator:
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng(42)
        return rng

    def uniform_variance(self, pauli_string: Sequence[str], num_samples=50, rng: Union[np.random.Generator, int, None] = None):
        rng = self._init_rng(rng)

        x = rng.uniform(0, 2 * np.pi, size=(num_samples, self.num_parameters))
        values = jax.vmap(lambda xi: self.expval(pauli_string, xi))(x)
        sample_variance = values.var(axis=0)
        return sample_variance

    def clifford_variance(self, pauli_string: Sequence[str], num_samples=50, rng: Union[np.random.Generator, int, None] = None):
        rng = self._init_rng(rng)
        x = np.pi / 2 * rng.choice(range(4), size=(num_samples, self.num_parameters), replace=True)
        values = jax.vmap(lambda xi: self.expval(pauli_string, xi))(x)
        sample_variance = values.var(axis=0)
        return sample_variance

    def grad_expval(self, pauli_strings, k, params):
        ek = np.zeros_like(params)
        ek[k] = 1

        f = lambda p: self.expval(pauli_strings, params + np.pi / 2 * p)
        return (f(ek) - f(-ek)) / 2

    def hess_expval(self, pauli_strings, k, l, params):
        ek = np.zeros_like(params)
        el = np.zeros_like(params)
        ek[k] = 1
        el[l] = 1

        f = lambda p: self.expval(pauli_strings, params + np.pi / 2 * p)
        return (f(ek + el) - f(ek - el) - f(el - ek) + f(-ek - el)) / 4

    def random_parameters(self, num_samples=None, rng=42):
        rng = self._init_rng(rng)
        if num_samples is None:
            size = self.num_parameters
        else:
            size = (num_samples, self.num_parameters)

        return rng.uniform(0, 2 * np.pi, size=size)

    def random_clifford_parameters(self, num_samples=None, rng=42):
        rng = self._init_rng(rng)
        if num_samples is None:
            size = self.num_parameters
        else:
            size = (num_samples, self.num_parameters)

        return np.pi / 2 * rng.choice(range(4), size=size, replace=True)

    def good_random_clifford_parameters(self, pauli_string):
        _, _, x, z = self.split_params(self.random_clifford_parameters())

        dev = qml.device('qiskit.aer', wires=range(self.num_qubits))

        @qml.qnode(dev)
        def circ(x, z):
            self.all_entangling_layers(x, z)
            return qml.expval(qml.Identity(wires=range(self.num_qubits)))

        circ(x, z)
        qc = dev._circuit
        qc.remove_final_measurements()

        entangled_pauli_string = Pauli(pauli_string[::-1]).evolve(Clifford(qc)).to_label()

        x0 = []
        z0 = []
        for p in entangled_pauli_string[::-1]:
            if p == 'I':
                x0.append(np.random.choice([0, np.pi / 2]))
                z0.append(np.random.choice([0, np.pi / 2]))
            elif p == 'Z':
                x0.append(0)
                z0.append(np.random.choice([0, np.pi / 2]))
            if p == 'Y':
                x0.append(np.pi / 2)
                z0.append(0)
            elif p == 'X':
                x0.append(np.pi / 2)
                z0.append(np.pi / 2)

        good_params = np.concatenate([x0, z0, x, z])
        assert np.allclose(np.abs(self.expval(pauli_string, good_params)), 1, atol=1e-5, rtol=1e-5)
        return good_params

    def draw(self):
        qml.draw_mpl(self.circuit)(*self.random_parameters())
