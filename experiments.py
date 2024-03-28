import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Sequence, Tuple

import jax
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from pauli import all_local_two_body_pauli, all_two_body_pauli
from traps import LocalVQA


class Experiment:
    def __init__(self, name: str, results: dict | None = None, aux: dict | None = None):
        self.name = name
        if results is None:
            self.results = defaultdict(dict)
        else:
            self.results = results
        self.aux = aux

    def run(self):
        pass

    def plot_results(self, marker='o'):
        cmap = plt.colormaps['viridis']

        variances = self.results
        qubits = sorted(list(variances.keys()))

        for n in qubits:
            layers = list(variances[n].keys())
            mean_variances = [variances[n][l].mean() for l in layers]  # Average over different observables

            color = cmap(n / max(qubits))
            plt.scatter(layers, mean_variances, marker=marker, color=color)
            plt.axhline(2 ** -n, color=color, label=f'n={n}')

        plt.xlabel('Layers')
        plt.ylabel('Sample variance')
        plt.yscale('log', base=2)

        plt.title(self.name)


    def save(self, path='results'):
        with open(path+'/'+self.name, 'wb') as f:
            pickle.dump([self.results, self.aux], f)

    @classmethod
    def load(cls, name: str, path='results'):
        with open(path+'/'+name, 'rb') as f:
            results, aux = pickle.load(f)
            exp = cls(name=name, results=results, aux=aux)
            return exp


class BPExperiment(Experiment):
    def __init__(self, name: str, results: dict | None = None, aux: dict | None = None):
        assert 'method' in aux and aux['method'] in ('uniform', 'clifford')
        super().__init__(name=name, results=results, aux=aux)


    def run(self, qubits: Sequence[int], layers: Sequence[int], num_samples: int, seed=42):
        rng = np.random.default_rng(seed)
        for n in tqdm(qubits):
            observables = all_two_body_pauli(n)
            for l in tqdm(layers):
                vqa = LocalVQA(n, l)
                if self.aux['method'] == 'uniform':
                    var_function = vqa.uniform_variance
                elif self.aux['method'] == 'clifford':
                    var_function = vqa.clifford_variance

                vars = var_function(observables, num_samples=num_samples,
                                            rng=rng)  # (num_samples, num_observables)
                self.results[n][l] = vars
                self.save()


@dataclass
class ShallowingExperiment(Experiment):

    def run(self,
            num_qubits: int,
            num_layers: int,
            num_clifford_points: int,
            num_samples: int,
            fixing_schedule: Union[Sequence[float], None],
            seed: int = 42):

        rng = np.random.default_rng(seed)
        vqa = LocalVQA(num_qubits, num_layers)
        m = vqa.num_parameters

        if fixing_schedule is None:
            fixing_schedule = [int(m - m / 2 ** i) for i in range(int(np.log2(m)))]
        else:
            fixing_schedule = [int(m * f) for f in fixing_schedule]

        for num_fixed in tqdm(fixing_schedule):
            values = []
            for _ in tqdm(range(num_clifford_points)):
                x = rng.uniform(0, 2 * np.pi, size=(num_samples, m))
                x_cliff = np.pi / 2 * rng.choice(range(4), size=num_fixed, replace=True)
                i_cliff = rng.choice(np.arange(m), size=num_fixed, replace=False)
                x[:, i_cliff] = x_cliff

                observables = all_local_two_body_pauli(num_qubits)
                values.append(jax.vmap(lambda xi: vqa._expval_func(observables, xi))(x))

            self.results[num_fixed] = np.asarray(values).reshape(num_clifford_points * len(observables), num_samples)
            self.save()

    def plot_results(self, marker='o', num_qubits: int = 0):
        expvals = self.results
        fixed = sorted(list(expvals.keys()))

        for f in fixed:
            values = self.results[f]
            sample_variance = [np.var(v) for v in values]
            for var in sample_variance:
                plt.scatter([f], var, marker=marker)

        if num_qubits:
            plt.axhline(2 ** -num_qubits)

        plt.xlabel('Number of fixed parameters')
        plt.ylabel('Sample variance')
        plt.yscale('log', base=2)



@dataclass
class ExactMinExperiment(Experiment):

    def run(self,
            num_qubits,
            num_layers,
            ):
        pass


def find_nonzero_pauli(
        vqa: LocalVQA,
        paulis: Sequence[str],
        x: np.ndarray,
        i_fixed: np.ndarray,
        rng: np.random.Generator,
        num_samples: int = 100
) -> Tuple[bool, str, np.ndarray]:
    """
    Try finding a pauli string from the given list, which has a non-zero expectation value, subject to constraints.
    The constraints are that angles indexed by i_fixed are fixed to valued x_fixed.

    Apparently, there is no efficient way to do this. An inefficient way is to sample many clifford points and look
    at many observables at once, choosing any one that works.
    """

    y = vqa.random_clifford_parameters(num_samples, rng)
    if len(i_fixed):
        y[:, i_fixed] = x[i_fixed]
    expvals = jax.vmap(vqa._expval_func(paulis))(y)
    negative = np.argwhere(expvals < -0.99)
    if len(negative) == 0:
        return False, '', x

    # Index of a random negative expectation value
    i, j = rng.choice(negative)

    return True, paulis[j], y[i]


def find_fixed_angles(
        vqa: LocalVQA,
        pauli: str,
        x: np.ndarray
) -> np.ndarray:

    """Find which angles at a clifford point are fixed.
    Assumes that vqa(pauli, x) = -1.

    Based on a simple observation that shifting any fixed angle changes the sign of the expectation.
    """

    x_shifted = x + np.pi * np.eye(vqa.num_parameters)
    expvals = jax.vmap(lambda xi: vqa._expval_func(pauli, xi))(x_shifted)

    return np.argwhere(expvals > 0.99).squeeze()