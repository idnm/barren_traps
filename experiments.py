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
    def __init__(self, name: str, results: dict | None = None):
        self.name = name
        self.results = results

    def run(self):
        pass

    @staticmethod
    def _plot_results(results, marker='o'):
        cmap = plt.colormaps['viridis']
        qubits = sorted(list(results.keys()))

        for n in qubits:
            layers = list(results[n].keys())
            mean_variances = [results[n][l].mean() for l in layers]  # Average over different observables

            color = cmap(n / max(qubits))
            plt.scatter(layers, mean_variances, marker=marker, color=color)
            plt.axhline(2 ** -n, color=color, label=f'n={n}')

        plt.xlabel('Layers')
        plt.ylabel('Sample variance')
        plt.yscale('log', base=2)


    def save(self, path='results'):
        with open(path+'/'+self.name, 'wb') as f:
            pickle.dump(self.results, f)

    @classmethod
    def load(cls, name: str, path='results'):
        with open(path+'/'+name, 'rb') as f:
            results = pickle.load(f)
            exp = cls(name, results)
            return exp


class BPExperiment(Experiment):

    def __init__(self, name: str, results: dict | None = None):
        if results is None:
            results = (defaultdict(dict), defaultdict(dict), defaultdict(dict))
        super().__init__(name=name, results=results)

    def run(self, qubits: Sequence[int], layers: Sequence[int], num_samples: int, seed=42):
        rng = np.random.default_rng(seed)

        results_uniform, results_clifford, results_clifford_plus = self.results
        for num_qubits in tqdm(qubits):
            observables = all_two_body_pauli(num_qubits)
            for num_layers in tqdm(layers):
                vqa = LocalVQA(num_qubits, num_layers)

                # Uniform variance
                uniform_vars = vqa.uniform_variance(observables, num_samples=num_samples, rng=rng)
                results_uniform[num_qubits][num_layers] = uniform_vars

                # Clifford variance
                clifford_values = vqa.clifford_samples(observables, num_samples=num_samples, rng=rng)  # (num_samples, num_observables)
                clifford_vars = clifford_values.var(axis=0)
                results_clifford[num_qubits][num_layers] = clifford_vars

                # Clifford variance conditioned on the presence of non-zero paulis

                # Remove clifford points where all Paulis are zero
                cond_clifford_values = clifford_values[np.any(np.abs(clifford_values) > 0.5, axis=1)]

                # Before computing variances, one Pauli observable with non-zero exp value needs to be excluded.
                # Otherwise, we get biased results.
                # Simplest way to do this is to sort, and remove the last column (should be all ones)
                # Sorting does not affect the variance.

                i_sort = np.argsort(np.abs(cond_clifford_values), axis=1)
                cond_clifford_values = np.take_along_axis(cond_clifford_values, i_sort, axis=1)

                assert np.all(np.abs(cond_clifford_values[:, -1]) > 0.5) # check that the last column has no zeros.
                cond_clifford_values = cond_clifford_values[:, :-1]

                cond_clifford_vars = cond_clifford_values.var(axis=0)
                results_clifford_plus[num_qubits][num_layers] = cond_clifford_vars

                self.save()

    def plot_results(self):

        result_types = ('uniform', 'clifford', 'clifford+')
        markers = ('o', '^', '+')
        for result_type, marker, result in zip(result_types, markers, self.results):
            self._plot_results(result, marker)

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

    def _plot_results(self, marker='o', num_qubits: int = 0):
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
    expvals = jax.vmap(vqa.expval(paulis))(y)
    negative = np.argwhere(expvals < -0.99)
    if len(negative) == 0:
        return False, '', x

    # Index of a random negative expectation value
    i, j = rng.choice(negative)

    return True, paulis[j], y[i]


def find_indices_of_fixed_angles(
        vqa: LocalVQA,
        pauli: str,
        x: np.ndarray
) -> np.ndarray:

    """Find which angles at a clifford point are fixed.
    Assumes that vqa(pauli, x) = -1.

    Based on a simple observation that shifting any fixed angle by pi changes the sign of the expectation.
    """

    x_shifted = x + np.pi * np.eye(vqa.num_parameters)
    expvals = jax.vmap(vqa.expval([pauli]))(x_shifted)[:, 0]

    return np.argwhere(expvals > 0.99).squeeze()