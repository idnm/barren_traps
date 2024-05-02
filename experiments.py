import itertools
import pickle
from collections import defaultdict
from functools import partial
from typing import Sequence, Tuple

import jax
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from pauli import pauli_batch
from traps import LocalVQA


class Experiment:
    """Base class for experiments providing saving, loading, and plotting methods."""

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
        # Results for this experiment consists of three separate statistics:
        # over uniform distribution, clifford points, and clifford points conditioned on existence of non-zero Pauli.
        if results is None:
            results = (defaultdict(dict), defaultdict(dict), defaultdict(dict))
        super().__init__(name=name, results=results)

    def run(self, qubits: Sequence[int], layers: Sequence[int], num_paulis: int, num_samples: int, seed=42):
        """
        qubits: a sequence of circuit sizes to run the experiment on
        layers: a sequence of depths to run the experiment on
        num_paulis: how many pauli observables to include for each (num_qubits, num_layers) pair.
        num_samples: how many points (uniform of clifford) to sample for each (num_qubits, num_layers) pair.
        """
        rng = np.random.default_rng(seed)

        results_uniform, results_clifford, results_clifford_plus = self.results
        for num_qubits in tqdm(qubits):
            observables = pauli_batch(num_qubits, num_paulis, max_weight=4, seed=rng)
            for num_layers in tqdm(layers):
                vqa = LocalVQA(num_qubits, num_layers)

                # Uniform variance
                x_uniform = 2 * np.pi * rng.uniform(size=(num_samples, vqa.num_parameters))
                uniform_values = jax.vmap(vqa.expval(observables))(x_uniform)  # (num_samples, num_observables)
                uniform_vars = uniform_values.var(axis=0)
                results_uniform[num_qubits][num_layers] = uniform_vars

                # Clifford variance
                x_clifford = np.pi / 2 * rng.choice(range(4), size=(num_samples, vqa.num_parameters), replace=True)
                clifford_values = jax.vmap(vqa(observables))(x_clifford) # (num_samples, num_observables)
                clifford_vars = clifford_values.var(axis=0)
                results_clifford[num_qubits][num_layers] = clifford_vars

                # Clifford variance conditioned on the presence of non-zero paulis

                # Remove clifford points where all Paulis are zero
                # (0.5 is an arbitrary numeric cutoff (all values are either near 0 or near +-1 )
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


class ExactMinExperiment(Experiment):
    """
    Experiment that attempts to find an exact local minimum of a VQA.
    First it greedily looks for a number of Pauli operators that can be simultaneously minimized.
    This results in a split (fixed_angles, free_angles).

    Then it tests how likely is that a new, unrelated Pauli op, has loss function identically zero as a function of free_angles.
    It also tests how likely are the gradients w.r.t. fixed_angles to be identically zero as well.
    """

    def __init__(self, name: str, results: dict | None = None):
        if results is None:
            # A way to make default dict[dict[dict[list]]] structure that is pickl'able.
            results = defaultdict(partial(defaultdict, partial(defaultdict,list)))

        super().__init__(name, results)

    def run(self,
            qubits: Sequence[int],
            layers: Sequence[int],
            num_samples_per_circuit: int = 10,
            num_test_clifford_points: int = 10,
            num_test_uniform_points: int = 10,
            max_grads: int = 100,
            max_obs: int = 100,
            seed: int = 42):

        """
        For each circuit specified by num_qubits from qubits, and num_layers from layers construct a VQA.
        Then propose an exact local minimum for this VQA, by greedily adding Pauli operators that can be
        simultaneously minimized.
        Pauli operators are taken from all possible two-body operators, not necessarily local.

        Find a split (fixed_angles, free_angles) corresponding to the proposed exact minimum.
        Finally, to access whether a given point is likely be an exact local minimum, gauge how likely is a generic
        Pauli op to have loss function identically zero w.r.t. free_angles, as well as identically zero derivatives
        w.r.t. fixed_angles.

        num_samples_per_circuit: how many times to repeat the procedure for a given circuit (num_qubits, num_layers)
        num_test_clifford_points: how many clifford points to try to find the next non-zero Pauli.
        num_test_uniform_points: how many uniform points to sample to check if the loss is exactly zero.
        max_grads: only take derivatives w.r.t. to this number of fixed_angles. Otherwise, deep circuits become too costly.
        max_obs: likewise, the amount of possible observables becomes too large and needs to be limited.
        """

        rng = np.random.default_rng(seed)
        for num_qubits in tqdm(qubits):
            for num_layers in tqdm(layers):
                vqa = LocalVQA(num_qubits, num_layers)
                observables = pauli_batch(num_qubits)
                for _ in tqdm(range(num_samples_per_circuit)):
                    nonzero_paulis, nonzero_grad_rate = self._run(
                        vqa,
                        observables,
                        num_test_clifford_points,
                        num_test_uniform_points,
                        max_grads,
                        max_obs,
                        rng)

                    self.results[num_qubits][num_layers]['paulis'].append(nonzero_paulis)
                    self.results[num_qubits][num_layers]['rates'].append(nonzero_grad_rate)
                    self.save()

                    jax.clear_caches()

    def _run(self,
             vqa: LocalVQA,
             observables: Sequence[str],
             num_test_clifford_points: int,
             num_test_uniform_points: int,
             max_grads: int,
             max_obs: int,
             rng: np.random.Generator
            ) -> Tuple[Sequence[str], float]:
        """
        Compute the proportion of non-zero values/gradients for a given circuit.
        """

        print('\n Looking for exact minimum')
        fixed_paulis, x, i_fixed = self.propose_exact_minimum(vqa, observables, num_test_clifford_points, rng)
        remaining_paulis = list(set(observables) - set(fixed_paulis))

        # assert (len(all_two_body_pauli(vqa.num_qubits)) - len(fixed_paulis)) == len(remaining_paulis)x`
        if len(remaining_paulis) > max_obs:
            remaining_paulis = rng.choice(remaining_paulis, size=max_obs, replace=False)

        print('\n Computing rates')
        nonzero_grad_rate = self.nonzero_gradient_rate(
            vqa,
            remaining_paulis,
            x,
            i_fixed,
            num_test_uniform_points,
            max_grads,
            rng
        )
        # print(f'\n rate {nonzero_grad_rate}')

        return fixed_paulis, nonzero_grad_rate

    def propose_exact_minimum(
            self,
            vqa: LocalVQA,
            observables: Sequence[str],
            num_test_clifford_points: int,
            rng: np.random.Generator) -> Tuple[Sequence[str], np.ndarray, np.ndarray]:

        """Proposes an exact minimum by greedily finding a subset of observables that are
        simultaneously minimized at some Clifford point.

        Works as follows. Sample a batch of clifford points and compute expectations of each `observable`.
        Select a random clifford point and a random observable that has exp=-1 at that clifford point.
        Split the clifford point into (fixed angles, free_angles).

        Repeat by sampling from the free_angles, and finding the next exp=-1 observable, and the next free_angles.
        Stop when no Pauli operators have expectation value -1 at the samples Clifford points.
        """

        observables = observables.copy()
        
        # Initialize all angles to 0, and indices of the fixed_angle to an empty array.
        # Later, x[i_fixed] are the values of fixed_angles, the rest are free_angles.
        x = np.zeros(vqa.num_parameters)
        i_fixed = np.array([], dtype=int)

        nonzero_paulis = []
        while True:
            rng.shuffle(observables)
            batch_size = max(len(observables) // 10, 10)
            for observables_batch in itertools.batched(observables, batch_size):
                pauli, x = self.find_nonzero_pauli(vqa, observables_batch, x, i_fixed, num_test_clifford_points, rng)
                if pauli:
                    break
            else:
                break
            
            # indices of fixed_angles w.r.t. to the newly added Pauli
            new_fixed = self.find_indices_of_fixed_angles(vqa, pauli, x)
            # indices of fixed_angles w.r.t. to all of the non-zero Pauli
            i_fixed = np.unique(np.concatenate([i_fixed, new_fixed]))
            
            observables.remove(pauli)
            nonzero_paulis.append(pauli)
            print(f'paulis {nonzero_paulis}, len(i_fixed) {len(i_fixed)}({vqa.num_parameters})')

        return nonzero_paulis, x, i_fixed

    @staticmethod
    def _gradient_values(
            vqa: LocalVQA,
            paulis: Sequence[str],
            x: np.ndarray,
            i_fixed: np.ndarray,
            num_test_uniform_points: int,
            max_grads: int,
            rng: np.random.Generator
    ) -> np.ndarray:
        """
        We are given a VQA, a list of Pauli operators, and an angle configuration split as (fixed_angles, free_angles).

        We want to test whether expvals of the Pauli operators at fixed_angles, as well as their derivatives
        w.r.t. fixed_angles, depend on free_angles.
        For this, for each configuration (pauli, derivative w.r.t. a fixed_angle) we sample from free_angles uniformly.
        The expectation is that the resulting values are zero within machine precision when the loss functions
        (and their gradients) are identically vanishing.

        Note that `gradient values` also include the value of the function itself.
        Also, we do not actually compute the gradients. These are given by d L = L(+pi/2) - L(-pi/2). Instead,
        we only compute L(pi/2). These 'half-gradients' being zero is sufficient for the full gradient to be zero.

        """

        # Initialize fixed angles the way they should be, and free angles randomly.
        # (num_uniform_samples, num_angles)
        x_rnd = vqa.random_parameters(num_samples=num_test_uniform_points, rng=rng)
        x_rnd[:, i_fixed] = x[i_fixed]

        # Generate shifts corresponding to 'half-gradients' w.r.t. fixed_angles
        # (max_grads + 1, num_angles)
        shifts = np.pi / 2 * np.eye(vqa.num_parameters)
        shifts = shifts[i_fixed] # only take gradients (shifts) in the directions of fixed_angles
        # For deep circuits, there are too many angles. So only take the derivatives wrt to a max_grads number of them.
        num_grads = min(max_grads, len(shifts))
        random_subset = rng.choice(np.arange(len(shifts)), size=num_grads, replace=False)
        shifts = shifts[random_subset]

        # Zero shift to probe the value of the function as well.

        shifts = np.vstack([np.zeros(vqa.num_parameters), shifts])

        # Array of angles probing both, gradients in the direction of fixed angles, and random values
        # in the direction of free angles.
        # (num_uniform_samples, num_fixed_angles + 1, num_angles)

        x_grad = x_rnd.reshape(num_test_uniform_points, 1, vqa.num_parameters) + shifts

        return jax.vmap(jax.vmap(vqa.expval(paulis)))(x_grad)

    def nonzero_gradient_rate(
            self,
            vqa: LocalVQA,
            paulis: Sequence[str],
            x: np.array,
            i_fixed: np.array,
            num_test_uniform_points: int,
            max_grads: int,
            rng: np.random.Generator
    ) -> float:

        values = self._gradient_values(vqa, paulis, x, i_fixed, num_test_uniform_points, max_grads, rng)

        # Select pairs (gradient, observable) which have zero values (within machine precision) at all sampled points.
        all_nonzero = np.all(np.abs(values) > 1e-5, axis=0)
        rate = np.count_nonzero(all_nonzero) / all_nonzero.size

        vars = values.var(axis=0)
        vars_rate = np.count_nonzero(vars > 1e-5) / vars.size

        values_values = values[:, 0, :]

        values_values_rate = np.count_nonzero(np.abs(values_values) > 0.5) / values_values.size

        vars_values = values_values.var(axis=0)
        vars_values_rate = np.count_nonzero(vars_values > 1e-5) / vars_values.size

        print(f'rate {rate}, values rate {values_values_rate}, vars rate {vars_rate}, vars (values) rate {vars_values_rate}')

        return rate

    @staticmethod
    def find_nonzero_pauli(
            vqa: LocalVQA,
            paulis: Sequence[str],
            x: np.ndarray,
            i_fixed: np.ndarray,
            num_samples: int,
            rng: np.random.Generator
    ) -> Tuple[str, np.ndarray]:
        """
        Try finding a pauli string from the given list, which has a non-zero expectation value, subject to constraints.
        The constraints are that angles indexed by i_fixed are fixed to valued x_fixed.

        Apparently, there is no efficient way to do this. An inefficient way is to sample many clifford points and look
        at many observables at once, choosing any one that works.
        """

        y = vqa.random_clifford_parameters(num_samples, rng)
        if len(i_fixed):
            y[:, i_fixed] = x[i_fixed]
        expvals = jax.vmap(vqa.expval(paulis))(y)  # (num_samples, num_paulis)
        negative = np.argwhere(expvals < -0.99)
        if len(negative) == 0:
            return '', x

        # Indices of all angles with non-zero observables.
        # Choosing only from unique ones should remove the bias towards shallower circuits.
        i_unique = np.unique(negative[:, 0])
        i = rng.choice(i_unique)
        i, j = rng.choice(negative[negative[:, 0] == i])

        return paulis[j], y[i]

    @staticmethod
    def find_indices_of_fixed_angles(
            vqa: LocalVQA,
            pauli: str,
            x: np.ndarray
    ) -> np.ndarray:

        """Find which angles at a Clifford point are fixed.
        Assumes that vqa(pauli, x) = -1.

        Based on a simple observation that shifting any fixed angle by pi changes the sign of the expectation.
        """

        x_shifted = x + np.pi * np.eye(vqa.num_parameters)
        expvals = jax.vmap(vqa.expval([pauli]))(x_shifted)[:, 0]

        return np.argwhere(expvals > 0.99).squeeze()