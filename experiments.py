import itertools
import pickle
from collections import defaultdict
from functools import partial
from typing import Sequence, Tuple

import jax
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from pauli import pauli_batch, all_local_two_body_pauli, all_one_body_pauli, all_independent_local_two_body_pauli, \
    all_independent_two_body_pauli, PauliTerms, all_two_body_pauli
from traps import LocalVQA
import matplotlib.lines as mlines



class Experiment:
    """Base class for experiments providing saving, loading, and plotting methods."""

    def __init__(self, name: str, results: dict | None = None):
        self.name = name
        self.results = results

    def run(self):
        pass

    @staticmethod
    def _color(num_qubits, max_qubits):
        cmap = plt.colormaps['viridis']
        return cmap(num_qubits / max_qubits)

    @staticmethod
    def _plot_lines(qubits):
        for n in qubits:
            color = Experiment._color(n, max(qubits))
            plt.axhline(2 ** -n, color=color, label=f'n={n}')

    @staticmethod
    def _plot_results(results, marker='o', offset=0.):
        qubits = sorted(list(results.keys()))

        for n in qubits:
            layers = list(results[n].keys())
            mean_variances = np.asarray([results[n][l].mean() for l in layers])  # Average over different observables
            # mean_variances = mean_variances[mean_variances > 1e-5] # Truncate variances that are too small

            color = Experiment._color(n, max(qubits))
            plt.scatter([l+offset for l in layers], mean_variances, marker=marker, color=color, edgecolors='black', alpha=0.7)

        Experiment._plot_lines(qubits)

        plt.xlabel('Layers')
        plt.ylabel('Sample variance')
        plt.yscale('log', base=2)
        plt.ylim(2 ** -(max(qubits) + 0.9), 2 ** -(min(qubits) - 0.9))


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
            observables = all_local_two_body_pauli(num_qubits)
            for num_layers in tqdm(layers):
                vqa = LocalVQA(num_qubits, num_layers)

                # Uniform variance
                x_uniform = 2 * np.pi * rng.uniform(size=(num_samples, vqa.num_parameters))
                uniform_values = jax.vmap(vqa.expval(observables))(x_uniform)  # (num_samples, num_observables)
                uniform_vars = uniform_values.var(axis=0)
                results_uniform[num_qubits][num_layers] = uniform_vars

                # Clifford variance
                x_clifford = np.pi / 2 * rng.choice(range(4), size=(num_samples, vqa.num_parameters), replace=True)
                clifford_values = jax.vmap(vqa.expval(observables))(x_clifford) # (num_samples, num_observables)
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
        markers = ('o', 's', '^')
        offsets = (-1, 0., 1)
        labels = ('Uniform', 'Clifford', 'Conditioned Clifford')
        handles = []
        for result_type, marker, result, offset, label in zip(result_types, markers, self.results, offsets, labels):
            self._plot_results(result, marker, offset=offset)
            handles.append(mlines.Line2D([], [], marker=marker, markerfacecolor='None', markeredgecolor='black', linestyle='None',
                                  markersize=10, label=label))

        plt.legend(handles=handles, loc=(0.6, 1.05))


class ExactMinExperiment(Experiment):
    """
    Experiment that attempts to find an exact local minimum of a VQA.
    First it greedily looks for a bunch of Pauli operators that can be simultaneously minimized.
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
            num_test_uniform_points: int = 10,
            num_test_clifford_points: int | None = None,
            max_grads: int | None = None,
            # max_obs: int | None = None,
            seed: int = 42):

        """
        For each circuit specified by `num_qubits` from `qubits`, and `num_layers` from `layers` construct a VQA.
        Then propose an exact local minimum for this VQA, by greedily adding Pauli operators that can be
        simultaneously minimized. Pauli's operators are under weight=4 are samples randomly.

        Find a split (fixed_angles, free_angles) corresponding to the proposed exact minimum.
        Finally, access whether a given point is likely be an exact local minimum, to gauge how likely is a generic
        Pauli op to have loss function identically zero w.r.t. free_angles, as well as identically zero derivatives
        w.r.t. fixed_angles.

        num_samples_per_circuit: how many times to repeat the procedure for a given circuit (num_qubits, num_layers)
        num_test_clifford_points: how many clifford points to try to find the next non-zero Pauli.
        num_test_uniform_points: how many uniform points to sample to check if the loss is exactly zero.
        max_grads: only take derivatives w.r.t. to this number of fixed_angles. Otherwise, deep circuits become too costly.
        max_obs: likewise, the amount of possible observables becomes too large and needs to be limited.
        """

        rng = np.random.default_rng(seed)
        for num_qubits, num_layers in tqdm(zip(qubits, layers)):
                vqa = LocalVQA(num_qubits, num_layers)
                observables = all_two_body_pauli(num_qubits)

                if num_test_clifford_points is None:
                    num_test_clifford_points = int(30 * 2 ** num_qubits / len(observables))
                if max_grads is None:
                    max_grads = int(30 * 2 ** num_qubits / len(observables))

                for _ in tqdm(range(num_samples_per_circuit)):
                    pauli_terms = PauliTerms(observables)
                    nonzero_grad_rate = self.single_run(
                        vqa,
                        pauli_terms,
                        num_test_clifford_points,
                        num_test_uniform_points,
                        max_grads,
                        rng)

                    self.results[num_qubits][num_layers]['paulis'].append(pauli_terms)
                    self.results[num_qubits][num_layers]['rates'].append(nonzero_grad_rate)
                    self.save()

                    jax.clear_caches()



    def single_run(self,
                   vqa: LocalVQA,
                   pauli_terms: PauliTerms,
                   num_test_clifford_points: int,
                   num_test_uniform_points: int,
                   max_grads: int,
                   rng: np.random.Generator
                   ) -> float:
        """
        Propose an exact minimum and compute the proportion of non-zero values/gradients for a given circuit.
        """

        print('\n Looking for exact minimum')
        x, i_fixed = self.propose_exact_minimum(vqa, pauli_terms, num_test_clifford_points, rng)

        print('\n Computing rates')
        nonzero_grad_rate = self.nonzero_gradient_rate(
            vqa,
            pauli_terms.remaining_paulis,
            x,
            i_fixed,
            num_test_uniform_points,
            max_grads,
            rng
        )

        return nonzero_grad_rate

    @staticmethod
    def propose_exact_minimum(
            vqa: LocalVQA,
            pauli_terms: PauliTerms,
            num_test_clifford_points: int,
            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:

        """Proposes an exact minimum by greedily finding a subset of observables that are
        simultaneously minimized at some Clifford point.

        Works as follows. Sample a batch of clifford points and compute expectations of each `observable`.
        Select a random clifford point and a random observable that has exp=-1 at that clifford point.
        Split the clifford point into (fixed angles, free_angles).

        Repeat by sampling from the free_angles, and finding the next exp=-1 observable, and the next free_angles.
        Stop when no Pauli operators have expectation value -1 at the samples Clifford points.
        """

        # observables = observables.copy()
        
        # Initialize all angles to 0, and indices of the fixed_angles to an empty array.
        # Later, x[i_fixed] are the values of fixed_angles, the rest are free_angles.
        x = np.zeros(vqa.num_parameters)
        i_fixed = np.array([], dtype=int)

        while True:
            # For efficiency, batch observables while looking for the next non-zero paulis.
            batch_size = 50
            candidate_paulis = pauli_terms.remaining_paulis.copy()
            rng.shuffle(candidate_paulis)
            for paulis in [candidate_paulis[i:i+batch_size] for i in range(0, len(candidate_paulis), batch_size)]:
                pauli, x = ExactMinExperiment.find_nonzero_pauli(vqa, paulis, x, i_fixed, num_test_clifford_points, rng)
                if pauli:
                    break
            else:
                break

            # indices of fixed_angles w.r.t. to the newly added Pauli
            new_fixed = ExactMinExperiment.find_indices_of_fixed_angles(vqa, pauli, x)
            # indices of fixed_angles w.r.t. to all of the non-zero Pauli
            i_fixed = np.unique(np.concatenate([i_fixed, new_fixed]))

            pauli_terms.add_fixed_pauli(pauli)
            # observables.remove(pauli)
            # nonzero_paulis.append(pauli)
            print(f'paulis {pauli_terms.fixed_paulis}, len(i_fixed) {len(i_fixed)}({vqa.num_parameters})')

        return x, i_fixed

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
        To test this, for each configuration (pauli, derivative w.r.t. a fixed_angle) we sample from free_angles uniformly.
        The expectation is that the resulting values are zero within machine precision when the loss functions
        (and their gradients) are identically vanishing.

        Note that `gradient values` also include the value of the function itself.
        Also, we do not actually compute the gradients. These are given by dL = L(+pi/2) - L(-pi/2). Instead,
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
        # For deep circuits, there are too many angles. So only take the derivatives wrt to `max_grads` number of them.
        num_grads = min(max_grads, len(shifts))
        random_subset = rng.choice(np.arange(len(shifts)), size=num_grads, replace=False)
        shifts = shifts[random_subset]

        # Add zero shift to probe the value of the function as well.
        shifts = np.vstack([np.zeros(vqa.num_parameters), shifts])

        # Array of angles probing both, gradients in the direction of fixed angles, and random values
        # in the direction of free angles.
        # (num_uniform_samples, num_grads + 1, num_angles)

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
        """
        We are given a candidate local minimum defined by (x, i_fixed).
        We want to check how often other Pauli operators and their gradients (w.r.t. to the fixed angles) are
        exactly zero.

        num_test_uniform_points: how many uniform points to sample to decide that the function is exactly zero.
        max_grads: maximum number of grad component to consider.
        """
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
        The constraints are that angles indexed by i_fixed are fixed to valued x[i_fixed].

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

    def plot_results(self):
        results = self.results
        qubits = sorted(list(results.keys()))

        self._plot_lines(qubits)

        for num_qubits in qubits:
            num_layers = list(results[num_qubits].keys())[0]
            rates = np.asarray(results[num_qubits][num_layers]['rates'])
            rates = rates[rates < 0.9]  # Discards failed runs with no fixed Paulis found

            color = Experiment._color(num_qubits, max(qubits))
            plt.scatter([num_qubits + 0.1] * len(rates), rates, color=color, marker='o', edgecolors='black', alpha=0.8)
            plt.scatter([num_qubits - 0.1], rates.mean(), color=color, marker='s', edgecolors='black', s=70)

        handle_samples = mlines.Line2D([], [], marker='o', markeredgecolor='black', markerfacecolor='None', linestyle='None',
                                  markersize=10, label='Individual samples')
        handle_average = mlines.Line2D([], [], marker='s', markeredgecolor='black', markerfacecolor='None', linestyle='None',
                                  markersize=10, label='Sample average')

        plt.ylim(2 ** -(max(qubits)+0.9), 2 ** -(min(qubits) - 0.9))
        plt.yscale('log', base=2)

        plt.ylabel('Vanishing probability')
        plt.xlabel('Number of qubits')

        plt.legend(handles=[handle_samples, handle_average], loc=(0.6, 1.05))

