import jax
import jaxopt
import numpy as np
import optax

from experiments import find_nonzero_pauli, find_indices_of_fixed_angles, BPExperiment
from pauli import all_two_body_pauli
from traps import LocalVQA


def test_exp():
    exp1 = BPExperiment.load('test1')
    qubits = (2,)
    layers = (10, 50)
    num_samples = 5

    exp1.run(qubits, layers, num_samples)


def test_find_nonzero_consistency():
    num_qubits = 6
    num_layers = 20

    vqa = LocalVQA(num_qubits, num_layers)
    paulis = all_two_body_pauli(num_qubits)
    rng = np.random.default_rng(42)

    success, pauli, x = find_nonzero_pauli(
        vqa,
        paulis,
        [],
        [],
        rng)

    assert success

    # Check that the pauli found indeed has the correct expectation value at the point found.
    assert np.allclose(vqa.expval([pauli])(x), -1, atol=1e-5, rtol=1e-5)


def test_free_angles_consistency():
    """ Check that angles returned by the function 'fixed_anlges' are correct.
    More precisely, check that the angles not considered to be 'fixed_angles' do not change the function value"
    """

    num_qubits = 2
    num_layers = 4

    vqa = LocalVQA(num_qubits, num_layers)
    paulis = all_two_body_pauli(num_qubits)
    rng = np.random.default_rng(42)

    print('\n')
    print('defined vqa')
    success, pauli, x = find_nonzero_pauli(
        vqa,
        paulis,
        [],
        [],
        rng)
    print('found nonzero')
    i_fixed = find_indices_of_fixed_angles(vqa, pauli, x)

    print('found fixed')
    y = vqa.random_parameters(num_samples=50, rng=rng)
    y[:, i_fixed] = x[i_fixed]

    values = jax.vmap(vqa.expval([pauli]))(y)[:, 0]
    print('evaluated values')
    assert np.allclose(values, -1, atol=1e-5, rtol=1e-5)

test_free_angles_consistency()

def test_full_opt():
    rng = np.random.default_rng(42)

    num_qubits = 2
    num_layers = 2
    vqa = LocalVQA(num_qubits, num_layers)

    observables = all_two_body_pauli(num_qubits)
    coefficients = rng.uniform(low=-1, high=1, size=len(observables))

    def loss(x):
        values = vqa._expval_func(observables, x)
        return (values * coefficients).sum()

    learning_rate = 0.01
    num_iterations = 100
    num_initial_points = 100
    opt = optax.adam(learning_rate)

    solver = jaxopt.OptaxSolver(loss, opt, maxiter=num_iterations, jit=True)

    x0 = 2 * np.pi * rng.uniform(size=(num_initial_points, vqa.num_parameters))
    values = jax.vmap(solver.run)(x0)