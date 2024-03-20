import jax
import numpy as np

from experiments import find_nonzero_pauli, find_fixed_angles
from pauli import all_two_body_pauli
from traps import LocalVQA


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
    assert np.allclose(vqa.expval(pauli, x), -1, atol=1e-5, rtol=1e-5)


def test_free_angles_consistency():
    num_qubits = 6
    num_layers = 10

    vqa = LocalVQA(num_qubits, num_layers)
    paulis = all_two_body_pauli(num_qubits)
    rng = np.random.default_rng(42)

    success, pauli, x = find_nonzero_pauli(
        vqa,
        paulis,
        [],
        [],
        rng)

    i_fixed = find_fixed_angles(vqa, pauli, x)

    # Check that the free parameters indeed do not affect the function value.
    y = vqa.random_parameters(num_samples=50, rng=rng)
    y[:, i_fixed] = x[i_fixed]

    values = jax.vmap(lambda xi: vqa.expval(pauli, xi))(y)
    assert np.allclose(values, -1, atol=1e-5, rtol=1e-5)