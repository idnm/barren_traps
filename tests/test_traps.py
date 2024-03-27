import pytest
import jax
import jax.numpy as jnp
import numpy as np
# from catalyst import qjit, for_loop

from pauli import random_local_two_body_pauli, all_two_body_pauli
from traps import LocalVQA

import pennylane as qml

def test_penny_circuit():

    num_qubits = 4
    num_layers = 3
    vqa = LocalVQA(num_qubits, num_layers)

    dev0 = qml.device('lightning.qubit', wires=num_qubits)
    @qml.qnode(dev0)
    def circ0(x):
        vqa.penny_circuit_from_qiskit(vqa._params_dict(x))
        return qml.state()

    dev1 = qml.device('lightning.qubit', wires=num_qubits)
    @qml.qnode(dev1)
    def circ1(x):
        vqa.penny_circuit(x)
        return qml.state()

    @qml.qjit(static_argnums=(0, 1))
    def expval(observables, circuit, x):
        state = circuit(x).reshape([2]*num_qubits)
        return jnp.asarray([qml.devices.qubit.measure(qml.expval(qml.pauli.string_to_pauli_word(obs)), state) for obs in observables])

    observables = tuple(all_two_body_pauli(num_qubits))
    x = vqa.random_parameters()

    e0 = expval(observables[:1], circ0, x)
    e1 = expval(observables[:1], circ1, x)

    assert np.allclose(e0, e1, atol=1e-5, rtol=1e-5)


def test_grads(num_qubits, num_layers):
    vqa = LocalVQA(num_qubits, num_layers)
    obs = random_local_two_body_pauli(vqa.num_qubits)

    p0 = vqa.random_clifford_parameters()
    f = lambda params: vqa.expval(obs, params, interface='jax')

    jax_grads = jax.grad(f)(p0)
    shift_grads = [vqa.grad_expval(obs, k, p0) for k in range(vqa.num_parameters)]

    jax_hess = jax.hessian(f)(p0)
    shift_hess = [[vqa.hess_expval(obs, k, l, p0) for k in range(vqa.num_parameters)] for l in
                  range(vqa.num_parameters)]

    assert np.allclose(jax_grads, shift_grads, atol=1e-5, rtol=1e-5)
    assert np.allclose(jax_hess, shift_hess, atol=1e-5, rtol=1e-5)
