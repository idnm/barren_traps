import jax
import numpy as np

from pauli import random_local_two_body_pauli
from traps import LocalVQA


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
