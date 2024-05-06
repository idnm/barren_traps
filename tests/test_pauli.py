from pauli import *


import jax
from pauli import *
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments import BPExperiment, ExactMinExperiment
from traps import LocalVQA

# jax.config.update("jax_enable_x64", True)

def test_exp():
    exp = ExactMinExperiment('test')

    qubits = (6,)
    layers = (50,)

    exp.run(qubits, layers, num_samples_per_circuit=1)

def test_dependent_paulis():

    num_qubits = 2
    paulis = all_two_body_pauli(num_qubits)

    pt = PauliTerms(paulis)

    pt.add_fixed_pauli('XX')
    assert pt.dependent_paulis == {'II', 'XX'}

    pt.add_fixed_pauli('XZ')
    assert pt.dependent_paulis == {'II', 'XX', 'XZ', 'IY'}

    pt.add_fixed_pauli('ZX')
    assert pt.dependent_paulis == {'II', 'XX', 'XZ', 'IY', 'ZX', 'YI', 'YY', 'ZZ'}

    pt.add_fixed_pauli('ZX')
    assert pt.dependent_paulis == {'II', 'XX', 'XZ', 'IY', 'ZX', 'YI', 'YY', 'ZZ'}

