import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments import BPExperiment, ExactMinExperiment

qubits = (10, )
layers = (100, )

num_samples_per_circuit = 20
num_test_clifford_points = 20
num_test_uniform_points = 10
max_grads = 20
max_obs = 50

exp_exact = ExactMinExperiment.load('main')
exp_exact.run(
    qubits,
    layers,
    num_samples_per_circuit,
    num_test_clifford_points,
    num_test_uniform_points,
    max_grads,
    max_obs,
    seed=43)




# import jaxopt
# import matplotlib.pyplot as plt
# import numpy as np
# import optax
#
# from experiments import BPExperiment, ExactMinExperiment
# from pauli import all_local_two_body_pauli, all_two_body_pauli
# from traps import LocalVQA
#
# qubits = (8, )
# layers = (800, )
#
# num_samples_per_circuit = 10
# num_test_clifford_points = 5
# num_test_uniform_points = 10
# max_grads = 30
# max_obs = 30
#
# exp_exact = ExactMinExperiment.load('new_test')
# exp_exact.run(
#     qubits,
#     layers,
#     num_samples_per_circuit,
#     num_test_clifford_points,
#     num_test_uniform_points,
#     max_grads,
#     max_obs,
#     seed=42)