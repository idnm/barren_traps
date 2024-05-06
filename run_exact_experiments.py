from experiments import ExactMinExperiment

qubits = list(range(2, 10+1, 2))
layers = (50, ) * len(qubits)


exp_exact = ExactMinExperiment('main')
exp_exact.run(
    qubits,
    layers,
    num_samples_per_circuit=10,
    num_test_uniform_points=10,
    seed=42)