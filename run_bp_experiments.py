from experiments import BPExperiment

qubits = list(range(2, 10+1, 2))
layers = list(range(10, 50+1, 10))
exp_bp = BPExperiment('main')

exp_bp.run(
    qubits,
    layers,
    num_paulis=50,
    num_samples=50,
    seed=42)