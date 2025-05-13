from experiments import BPExperiment

qubits = list(range(2, 10+1, 2))
layers = list(range(5, 30+1, 5))
exp_bp = BPExperiment('bp_hea')

exp_bp.run(
    qubits,
    layers,
    num_paulis=50,
    num_samples=50,
    seed=42)