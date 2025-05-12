from experiments import BPExperiment

# qubits = list(range(2, 12+1, 2))
# layers = list(range(5, 25+1, 5))
# exp_bp = BPExperiment('bp_hea')

# qubits = list(range(2, 12+1, 2))
# layers = list(range(20, 25+1, 5))
# exp_bp = BPExperiment.load('bp_hea')
#
#
# exp_bp.run(
#     qubits,
#     layers,
#     num_paulis=50,
#     num_samples=50,
#     seed=42)

qubits = list(range(12, 12+1, 2))
layers = list(range(5, 25+1, 5))
exp_bp = BPExperiment.load('bp_hea')


exp_bp.run(
    qubits,
    layers,
    num_paulis=50,
    num_samples=50,
    seed=42)

