import pennylane as qml
import jax
import jax.numpy as jnp
import catalyst

num_qubits = 10
dev = qml.device('lightning.qubit', wires=num_qubits)
@qml.qnode(dev)
def circuit(x):

    # for i in range(len(x)):
    #     w = i % num_qubits
    #     qml.RX(x[i], wires=w)

    def loop_fn(i, w):
        qml.RX(x[i], wires=w)
        return i % num_qubits

    catalyst.for_loop(0, len(x), 1)(loop_fn)(0)
    return qml.state()


observables = [qml.PauliX(0), qml.PauliZ(0)] * 100
@qml.qjit
def expectation(x):
    state = circuit(x).reshape([2]*num_qubits)
    return [qml.devices.qubit.measure(qml.expval(obs), state) for obs in observables]

    # return [qml.devices.qubit.measure(qml.expval(obs), state) for obs in observables]

x = jnp.linspace(0, 1, num_qubits)
print(qml.draw(circuit)(x))
print(expectation(x))

# def expectation(x):
#     state = circuit(x)
#     observables =[qml.PauliX(0), qml.PauliZ(0)] * 100
#     return jnp.array([qml.devices.qubit.measure(qml.expval(obs), state) for obs in observables])
#
# x = jnp.linspace(0, 1, 1000)
# print(jax.grad(jax.vmap(expectation))(jnp.array([x, 2*x]))).sum()))

# dev = qml.device('lightning.qubit', wires=1)
# @qml.qnode(dev, interface='jax')
# def circuit(x):
#     for xi in x:
#         qml.RX(xi, wires=0)
#     return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0))
#
# expectation = jax.jit(circuit)
# x = jnp.linspace(0, 1, 500)
# print(expectation(x))
