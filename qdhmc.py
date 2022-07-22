import sys
sys.path.append("../cv-tfq")

import numpy as np

import cirq
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

from hmc.hmc import HMC

from cv_ops import PositionOp, MomentumOp
from cv_subroutines import centeredQFT
from tests.util.cvutil import domain_bin
from cv_utils import domain_float


class QDHMC(HMC):
    def __init__(self, negative_log_prob, precision, t, r):
        super().__init__(negative_log_prob)
        self.precision = precision
        self.t = t
        self.r = r

    def proposer(self, q, p):
        qubits_all = [[cirq.GridQubit(i, j) for j in range(self.precision)] for i in range(len(q))]
        qdhmc_circuit = self.generate_circuit(qubits_all)
        input_circ = cirq.Circuit()
        for qubits, q_i in zip(qubits_all, q):
            for qubit, bit in zip(qubits, domain_bin(q_i, self.precision)):
                if bit=='1':
                    input_circ += cirq.X(qubit)
        
        add = tfq.layers.AddCircuit()
        input_circ = tfq.convert_to_tensor([input_circ])
        output_circuit = add(input_circ, append = qdhmc_circuit)
        sample_layer = tfq.layers.Sample()
        output = sample_layer(output_circuit, repetitions=1)
        bitstrings = [["".join([str(bit) for bit in bitlist[self.precision*var_idx:self.precision*(var_idx+1)]]) for var_idx in range(len(q))] for bitlist in output.numpy()[0]][0]
        q_new = [domain_float(bitstring) for bitstring in bitstrings]
        return q_new, p

    def generate_circuit(self, qubits_all):
        eta_mu, lam_mu = 0, 0
        eta_sig, lam_sig = 1, 1
        hyperparams = tfp.distributions.Normal(loc=[eta_mu, lam_mu], scale=[eta_sig, lam_sig]).sample([1]).numpy()[0]
        eta, lam = hyperparams[0], hyperparams[1]
        delta = self.t / self.r
        circuit = cirq.Circuit()

        for _ in range(self.r):
            circuit += tfq.util.exponential(operators = [self.neg_log_prob([PositionOp(qubits) for qubits in qubits_all])], coefficients = [eta*delta])
            circuit += [centeredQFT(qubits) for qubits in qubits_all]
            circuit += [tfq.util.exponential(operators = [(1/2) * MomentumOp(qubits).op ** 2], coefficients = [lam*delta]) for qubits in qubits_all]
            circuit += [centeredQFT(qubits, inverse=True) for qubits in qubits_all]
        circuit += [centeredQFT(qubits) for qubits in qubits_all]
        circuit += [cirq.X(qubits[0]) for qubits in qubits_all]
        circuit += [centeredQFT(qubits, inverse=True) for qubits in qubits_all]
        
        return circuit





