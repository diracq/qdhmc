import sys
sys.path.append("../cv-tfq")

import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow_probability as tfp
import cirq
import sympy
import collections
import random

from cv_ops import PositionOp, MomentumOp
from cv_subroutines import centeredQFT
from cv_utils import domain_float_tf, domain_bin_tf

RWResult = collections.namedtuple("RWResult", ['target_log_prob', 'log_acceptance_correction'])

class QDHMCKernel(tfp.python.mcmc.kernel.TransitionKernel):

    def __init__(self, target_log_prob, precision, t, r, num_vars):
        self._parameters = dict(
            target_log_prob_fn = target_log_prob,
            precision = precision,
            t = t,
            r = r,
            num_vars = num_vars
        )
        self.sample = tfq.layers.Sample()
        self.qubits = [[cirq.GridQubit(i, j) for j in range(self.precision)] for i in range(num_vars)]
        eta = sympy.symbols('eta')
        lam = sympy.symbols('lam')
        xs = sympy.symbols('xs0:%d'%(num_vars*precision))
        self.trotterized_circuit = self.generate_circuit(self.r, xs, eta, lam)
        self.params = list(xs) + [eta] + [lam]
        self.eta_mu = 0
        self.lam_mu = 0
        self.eta_sig = 1
        self.lam_sig = 1

    @property
    def target_log_prob_fn(self):
        return self._parameters['target_log_prob_fn']

    @property
    def precision(self):
        return self._parameters['precision']

    @property
    def t(self):
        return self._parameters['t']

    @property
    def r(self):
        return self._parameters['r']

    @property
    def num_vars(self):
        return self._parameters['num_vars']

    @property
    def is_calibrated(self):
        return False

    def generate_circuit(self, r, xs, eta, lam):
        circuit = cirq.Circuit()
        for i, qubits in enumerate(self.qubits):
            for j, qubit in enumerate(qubits):
                circuit += cirq.X(qubit)**xs[i * self.precision + j]
        for _ in range(r):
            circuit += tfq.util.exponential(operators = [self.target_log_prob_fn([PositionOp(qubits) for qubits in self.qubits]).op], coefficients = [eta])
            circuit += [centeredQFT(qubits) for qubits in self.qubits]
            circuit += [tfq.util.exponential(operators = [(1/2) * MomentumOp(qubits).op ** 2], coefficients = [lam]) for qubits in self.qubits]
            circuit += [centeredQFT(qubits, inverse=True) for qubits in self.qubits]
        circuit += [centeredQFT(qubits) for qubits in self.qubits]
        circuit += [cirq.X(qubits[0]) for qubits in self.qubits]
        circuit += [centeredQFT(qubits, inverse=True) for qubits in self.qubits]

        return circuit

    def one_step(self, current_state, previous_kernel_results, seed=None):
        eta = tf.random.normal(shape=[1], mean=self.eta_mu, stddev=self.eta_sig, seed=seed)
        eta *= self.t / self.r
        lam = tf.random.normal(shape=[1], mean=self.lam_mu, stddev=self.lam_sig, seed=seed)
        lam *= self.t / self.r
        
        bin_vals = domain_bin_tf(current_state, precision=self.precision)
        bin_vals = tf.reshape(bin_vals, [bin_vals.shape[0] * bin_vals.shape[1]])
        bin_vals = tf.cast(bin_vals, tf.float32)

        values = tf.concat([bin_vals, eta, lam], axis=0)
        circuit_output = self.sample(self.trotterized_circuit, \
            symbol_names=self.params, symbol_values=[values], repetitions=1).to_tensor()[0][0]
        bitstrings = tf.convert_to_tensor([circuit_output[i * self.precision : i * self.precision + self.precision] for i in range(self.num_vars)], dtype=tf.float32)

        next_state_list = domain_float_tf(bitstrings, self.precision)
        next_state = tf.reshape(next_state_list, current_state.shape)
        next_state = tf.cast(next_state, dtype=tf.float32)

        next_target_log_prob = self.target_log_prob_fn(next_state)
        
        new_kernel_results = previous_kernel_results._replace(
            target_log_prob = next_target_log_prob)

        return next_state, new_kernel_results

    def bootstrap_results(self, init_state):
        kernel_results = RWResult(
            target_log_prob = self.target_log_prob_fn(init_state),
            log_acceptance_correction = tf.convert_to_tensor(0.0, dtype=tf.float32)
        )
        return kernel_results

    
class HMC(object):

    def __init__(self, target_log_prob, num_vars, precision, kernel_type="classical", step_size=1.0, steps=3, t=None, r=None):
        self.precision = precision
        self.num_vars = num_vars
        if kernel_type != "classical":
            self.kernel = tfp.mcmc.MetropolisHastings(QDHMCKernel(target_log_prob, precision, t, r, num_vars))
        else:
            self.kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_log_prob, num_leapfrog_steps=steps, step_size=step_size)

    def run_hmc(self, num_samples, num_burnin, init_state=None):
        if init_state is None:
            init_state = tf.random.uniform(shape=[self.num_vars], minval=-2**(self.precision - 1), maxval=2**(self.precision - 1))
            
        @tf.function
        def run_chain():
            samples, (is_accepted, results) = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin,
                current_state=init_state,
                kernel=self.kernel,
                trace_fn=lambda _, pkr: (pkr.is_accepted, pkr.accepted_results))

            sample_mean = tf.reduce_mean(samples)
            sample_stddev = tf.math.reduce_std(samples)
            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            return samples, sample_mean, sample_stddev, is_accepted, results

        return run_chain()
