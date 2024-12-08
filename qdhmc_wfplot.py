"""File containing the QDHMC integration with WF support with TFP."""
import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow_probability as tfp
import cirq
import sympy
import collections

import matplotlib.pyplot as plt

from cv_tfq.cv_ops import PositionOp, MomentumOp
from cv_tfq.cv_subroutines import centeredQFT
from cv_tfq.cv_utils import domain_float_tf, domain_bin_tf

RWResult = collections.namedtuple(
    "RWResult", ["target_log_prob", "wfs", "log_acceptance_correction"]
)


class QDHMCKernel(tfp.python.mcmc.kernel.TransitionKernel):
    """
    The core transition kernel for the QDHMC algorithm.

    Contains information and functions for proposing updating according to the QDHMC algorithm
    as outlined in the paper. Designed to be wrapped with other TFP functions.

    Attributes:
        - _parameters (dict): dictionary containing all the key hyperparameters of the algorithm
        - sample (tf.keras.layers.Layer): the TFQ layer for sampling from circuits
        - state (tf.keras.layers.Layer): the TFQ layer for getting the state from circuits
        - all_circuits (list): the list of all intermediate circuits
        - trotterized_circuit (cirq.Circuit): the static circuit parameterized to enable random trotterization.
            This is generated up front (as opposed to on the fly) to improve speed and to integrate with TF
            autographing substantially easier.
        - params (list): the symbols used in the trotterized circuit
        - eta_mu (float): the mean of the eta parameter used to control the impact of the momentum
        - eta_sig (float): the standard deviation of the eta parameter
        - lam_mu (float): the mean of the lambda parameter used to control the impact of the cost function
        - lam_sig (float): the standard deviation of the lambda parameter
    """

    def __init__(self, target_log_prob, precision, t, r, num_vars):
        """
        Initialize member variables.

        Args:
            - target_lob_prob (Callable): the function to optimize
            - precision (int): number of qubits to use to represent each variable
            - t (float): the trotterized time to simualte
            - r (int): the number of trotter repetitions to do
            - num_vars (int): the number of variables that the function is of

        Returns:
            - None
        """
        self._parameters = dict(
            target_log_prob_fn=target_log_prob,
            precision=precision,
            t=t,
            r=r,
            num_vars=num_vars,
        )
        self.sample = tfq.layers.Sample()
        self.state = tfq.layers.State()
        self.qubits = [
            [cirq.GridQubit(i, j) for j in range(self.precision)]
            for i in range(num_vars)
        ]
        eta = sympy.symbols("eta")
        lam = sympy.symbols("lam")
        xs = sympy.symbols("xs0:%d" % (num_vars * precision))
        self.all_circuits = self.generate_circuits(self.r, xs, eta, lam)
        self.trotterized_circuit = self.all_circuits[-1]
        self.params = list(xs) + [eta] + [lam]
        self.eta_mu = 0
        self.lam_mu = 0
        self.eta_sig = 1
        self.lam_sig = 1

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def precision(self):
        return self._parameters["precision"]

    @property
    def t(self):
        return self._parameters["t"]

    @property
    def r(self):
        return self._parameters["r"]

    @property
    def num_vars(self):
        return self._parameters["num_vars"]

    @property
    def is_calibrated(self):
        return False

    def generate_circuits(self, r, xs, eta, lam):
        """
        Generate the trotterized circuit.

        Args:
            - r (int): the number of trotter repetitions
            - xs (list): the parameters to initialize a given bitstring
            - eta (list): the parameters representing eta
            - lam (list): the parameters representing lambda

        Returns:
            - (list): all intermediate trotterized circuits
        """
        circuit_list = []
        circuit = cirq.Circuit()
        for i, qubits in enumerate(self.qubits):
            for j, qubit in enumerate(qubits):
                circuit += cirq.X(qubit) ** xs[i * self.precision + j]
        circuit_list.append(circuit)
        for _ in range(r):
            circuit += tfq.util.exponential(
                operators=[
                    self.target_log_prob_fn(
                        [PositionOp(qubits) for qubits in self.qubits]
                    ).op
                ],
                coefficients=[eta],
            )
            circuit += [centeredQFT(qubits) for qubits in self.qubits]
            circuit += [
                tfq.util.exponential(
                    operators=[(1 / 2) * MomentumOp(qubits).op ** 2], coefficients=[lam]
                )
                for qubits in self.qubits
            ]
            circuit += [cirq.X(qubits[0]) for qubits in self.qubits]
            circuit += [centeredQFT(qubits, inverse=True) for qubits in self.qubits]
            circuit_list.append(circuit)
            circuit += [centeredQFT(qubits) for qubits in self.qubits]
            circuit += [cirq.X(qubits[0]) for qubits in self.qubits]
            circuit += [centeredQFT(qubits, inverse=True) for qubits in self.qubits]

        return circuit_list

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """
        Generate the next proposal.

        Randomly selects the eta and lambda parameters and parameterizes the trotterized
        circuit with them. Converts the current state and encodes it to the circuit. The
        circuit is then simulated and the output is converted back into a float and returned.

        The unusual formatting and usage of certain functions is to ensure it is compatible with
        all TF graphing techniques.

        Additionally saves the wavefunction for all intermediate trotterized circuits.

        Args:
            - current_state (float): the current parameters of the distribution
            - previous_kernel_results (RWResult): tuple that contains the information from the
                previous iteration
            - seed (int, optional): set the random seed (note that it is not used in the sampling
                from the circuit, so it will not ensure replicatability)

        Returns:
            - (RWResult): the next proposal
        """
        eta = tf.random.normal(
            shape=[1], mean=self.eta_mu, stddev=self.eta_sig, seed=seed
        )
        eta *= self.t / self.r
        lam = tf.random.normal(
            shape=[1], mean=self.lam_mu, stddev=self.lam_sig, seed=seed
        )
        lam *= self.t / self.r

        bin_vals = domain_bin_tf(current_state, precision=self.precision)
        bin_vals = tf.reshape(bin_vals, [bin_vals.shape[0] * bin_vals.shape[1]])
        bin_vals = tf.cast(bin_vals, tf.float32)

        values = tf.concat([bin_vals, eta, lam], axis=0)

        circuit_output = self.sample(
            self.trotterized_circuit,
            symbol_names=self.params,
            symbol_values=[values],
            repetitions=1,
        ).to_tensor()[0][0]
        bitstrings = tf.convert_to_tensor(
            [
                circuit_output[i * self.precision : i * self.precision + self.precision]
                for i in range(self.num_vars)
            ],
            dtype=tf.float32,
        )

        next_wfs = []
        for intermediate_circuit in self.all_circuits:
            final_state = self.state(
                intermediate_circuit, symbol_names=self.params, symbol_values=[values]
            ).to_tensor()
            wf = tf.reshape(
                tf.math.real(tf.math.conj(final_state) * final_state)[0],
                [2 ** (self.num_vars * self.precision)],
            )
            next_wfs.append(wf)
        next_wfs = tf.convert_to_tensor(next_wfs, dtype=tf.float32)

        next_state_list = domain_float_tf(bitstrings, self.precision)
        next_state = tf.reshape(next_state_list, current_state.shape)
        next_state = tf.cast(next_state, dtype=tf.float32)

        next_target_log_prob = self.target_log_prob_fn(next_state)

        new_kernel_results = previous_kernel_results._replace(
            target_log_prob=next_target_log_prob, wfs=next_wfs
        )

        return next_state, new_kernel_results

    def bootstrap_results(self, init_state):
        """
        Bootstrap an initial result from the given state.

        Args:
            - init_state (float): the current parameters of the distribution

        Returns:
            - (RWResult): the bootstrapped proposal
        """
        kernel_results = RWResult(
            target_log_prob=self.target_log_prob_fn(init_state),
            wfs=tf.convert_to_tensor(
                [[0.0] * (2 ** (self.num_vars * self.precision))] * (self.r + 1),
                dtype=tf.float32,
            ),
            log_acceptance_correction=tf.convert_to_tensor(0.0, dtype=tf.float32),
        )
        return kernel_results


class HMC(object):
    """
    The core wrapper class for the transition kernel.

    Attributes:
        - precision (int): number of qubits to represent each variable with
        - num_vars (int): number of variables in the function
        - kernel (mcmc object): type of optimization to be used
    """

    def __init__(
        self,
        target_log_prob,
        num_vars,
        precision,
        kernel_type="classical",
        step_size=1.0,
        steps=3,
        t=None,
        r=None,
    ):
        """
        Initialize member variables.

        Args:
            - target_lob_prob (Callable): the function to optimize
            - num_vars (int): the number of variables that the function is of
            - precision (int): number of qubits to use to represent each variable
            - kernel_type (str): whether to use classical or quantum HMC
            - steps_size (float): the time for the classical integrator to simulate
            - steps (int): the number of leapfrog steps for the classical integration
            - t (float): the trotterized time to simualte
            - r (int): the number of trotter repetitions to do

        Returns:
            - None
        """
        self.precision = precision
        self.num_vars = num_vars
        if kernel_type != "classical":
            self.kernel = tfp.mcmc.MetropolisHastings(
                QDHMCKernel(target_log_prob, precision, t, r, num_vars)
            )
        else:
            self.kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                num_leapfrog_steps=steps,
                step_size=step_size,
            )

    def run_hmc(self, num_samples, num_burnin, init_state=None):
        """
        Run the HMC optimization.

        Args:
            - num_samples (int): number of optimization steps
            - num_burnin (int): number of burn in steps
            - init_state (tensor): initial state for the optimization

        Returns:
            - (tuple): a tuple containing information about the states, acceptance rates, and results
        """
        if init_state is None:
            init_state = tf.random.uniform(
                shape=[self.num_vars],
                minval=-(2 ** (self.precision - 1)),
                maxval=2 ** (self.precision - 1),
            )

        @tf.function
        def run_chain():
            samples, (is_accepted, results) = tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin,
                current_state=init_state,
                kernel=self.kernel,
                trace_fn=lambda _, pkr: (pkr.is_accepted, pkr.accepted_results),
            )

            sample_mean = tf.reduce_mean(samples)
            sample_stddev = tf.math.reduce_std(samples)
            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            return samples, sample_mean, sample_stddev, is_accepted, results

        return run_chain()
