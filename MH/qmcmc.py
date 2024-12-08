"""File containing the Quantum enhanced MCMC for ising."""
import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow_probability as tfp
import cirq
import sympy
import collections

RWResult = collections.namedtuple("RWResult", "target_log_prob")


class QuantumMCMCIsingKernel(tfp.python.mcmc.kernel.TransitionKernel):
    """
    Transition kernel that integrates quantum enhanced MCMC for MH algorithms.

    Based on https://arxiv.org/pdf/2203.12497.pdf. Currently only supports 1D ising models.

    Inputs:
        - Size (int): length of the ising chain
        - Js (array-like): list of J values for the ising model [j1-j2, j2-3, ... jn-1]
        - Hs (array-like): list of H values for the ising model [h1, h2, h3, ... hn]
        - r (int): repetitions of the trotterized circuit
        - temp (float): temperature of the ising model
    """

    def __init__(self, size, js, hs, r, temp):
        """
        Initialize member variables.

        Args:
            - size (int): number of nodes in the ising model
            - js (list): J parameters for interacting terms
            - hs (list): h parameters for background terms
            - r (int): trotter repetitions
            - temp (float): temperature of the ising model

        Returns:
            - None
        """
        js = tf.cast(js, dtype=tf.float32)
        hs = tf.cast(hs, dtype=tf.float32)
        temp = tf.cast(temp, dtype=tf.float32)
        self._parameters = dict(
            target_log_prob_fn=lambda x : -1 * self.ising_model_energy_1d(x),
            size_q=size,
            js=js,
            hs=hs,
            rep=r,
            temperature=temp,
        )
        self.sample = tfq.layers.Sample()
        self.qubits = [cirq.GridQubit(0, i) for i in range(size)]
        self.alpha = tf.math.sqrt(tf.cast(size, dtype=tf.float32)) / tf.math.sqrt(
            tf.math.reduce_sum(tf.math.pow(js, 2))
            + tf.math.reduce_sum(tf.math.pow(hs, 2))
        )
        a = sympy.symbols("a")
        b = sympy.symbols("b0:%d" % size)
        theta = sympy.symbols("theta0:%d" % size)
        xs = sympy.symbols("xs0:%d" % size)
        self.trotterized_circuit = self.make_circuit(self.rep, xs, a, b, theta)
        self.params = list(xs) + [a] + list(b) + list(theta)

    def ising_model_energy_1d(self, spins):
        """
        Compute the energy for a 1D ising model.

        Args:
            - spins (1D tensor): the (-1, 1) values of the spins

        Returns:
            - (float): the energy of the ising model
        """
        if len(spins.shape) == 1:
            spins = tf.expand_dims(spins, axis=1)
        spins = tf.cast(spins, dtype=tf.float32)
        interaction_e = tf.math.reduce_sum(
            self.js * tf.roll(spins, -1, axis=1) * spins, axis=1
        )
        background_e = tf.math.reduce_sum(self.hs * spins, axis=1)
        return (-interaction_e - background_e) / self.temperature

    # CURRENTLY ONLY SUPPORTS 1D
    def ising_model_energy_2d(self, spins):
        spins = tf.cast(spins, dtype=tf.float32)
        # Since js is upper triangular matrix, where js[i][j] = J_{i, j}
        interaction_e = tf.math.reduce_sum(
            tf.multiply(tf.multiply(self.js, spins), tf.transpose(spins))
        )
        background_e = tf.math.reduce_sum(self.hs * spins, axis=1)
        return (-interaction_e - background_e) / self.temperature

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def size_q(self):
        return self._parameters["size_q"]

    @property
    def js(self):
        return self._parameters["js"]

    @property
    def hs(self):
        return self._parameters["hs"]

    @property
    def rep(self):
        return self._parameters["rep"]

    @property
    def temperature(self):
        return self._parameters["temperature"]

    @property
    def is_calibrated(self):
        return False

    def make_circuit(self, r, xs, a, b, theta):
        """
        Generate the circuit, with layers of RX and RZ gates, followed by RZZ gates.

        Args:
            - r (int): number of repetitions
            - xs (list): parameters to initialize bitstrings
            - a (list): the a parameters
            - b (list): the b parameters
            - theta (list): the theta parameters

        Returns:
            - (cirq.Circuit): trotterized circuit
        """
        circuit = cirq.Circuit()
        for i, q in enumerate(self.qubits):
            circuit += cirq.X(q) ** xs[i]
        for i in range(r):
            for j, q in enumerate(self.qubits):
                circuit += cirq.rx(2 * a).on(q)
                circuit += cirq.rz(2 * b[j]).on(q)
            for j in range(0, len(theta) - 1, 2):
                circuit += cirq.CNOT(self.qubits[j], self.qubits[j + 1])
                circuit += cirq.rz(theta[j]).on(self.qubits[j + 1])
                circuit += cirq.CNOT(self.qubits[j], self.qubits[j + 1])
            for j in range(1, len(theta) - 1, 2):
                circuit += cirq.CNOT(self.qubits[j], self.qubits[j + 1])
                circuit += cirq.rz(theta[j]).on(self.qubits[j + 1])
                circuit += cirq.CNOT(self.qubits[j], self.qubits[j + 1])
        for j, q in enumerate(self.qubits):
            circuit += cirq.rx(2 * a).on(q)
            circuit += cirq.rz(2 * b[j]).on(q)

        return circuit

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """
        Generate the next proposal.

        Randomly selects the a and b parameters and parameterizes the trotterized
        circuit with them. Converts the current state and encodes it to the circuit. The
        circuit is then simulated and the output is converted back into a float and returned.

        The unusual formatting and usage of certain functions is to ensure it is compatible with
        all TF graphing techniques.

        Args:
            - current_state (float): the current parameters of the distribution
            - previous_kernel_results (RWResult): tuple that contains the information from the
                previous iteration
            - seed (int, optional): set the random seed (note that it is not used in the sampling
                from the circuit, so it will not ensure replicatability)

        Returns:
            - (RWResult): the next proposal
        """
        gamma = tf.random.uniform(shape=[], minval=0.25, maxval=0.6, seed=seed)
        t = tf.random.uniform(shape=[], minval=2, maxval=20, seed=seed)
        dt = t / self.rep
        a = gamma * dt
        b = -(1 - gamma) * self.alpha * self.hs * dt
        theta = -2 * self.js * (1 - gamma) * self.alpha * dt

        # The first of these is to encode the binary values into the circuit
        values = tf.concat([(current_state[0] - 1) / (-2), [a], b, theta], axis=0)
        # This could be updates to increase samples and take the most common one
        next_state = self.sample(
            self.trotterized_circuit,
            symbol_names=self.params,
            symbol_values=[values],
            repetitions=1,
        ).to_tensor()[0]

        next_spins = next_state * -2 + 1
        next_spins = tf.reshape(next_spins, [1, self.size_q])
        next_spins = tf.cast(next_spins, dtype=tf.float32)

        next_target_log_prob = self.target_log_prob_fn(next_spins)

        new_kernel_results = previous_kernel_results._replace(
            target_log_prob=next_target_log_prob
        )

        return next_spins, new_kernel_results

    def bootstrap_results(self, init_state):
        """
        Bootstrap an initial result from the given state.

        Args:
            - init_state (float): the current parameters of the distribution

        Returns:
            - (RWResult): the bootstrapped proposal
        """
        kernel_results = RWResult(target_log_prob=self.target_log_prob_fn(init_state))
        return kernel_results


class ClassicalMCMCIsingKernel(tfp.python.mcmc.kernel.TransitionKernel):
    """
    Transition kernel that integrates TFP with discrete ising model methods.

    Currently only supports 1D ising models.
    Currently only supports uniform random new proposals (hamming distance 1 proposals to be added).

    Inputs:
        - size (int): length of the ising chain
        - js (array-like): list of J values for the ising model [j1-j2, j2-3, ... jn-1]
        - hs (array-like): list of H values for the ising model [h1, h2, h3, ... hn]
        - r (int): repetitions of the trotterized circuit
        - temp (float): temperature of the ising model
    """

    def __init__(self, size, js, hs, r, temp):
        """
        Initialize member variables.

        Args:
            - size (int): number of nodes in the ising model
            - js (list): J parameters for interacting terms
            - hs (list): h parameters for background terms
            - temp (float): temperature of the ising model

        Returns:
            - None
        """
        js = tf.cast(js, dtype=tf.float32)
        hs = tf.cast(hs, dtype=tf.float32)
        temp = tf.cast(temp, dtype=tf.float32)
        self._parameters = dict(
            target_log_prob_fn=lambda x : -1 * self.ising_model_energy_1d(x),
            size_q=size,
            js=js,
            hs=hs,
            temperature=temp,
        )

    def ising_model_energy_1d(self, spins):
        """
        Compute the energy for a 1D ising model.

        Args:
            - spins (1D tensor): the (-1, 1) values of the spins

        Returns:
            - (float): the energy of the ising model
        """
        if len(spins.shape) == 1:
            spins = tf.expand_dims(spins, axis=1)
        spins = tf.cast(spins, dtype=tf.float32)
        interaction_e = tf.math.reduce_sum(
            self.js * tf.roll(spins, -1, axis=1) * spins, axis=1
        )
        background_e = tf.math.reduce_sum(self.hs * spins, axis=1)
        return (-interaction_e - background_e) / self.temperature

    # CURRENTLY ONLY SUPPORTS 1D
    def ising_model_energy_2d(self, spins):
        spins = tf.cast(spins, dtype=tf.float32)
        # Since js is upper triangular matrix, where js[i][j] = J_{i, j}
        interaction_e = tf.math.reduce_sum(
            tf.multiply(tf.multiply(self.js, spins), tf.transpose(spins))
        )
        background_e = tf.math.reduce_sum(self.hs * spins, axis=1)
        return (-interaction_e - background_e) / self.temperature

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def size_q(self):
        return self._parameters["size_q"]

    @property
    def js(self):
        return self._parameters["js"]

    @property
    def hs(self):
        return self._parameters["hs"]

    @property
    def temperature(self):
        return self._parameters["temperature"]

    @property
    def is_calibrated(self):
        return False

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """
        Generate the next proposal.

        Randomly selects the Q(y|x) uniformly.

        Args:
            - current_state (float): the current parameters of the distribution
            - previous_kernel_results (RWResult): tuple that contains the information from the
                previous iteration
            - seed (int, optional): set the random seed (note that it is not used in the sampling
                from the circuit, so it will not ensure replicatability)

        Returns:
            - (RWResult): the next proposal
        """
        next_spins = tf.cast(
            tf.random.uniform(
                shape=[1, self.size_q], minval=0, maxval=2, dtype=tf.int32
            )
            * 2
            - 1,
            tf.float32,
        )

        next_target_log_prob = self.target_log_prob_fn(next_spins)

        next_target_log_prob = tf.cast(next_target_log_prob, dtype=tf.float32)

        new_kernel_results = previous_kernel_results._replace(
            target_log_prob=next_target_log_prob
        )

        next_spins = tf.cast(next_spins, dtype=tf.float32)
        return next_spins, new_kernel_results

    def bootstrap_results(self, init_state):
        """
        Bootstrap an initial result from the given state.

        Args:
            - init_state (float): the current parameters of the distribution

        Returns:
            - (RWResult): the bootstrapped proposal
        """
        kernel_results = RWResult(target_log_prob=self.target_log_prob_fn(init_state))
        return kernel_results


class IsingMH(object):
    """
    Integrates MH with discrete ising model transition kernel.

    Inputs:
        - size (int): length of the ising chain
        - js (array-like): list of J values for the ising model [j1-j2, j2-3, ... jn-1]
        - hs (array-like): list of H values for the ising model [h1, h2, h3, ... hn]
        - r (int): repetitions of the trotterized circuit
        - temp (float): temperature of the ising model
        - kernel type (string): whether to use the quantum enhanced MCMC kernel or the classical one
    """

    def __init__(self, size, js, hs, r, temp, kernel_type):
        """
        Initialize member variables.

        Args:
            - size (int): the number of spins in the ising model
            - js (array-like): list of J values for the ising model [j1-j2, j2-3, ... jn-1]
            - hs (array-like): list of H values for the ising model [h1, h2, h3, ... hn]
            - r (int): the number of trotter repetitions to do
            - temp (float): the trotterized time to simualte
            - kernel_type (str): whether to use the classical or quantum kernel

        Returns:
            - None
        """
        self.n = size
        if kernel_type == "quantum":
            self.kernel = tfp.mcmc.MetropolisHastings(
                QuantumMCMCIsingKernel(size, js, hs, r, temp)
            )
        else:
            self.kernel = tfp.mcmc.MetropolisHastings(
                ClassicalMCMCIsingKernel(size, js, hs, r, temp)
            )

    def run_mcmc(self, num_results, num_burnin, init_state=None):
        """
        Run the MCMC optimization.

        Args:
            - num_samples (int): number of optimization steps
            - num_burnin (int): number of burn in steps
            - init_state (tensor): initial state for the optimization

        Returns:
            - (tuple): a tuple containing information about the states, acceptance rates, and results
        """
        if init_state is None:
            init_state = tf.cast(
                (
                    tf.random.uniform(
                        shape=(1, self.n), minval=0, maxval=2, dtype=tf.int32
                    )
                    * 2
                )
                - 1,
                tf.float32,
            )

        @tf.function
        def run_chain():
            # Run the chain (with burn-in).
            samples, (is_accepted, results) = tfp.mcmc.sample_chain(
                num_results=num_results,
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
