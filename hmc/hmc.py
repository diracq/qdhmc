import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class HMC(object):
    def __init__(self, negative_log_prob):
        self.neg_log_prob = negative_log_prob
        
    def proposer(self, q, p):
        raise NotImplementedError("Please define a method to perform HMC updates.")

    def sample(self, num_samples, initial_state):
        """
        Once the HMC object is defined, sample from it.
        Args:
            num_samples (int): Number of samples to take.
            initial_state (list(float)): Location to start sampling. Vector whose size determines size of samples.
        Returns:
            samples (list(list(float))): A list of vectors samples from target distribution.
        """
        samples = [initial_state]

        # Generate gaussian momentum kicks
        momentum = tfp.distributions.Normal(loc=[0]*initial_state.shape[0], scale=[1]*initial_state.shape[0])

        count = 1
        for p0 in momentum.sample([num_samples]).numpy():

            # For debugging
            print("Sample " + str(count))

            # Proposal stage
            q_new, p_new = self.proposer(samples[-1], p0)

            # Metropolis-Hastings acceptance criteria
            start_log_p = self.neg_log_prob(samples[-1]) - np.sum(momentum.prob(p0).numpy())
            new_log_p = self.neg_log_prob(q_new) - np.sum(momentum.prob(p_new).numpy())
            if np.log(np.random.rand()) < start_log_p - new_log_p:
                samples.append(q_new)
            else:
                samples.append(np.copy(samples[-1]))

            count += 1
        
        return np.array(samples[1:])
