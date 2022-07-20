import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class HMC(object):
    def __init__(self, negative_log_prob):
        self.neg_log_prob = negative_log_prob
        
    def updater(self, q, p):
        raise NotImplementedError("Please implement")

    def sample(self, num_samples, initial_state):
        samples = [initial_state]
        momentum = tfp.distributions.Normal(loc=[0]*initial_state.shape[0], scale=[1]*initial_state.shape[0])
        for p0 in momentum.sample([num_samples]).numpy():
            q_new, p_new = self.updater(samples[-1], p0)

            start_log_p = self.neg_log_prob(samples[-1]) - np.sum(momentum.prob(p0).numpy())
            new_log_p = self.neg_log_prob(q_new) - np.sum(momentum.prob(p_new).numpy())
            if np.log(np.random.rand()) < start_log_p - new_log_p:
                samples.append(q_new)
            else:
                samples.append(np.copy(samples[-1]))
        
        return np.array(samples[1:])
