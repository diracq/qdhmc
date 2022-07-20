import numpy as np
import matplotlib.pyplot as plt

from hmc_classical import HMC_classical

mu = 5.0
Sigma = 1.0

def E_norm(x, mu, Sigma):
    # Define this as the energy of the state
    return 1/2 * (x - mu)**2 / Sigma**2

def neg_log_prob_norm(x):
    return E_norm(x, mu, Sigma) + 1/2*np.log(2*np.pi*Sigma**2)

k=10000

hmc_obj = HMC_classical(neg_log_prob_norm)
samples = hmc_obj.sample(num_samples=k, initial_state=np.array([3.4]))
print(samples)

plt.hist([p[0] for p in samples], 50)
plt.show()
plt.savefig("HMC_classical_test.png")
