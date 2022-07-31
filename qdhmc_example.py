import sys
sys.path.append("../cv-tfq")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from hmc_tfp import HMC
from cv_utils import domain_float

mu = 0.0
Sigma = 1.0

def E_norm(x, mu, Sigma):
    # Define this as the energy of the state
    return 1/2 * (x - mu)**2 / Sigma**2

def log_prob_norm(x):
    return -1*E_norm(x[0], mu, Sigma) - 1/2*np.log(2*np.pi*Sigma**2)

precision = 8
t = 2.0
r = 5
k = 1000

hmc = HMC(target_log_prob=log_prob_norm, step_size=3.0, steps=1)
classical_samples, cmean, cstddev, cacc, cresults = hmc.run_hmc(num_samples=k, num_burnin=0, init_state=tf.cast(tf.reshape([0.3],[1,1]), tf.float64))
classical_samples = [classical_samples.numpy()[i][0] for i in range(k)]

qdhmc = HMC(target_log_prob=log_prob_norm, kernel_type="quantum", precision=precision, t=t, r=r, num_vars=1)
quantum_samples, qmean, qstddev, qacc, qresults = qdhmc.run_hmc(num_samples=k, num_burnin=0)
quantum_samples = [quantum_samples.numpy()[i][0] for i in range(k)]

bins = [domain_float(bin(i + 2**precision)[-precision:]) - (domain_float(bin(2**(precision-1) + 1 + 2**precision)[-precision:]))/2 for i in range(2**precision) if (domain_float(bin(i + 2**precision)[-precision:]) > -5 and domain_float(bin(i + 2**precision)[-precision:]) < 5)]
plt.hist([p[0] for p in quantum_samples], bins=bins)
plt.show()
plt.savefig("QDHMC_TFP_test.png")
plt.close()

plt.hist([p[0] for p in classical_samples], bins=bins)
plt.show()
plt.savefig("HMC_TFP_test.png")
plt.close()