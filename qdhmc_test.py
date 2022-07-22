import sys
sys.path.append("../cv-tfq")

import numpy as np
import matplotlib.pyplot as plt

from qdhmc import QDHMC

mu = 0.0
Sigma = 1.0

from cv_ops import PositionOp, MomentumOp

def E_norm(x, mu, Sigma):
    # Define this as the energy of the state
    for idx, x_i in enumerate(x):
        if isinstance(x_i, PositionOp) or isinstance(x_i, MomentumOp):
            x[idx] = x_i.op
    return (1/(2*Sigma**2)) * x[0]**2 - (mu/Sigma**2) * x[0] + (mu**2/(2*Sigma**2)) * x[0]**0

def neg_log_prob_norm(x):
    return E_norm(x, mu, Sigma) + 1/2*np.log(2*np.pi*Sigma**2) * x[0]**0

k=200

hmc_obj = QDHMC(neg_log_prob_norm, precision=10, t=2, r=5)
samples = hmc_obj.sample(num_samples=k, initial_state=np.array([0.5]))
print(samples)

plt.hist([p[0] for p in samples], 16)
plt.show()
plt.savefig("QDHMC_test.png")