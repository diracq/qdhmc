from autograd import grad
import autograd.numpy as np
import scipy.stats as st

# Borrowing for test
def hamiltonian_monte_carlo(n_samples, negative_log_prob, initial_position, path_len=1, step_size=0.5, support=None):
    """Run Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    # autograd magic
    dVdq = grad(negative_log_prob)

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    for p0 in momentum.rvs(size=size):
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1]) - np.sum(momentum.logpdf(p0))
        new_log_p = negative_log_prob(q_new) - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p and in_support(q_new, support):
            samples.append(q_new)
        else:
            samples.append(np.copy(samples[-1]))

    return np.array(samples[1:])

# Borrowing for test
def leapfrog(q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)

    p -= step_size * dVdq(q) / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * p  # whole step
        p -= step_size * dVdq(q)  # whole step
    q += step_size * p  # whole step
    p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, -p

def in_support(x, support):
    if support == None:
        return True
    for idx, support_var in enumerate(support):
        if not (support_var[0] < x[idx] < support_var[1]):
            return False
    return True

def E_norm(x, mu, Sigma):
    # Define this as the energy of the state
    return 1/2 * np.matmul(np.matmul((x - mu), np.linalg.inv(Sigma)), (np.transpose(x) - np.transpose(mu)))

mu_1 = np.array([1.0, 2.0])
Sigma_1 = np.array([[0.6, 0.2], [0.2, 0.6]])

def neg_log_prob_norm(x):
    return E_norm(x, mu_1, Sigma_1) - 1/2 * np.log(np.linalg.det(2*np.pi*Sigma_1))

lambda_1 = 1

def neg_log_prob_exp(x):
    if x >= 0.0:
        return (lambda_1*x - np.log(lambda_1))
    return -np.inf

k=10000
#samples = hamiltonian_monte_carlo(k, neg_log_prob, np.array([0.0]))
samples = hamiltonian_monte_carlo(k, neg_log_prob_exp, np.array([2.0]), support=[[0.0,np.inf]])

import matplotlib.pyplot as plt

print(samples)

plt.hist([p[0] for p in samples],50)#,[p[1] for p in samples])
#plt.scatter([p[0] for p in samples],[p[1] for p in samples])
#plt.plot([p[0] for p in samples],[p[1] for p in samples])
plt.show()
plt.savefig("matplotlib.png")
