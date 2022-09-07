# QD-HMC

This repository contains the code and plotting functions to recreate the Quantum Dynamical Hamiltonian Monte Carlo (QD-HMC) algorithm. It is built on [TensorFlow-Quantum](https://www.tensorflow.org/quantum) (TFQ), [TensorFlow-Probability](https://www.tensorflow.org/probability) (TFP), and [Continuous-Variable TensorFlow-Quantum](https://github.com/QuantumVerd/cv-tfq) (CV-TFQ). 


## Set Up

To begin using the code, simply clone the repository and ensure you have a virtual environment set up with all the requirements. 

## Usage

QD-HMC is heavily integrated into TFP, and the usage is similar to the built in HMC. Just specify a few required hyperparameters and you are good use it just as you would classical HMC. Note that the `run_chain` function that is used when `.run_hmc()` is called is decorated with `@tf.function`, which gives substantial speed benefits. 

```python
test_q = HMC(log_prob, kernel_type='quantum', precision=precision, num_vars=n)
samples, sample_mean, sample_stddev, is_accepted, results = test_q.run_hmc(1000, 100)
```

There are a few limitations of note. Because this is built on CV-TFQ, there are limitations to the CV operators. Specifically, they are not ammenable to the usual numpy/TF broadcasting and vectorization. This requires all functions to be explicitly indexed, e.g. `lambda x : -x - x**2` will not work but `np.sum([-1 * x[i] for i in range(n)] + [-1 * x[i]**2 for i in range(n)])` will. These ops also have a few other quirks worth noting, `-op` is not allowed and you must used `-1 * op`. The only mathematical operations supported are the basic addition, subtraction, multiplication, and exponentiation (by positive integer). More complex functions must be approximated using these function. 

## Citation

If you use or build upon QD-HMC in your work, please cite our technical [paper]():

```bibtex
@article{qdhmc2022,
  TBD
}
