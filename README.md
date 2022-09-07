# QD-HMC

This repository contains the code and plotting functions to recreate the Quantum Dynamical Hamiltonian Monte Carlo (QD-HMC) algorithm. It is built on [TensorFlow-Quantum](https://www.tensorflow.org/quantum) (TFQ) and [TensorFlow-Probability](https://www.tensorflow.org/probability) (TFP). 


## Set Up

To begin using the code, simply clone the repository and ensure you have a virtual environment set up with all the requirements. 

## Usage

QD-HMC is heavily integrated into TFP, and the usage is similar to the built in HMC. Just specify a few required hyperparameters and you are good use it just as you would classical HMC. 

```python
test_q = HMC(log_prob, kernel_type='quantum', precision=precision, num_vars=n)
```

## Citation
