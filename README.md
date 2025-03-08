# Enhancing -SNE Visualisations With Adaptive Degrees of Freedom Parameter

## Motivation
T-distributed Stochastic Neighbor Embedding (t-SNE) is a widely used tool for dimensionality reduction and visualization of high-dimensional datasets. By replacing the Gaussian kernel in SNE with a Cauchy kernel (Student t-distribution with the degree-of-freedom parameter alpha set to 1), it alleviates the “crowding problem” in low-dimensional embeddings. Varying the degree-of-freedom parameter alpha affects t-SNE embeddings on both toy and real-world datasets (e.g., MNIST and single-cell RNA sequencing).
Thus alpha can be regarded as a trainable parameter, allowing it to be adjusted during embedding optimization via a gradient-based method. alpha values different from 1 can yield superior embeddings, reflected by reduced Kullback-Leibler (KL) divergence and higher k-Nearest Neighbors (kNN) recall scores, on some datasets. Overall, these results suggest that alpha optimization can lead to more faithful low-dimensional representations of high-dimensional data.

## Contents
This repository containes all source files from the original openTSNE library: 
https://github.com/pavlin-policar/openTSNE

with modifications, aimed to allow dof (alpha) optimization.

## Modifications

### package name
- I changed it to OpTSNE to not confuse it with original openTSNE

### setup.py
- I tried to change it to make it compile seamlessly on my system (Intel-based MacOS)

### _tsne.pyx
Main changes with regards to dof optimization:
- `estimate_positive_gradient_nn` function:
    - added computations for the alpha gradient positive term
- `_estimate_negative_gradient_single` function
    - added computations for the alpha gradient negative term
- `estimate_negative_gradient_bh` function
    - added normalization of alpha gradient negative term on `sum_Q`
- some print statements for debugging

### tsne.py
- added dataclass `OptimizationStats` to track changes in KL-divergence, dof values, alpha gradient values and embeddings with every iteration
- `kl_divergence_bh` function
    - added dof gradient computations (`alpha_grad`), based on the ouputs of _tsne module
- `gradient_descent` class
    - added optional `dof` update with current value of `alpha_grad` and a fixed learning rate of `0.5`


## Performance evaluation
Performance of this particular implementation is showncased in the `Performance.ipynb` and compared to the result obtained with the original openTSNE implementation.

## Installation
Navigate to the `OpTSNE-source-code` folder in terminal / command prompt and run
``` bash
pip install .
```
**Note.**
On my system (Intel-based MacOS) I had to install `gcc`, `libomp` and `fftw` libraris via
``` bash
brew install libomp
brew install fftw
brew install gcc@14
```
and export specific flags:
```bash
export CPPFLAGS="-I/usr/local/opt/libomp/include $CPPFLAGS" 
export LDFLAGS="-L/usr/local/opt/libomp/lib $LDFLAGS"
export ARCHFLAGS="-arch x86_64"
export CPPFLAGS="-I/usr/local/opt/fftw/include" 
export LDFLAGS="-L/usr/local/opt/fftw/lib"
export CC=gcc-14
export CXX=g++-14
export LDFLAGS="-L/usr/local/opt/libomp/lib"
export CPPFLAGS="-I/usr/local/opt/libomp/include"
```