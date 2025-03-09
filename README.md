# Enhancing -SNE Visualisations With Adaptive Degrees of Freedom Parameter

## Motivation
T-distributed Stochastic Neighbor Embedding (t-SNE) is a widely used tool for dimensionality reduction and visualization of high-dimensional datasets. By replacing the Gaussian kernel in SNE with a Cauchy kernel (Student t-distribution with the degree-of-freedom (dof) parameter alpha set to 1), it alleviates the “crowding problem” in low-dimensional embeddings. Varying the degree-of-freedom parameter alpha affects t-SNE embeddings on both toy and real-world datasets (e.g., MNIST and single-cell RNA sequencing). Moreover, alpha can be regarded as a trainable parameter, allowing it to be adjusted during embedding optimization via a gradient-based method. Alpha values different from 1 can yield superior embeddings, reflected by reduced Kullback-Leibler (KL) divergence and higher k-Nearest Neighbors (kNN) recall scores, at least on some datasets. Overall, this suggests that alpha optimization can lead to more faithful low-dimensional representations of high-dimensional data.
For full related report please refer to `docs/T-SNE Visualisations With Adaptive Degrees of Freedom Parameter.pdf`

## Contents
This repository containes:
* all source code from the original openTSNE library: https://github.com/pavlin-policar/openTSNE
with modifications, aimed to allow for dof-optimization. The resulting implementation is here called `OpTSNE`
* Performance.ipynb jupyter notebook, where `OpTSNE` perormance is evaluated on several toy and real dataset and compared to the original `openTSNE` library.
* SwissRoll.ipynb jupyter notebook where dof-optimization is combined with different values of the `perplexity` parameter to achieve superior results and compared to the original `openTSNE` library.
* Utilitary modules `utils.py` and `tsne_api.py`
* separate `_tsne.pyx` file from the original `openTSNE` library to make the comparison more easy
* `docs` folder with detailed report on dof-optimization

## Modifications

### Package Name
- Changed to OpTSNE to not confuse it with original openTSNE when cross-testing

### `setup.py`
- Changes in attempt to make it compile seamlessly on an Intel-based MacOS

### `_tsne.pyx`
Main changes with regards to dof optimization:
- `estimate_positive_gradient_nn` function:
    - added computations for the alpha gradient positive term
- `_estimate_negative_gradient_single` function
    - added computations for the alpha gradient negative term
- `estimate_negative_gradient_bh` function
    - added normalization of alpha gradient negative term on `sum_Q`

### `tsne.py`
- added dataclass `OptimizationStats` to track changes in KL-divergence, dof values, alpha gradient values and embeddings with every iteration
- `kl_divergence_bh` function
    - added dof gradient computations (`alpha_grad`), based on the ouputs of _tsne module
- `gradient_descent` class
    - added optional `optimize_for_alpha` bool argument to trigger dof-optimization
    - added optional `dof_lr` argument 
    - added optional `dof` update with current value of `alpha_grad` and `dof_lr` learning rate
    - `eval_error_every_iter` argument to make tracking of KL-divergence more flexible


## Performance Evaluation
Performance review on toy and real datasets is performed in the `Performance.ipynb`. Results are compared to the those obtained with the original openTSNE implementation.

## Installation
In order to install the `OpTSNE` implementation, navigate to the `OpTSNE-source-code` folder in terminal / command prompt and run
``` bash
pip install .
```
**Note.**
On Intel-based MacOS one might need to install `gcc`, `libomp` and `fftw` libraris via
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

On Windows system the installation is straightforward. Only requirement is `Visual Studio` and `Visual C++`

Other dependancies are listed in `requirements.txt` and can be installed via:
```bash
pip install -r requirements.txt
```