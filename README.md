# Torch Connected Components

## Compatibility

| Torch Version | CUDA 9.2 | CUDA 10.1 | CUDA 10.2 | CUDA 11.0 | CUDA 11.1 | CUDA 11.3 | CUDA 11.6 | CUDA 11.7 | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | Min. Python Version | Max. Python Version |
| ------------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------------------- | ------------------- |
| 1.7.1         | ☑️       | ☑️        | ✅        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.8.0         | ❌       | ☑️        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.8.1         | ❌       | ☑️        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.9.0         | ❌       | ❌        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.9.1         | ❌       | ❌        | ✅        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.10.0        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.10.1        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.9                 |
| 1.10.2        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.6                 | 3.10                |
| 1.11.0        | ❌       | ❌        | ✅        | ❌        | ❌        | ☑️        | ❌        | ❌        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.12.0        | ❌       | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.12.1        | ❌       | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.13.0        | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 1.13.1        | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | ❌        | 3.7                 | 3.10                |
| 2.0.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | 3.8                 | 3.11                |
| 2.0.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | ❌        | 3.8                 | 3.11                |
| 2.1.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.1.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.1.2         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.2.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.2.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.3.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.3.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ✅        | 🟦        | ❌        | 3.8                 | 3.11                |
| 2.4.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ☑️        | ✅        | 🟦        | 3.8                 | 3.12                |
| 2.4.1         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ☑️        | ✅        | 🟦        | 3.8                 | 3.12                |
| 2.5.0         | ❌       | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ❌        | ☑️        | ✅        | ☑️        | 3.9                 | 3.12                |

- ✅ : Default Stable CUDA
- ☑️ : Stable CUDA
- 🟦 : Experimental CUDA
- ❌ : Unsupported

More details : https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix

## Development

For local development, a few libraries are required. We need the NVIDIA CUDA Toolkit to build our CUDA kernels along with the Torch library (`libtorch`) for PyTorch bindings. We also need the Python headers for the bindings. The following instructions assume Linux (or WSL). This is mostly for integration with VSCode.

Please carefully read all the instructions before running the setup.

### NVIDIA CUDA Toolkit

The NVIDIA CUDA Toolkit can be downloaded from https://developer.nvidia.com/cuda-toolkit-archive. It will create a symbolic link to `/usr/local/cuda`. Versions can be switched with `update-alternatives`.

### Torch

As we're building some bindings for Torch, we also need the different header files of `libtorch`. These can be downloaded from the official PyTorch website : https://pytorch.org/get-started/locally. The library must go to the `lib` directory.

For example :

```bash
cd lib
wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.4.1%2Bcu121.zip # libtorch 2.4.1 for CUDA 12.1 (Linux)
unzip libtorch-shared-with-deps-2.4.1%2Bcu121.zip
rm libtorch-shared-with-deps-2.4.1%2Bcu121.zip
```

> [!WARNING]
> Make sure you downloaded the proper version of `libtorch` for both CUDA and your system. The `libtorch` version must match the PyTorch's one.

One other alternative is to use the headers shipped with PyTorch directly.

```bash
cd lib
ln -s $(poetry env info --path)/lib/python*/site-packages/torch libtorch
```

### Python

Because we can use different versions of Python, we need to provide the proper headers. On Linux, these are usually stored to `/usr/include/python<x>.<y>`. One easy solution is just to create another symbolic link in the `lib` directory.

```bash
cd lib
ln -s /usr/include/$(basename $(realpath $(poetry env info --executable))) python
```

## References

This work is based on the following articles.

[1] Allegretti, S., Bolelli, F., & Grana, C. (2020). [Optimized Block-Based Algorithms to Label Connected Components on GPUs](https://federicobolelli.it/pub_files/2019tpds.pdf). IEEE Transactions on Parallel and Distributed Systems, 31(2), 423–438. https://doi.org/10.1109/TPDS.2019.2934683

[2] Grana, C., Bolelli, F., Baraldi, L., & Vezzani, R. (2016). [YACCLAB-Yet Another Connected Components Labeling Benchmark](https://federicobolelli.it/pub_files/2016icpr.pdf). https://doi.org/10.1109/ICPR.2016.7900112

[3] Bolelli, F., Allegretti, S., Lumetti, L., & Grana, C. (2024). [A State-of-the-Art Review with Code about Connected Components Labeling on GPUs](https://federicobolelli.it/pub_files/2024tpds.pdf).
