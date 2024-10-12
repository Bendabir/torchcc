# Torch Connected Components

TorchCC is a library that brings efficient CUDA implementations of Connected Components related algorithms (such as Connection Components Labeling).
CPU support is delegated to external libraries such as OpenCV. It is built to support multiple CUDA and PyTorch versions.

> [!NOTE]
> This library is still an early prototype. It's definitely not suitable for production. Only Linux is supported at the moment. Windows could probably be supported with some extra efforts.

## Install

### From Package Manager

You can install TorchCC by using a custom PyPI index. There is one index per supported version of CUDA. This is still experimental though.

#### pip

```bash
pip install --index-url https://bendabir.github.io/pypi/cu121 torchcc
```

#### Poetry

```toml
[[tool.poetry.source]]
name = "bendabir-cu121"
url = "https://bendabir.github.io/pypi/cu121"
priority = "explicit"

[tool.poetry.dependencies]
torchcc = { version = "^0.1.0", source = "bendabir-cu121" }
```

### Direct Install

> [!NOTE]
> This is a temporary solution. I might also upload artifacts to PyPI, but it means that one version of CUDA needs to be selected. It's not optimal as I'd like to uncouple CUDA versions from the library version.

1. Go the release page.
2. Download a Wheel file for your version of CUDA and Python. Not all versions are supported (see [the compatibility matrix](#compatibility)).
3. Install it in your env. For example : `pip install torchcc-0.1.0+cu121-cp312-cp312-linux_x86_64.whl`

## Usage

All algorithms are implemented in a batch fashion for easy integration with PyTorch.

### 2D Connected Components Labeling

```python
labels = torchcc.ccl2d(masks, connectivity = 8) # masks : [N x H x W]
```

### 3D Connected Components Labeling

**TODO**

### CPU support

CPU support is delegated to third-party libraries. [OpenCV](https://github.com/opencv/opencv-python) is used for 2D Connected Components Labeling on CPU, while [ConnectedComponents3D](https://github.com/seung-lab/connected-components-3d) is used for 3D counterpart. Both libraries are optional. Install them for transparent CPU support.

```bash
pip install "opencv-python-headless>=4,<5" # 2D
pip install "connected-components-3d>=3,<4" # 3D
```

## Compatibility

Below is the planned compatibility matrix for TorchCC. If possible, all PyTorch version after Python 3.9 will be supported. For various compatibility reasons, only PyTorch versions supporting at least CUDA 11.0 are supported.

| CUDA Version | PyTorch Version                                                                                            | Supported |
| ------------ | ---------------------------------------------------------------------------------------------------------- | :-------: |
| `9.2`        | `1.7.1`                                                                                                    |    ❌     |
| `10.1`       | `1.7.1`, `1.8.0`, `1.8.1`                                                                                  |    ❌     |
| `10.2`       | `1.7.1`, `1.8.0`, `1.8.1`, `1.9.0`, `1.9.1`, `1.10.0`, `1.10.1`, `1.10.2`, `1.11.0`                        |    ❌     |
| `11.0`       | `1.7.1`                                                                                                    |    ❕     |
| `11.1`       | `1.8.0`, `1.8.1`, `1.9.0`, `1.9.1`                                                                         |    ❕     |
| `11.3`       | `1.10.0`, `1.10.1`, `1.10.2`, `1.11.0`, `1.12.0`, `1.12.1`                                                 |    ❕     |
| `11.6`       | `1.12.0`, `1.12.1`, `1.13.0`, `1.13.1`                                                                     |    ✔️     |
| `11.7`       | `1.13.0`, `1.13.1`, `2.0.0`, `2.0.1`                                                                       |    ✔️     |
| `11.8`       | `2.0.0`, `2.0.1`, `2.1.0`, `2.1.1`, `2.1.2`, `2.2.0`, `2.2.1`, `2.3.0`, `2.3.1`, `2.4.0`, `2.4.1`, `2.5.0` |    ✔️     |
| `12.1`       | `2.1.0`, `2.1.1`, `2.1.2`, `2.2.0`, `2.2.1`, `2.3.0`, `2.3.1`, `2.4.0`, `2.4.1`, `2.5.0`                   |    ✔️     |
| `12.4`       | `2.4.0`, `2.4.1`, `2.5.0`                                                                                  |    ✔️     |

TorchCC is not yet built for some versions of CUDA because of some build errors due to changes in `torchlib`.

<details>
  <summary><b>PyTorch Compatibility Matrix</b></summary>

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

</details>

## Development

### Libraries

For local development, a few libraries are required. We need the NVIDIA CUDA Toolkit to build our CUDA kernels along with the Torch library (`libtorch`) for PyTorch bindings. We also need the Python headers for the bindings. The following instructions assume Linux (or WSL). This is mostly for integration with VSCode.

Please carefully read all the instructions before running the setup.

#### NVIDIA CUDA Toolkit

The NVIDIA CUDA Toolkit can be downloaded from https://developer.nvidia.com/cuda-toolkit-archive. It will create a symbolic link to `/usr/local/cuda`. Versions can be switched with `update-alternatives`.

#### Torch

As we're building some bindings for Torch, we also need the different header files of `libtorch`. These can be downloaded from the official PyTorch website : https://pytorch.org/get-started/locally. The library must go to the `lib` directory.

For example :

```bash
cd lib
wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.4.1%2Bcu121.zip # libtorch 2.4.1 for CUDA 12.1 (Linux)
unzip libtorch-shared-with-deps-2.4.1%2Bcu121.zip
rm libtorch-shared-with-deps-2.4.1%2Bcu121.zip
```

> [!WARNING]
> Make sure you've downloaded the proper version of `libtorch` for both CUDA and your system. The `libtorch` version must match the PyTorch's one.

One other alternative is to use the headers shipped with PyTorch directly.

```bash
cd lib
ln -s $(poetry env info --path)/lib/python*/site-packages/torch libtorch
```

#### Python

Because we can use different versions of Python, we need to provide the proper headers. On Linux, these are usually stored to `/usr/include/python<x>.<y>`. One easy solution is just to create another symbolic link in the `lib` directory.

```bash
cd lib
ln -s /usr/include/$(basename $(realpath $(poetry env info --executable))) python
```

### Install

Prepare a virtual as follow :

```bash
poetry env use python
poetry lock
poetry install --with dev
```

### Build

Once everything is setup, the library can be build with the following script. This is a bit hacky as we don't have control on the build env used by Poetry, which is an issue here (as we need dynamic control over CUDA version, thus PyTorch version).
First, a build env must be prepared.

```bash
# First, install PyTorch for a given version of CUDA
poetry run pip install torch --index-url https://download.pytorch.org/whl/121

# Then install required build packages
poetry install --no-root --only build

# Generate a setup.py file
poetry run python poetry2setup.py
```

```bash
# Then build
poetry run python setup.py bdist_wheel
```

One can define `TORCH_CUDA_ARCH_LIST` to tune the CUDA architectures the library is built for. Use `MAX_JOBS` to tune build parallelism. Both variables use defaults if not provided.

```bash
# Select the architectures
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5" poetry run python setup.py bdist_wheel
```

The library can be built with debug mode by defined the `DEBUG_MODE=true` env variable.

### Quality

Code quality is ensured by several tools (`ruff`, `mypy` & `black`).

```bash
poetry run black src tests *.py
poetry run ruff check --fix src tests *.py
poetry run mypy src tests *.py
```

### Tests

Tests use a subset of the datasets from the [YACCLAB](https://github.com/prittt/YACCLAB) project [2]. These can be downloaded with the scripts in the [`datasets`](./datasets) directory.

```bash
cd datasets
./2d.sh # Download and extract the 2D data
./3d.sh # Download and extract the 3D data
```

PyTest is used to run unit tests. Only the GPU code is tested (after build). CPU code is assumed correct as it lies on third-party libraries.

```bash
poetry run pytest --cov=src/torchcc tests/unit
```

## Implementation

The following algorithms are used in the implementation. Some algorithms are not implemented yet. This is still a work in progress.

| Geometry | Connectivity | Algorithm | References | Implemented |
| -------- | ------------ | --------- | ---------- | :---------: |
| 2D       | 4            | HA4       | [3], [4]   |     ❌      |
| 2D       | 8            | BUF       | [1], [3]   |     ✔️      |
| 3D       | 6            |           |            |     ❌      |
| 3D       | 18           |           |            |     ❌      |
| 3D       | 26           | BUF       | [1], [3]   |     ❌      |

## References

This work is based on the following articles.

[1] Allegretti, S., Bolelli, F., & Grana, C. (2020). [Optimized Block-Based Algorithms to Label Connected Components on GPUs](https://federicobolelli.it/pub_files/2019tpds.pdf). IEEE Transactions on Parallel and Distributed Systems, 31(2), 423–438. https://doi.org/10.1109/TPDS.2019.2934683

[2] Grana, C., Bolelli, F., Baraldi, L., & Vezzani, R. (2016). [YACCLAB-Yet Another Connected Components Labeling Benchmark](https://federicobolelli.it/pub_files/2016icpr.pdf). https://doi.org/10.1109/ICPR.2016.7900112

[3] Bolelli, F., Allegretti, S., Lumetti, L., & Grana, C. (2024). [A State-of-the-Art Review with Code about Connected Components Labeling on GPUs](https://federicobolelli.it/pub_files/2024tpds.pdf).

[4] Hennequin, A., Lacassagne, L., Cabaret, L., Meunier, Q., & Meunier, Q. A. (2018). [A new Direct Connected Component Labeling and Analysis Algorithms for GPUs](https://hal.science/hal-01923784/document). https://doi.org/10.1109/dasip.2018.8596835ï

### Online Resources

Deeper details on CUDA can be found here.

- [CUDA C/C++ Basics](https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf)
- [CUDA C/C++ Streams and Concurrency](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)
- [CUDA Tutorial 2022 Part 1 : Introduction and CUDA Basics](https://cuda-tutorial.github.io/part1_22.pdf)
- [CUDA Tutorial 2022 Part 2 : GPU Hardware Details](https://cuda-tutorial.github.io/part2_22.pdf)
- [CUDA Cheatsheet](https://kdm.icm.edu.pl/Tutorials/GPU-intro/introduction.en)

## TODOs

Here is a list of possible improvements to investigate :

- [ ] Proper benchmarking.
- [ ] 2D 4-connectivity CCL.
- [ ] 3D CCL.
- [ ] Make IDs consecutive for 2D 8-connectivity CCL.
