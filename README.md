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
pip install --index-url https://bendabir.github.io/pypi/cu126 torchcc
```

#### uv

```toml
[project]
dependencies = ["torchcc>=0.1.0"]

[tool.uv.sources]
torchcc = { index = "bendabir-cu126" }

[[tool.uv.index]]
name = "bendabir-cu126"
url = "https://bendabir.github.io/pypi/cu126"
explicit = true
```

#### Poetry

```toml
[[tool.poetry.source]]
name = "bendabir-cu126"
url = "https://bendabir.github.io/pypi/cu126"
priority = "explicit"

[tool.poetry.dependencies]
torchcc = { version = "^0.1.0", source = "bendabir-cu126" }
```

### Direct Install

> [!NOTE]
> This is a temporary solution. I might also upload artifacts to PyPI, but it means that one version of CUDA needs to be selected. It's not optimal as I'd like to uncouple CUDA versions from the library version.

1. Go to the [release page](https://github.com/Bendabir/torchcc/releases).
2. Download a Wheel file for your version of CUDA and Python. Not all versions are supported (see [the compatibility matrix](#compatibility)).
3. Install it in your env. For example : `pip install torchcc-0.1.0+cu126-cp312-cp312-linux_x86_64.whl`

## Usage

All algorithms are implemented in a batch fashion for easy integration with PyTorch.

### 2D Connected Components Labeling

```python
labels = torchcc.ccl2d(masks, connectivity = 8) # masks : [N x H x W]
```

### 3D Connected Components Labeling

**TODO**

### CPU support

CPU support is delegated to third-party libraries. [OpenCV](https://github.com/opencv/opencv-python) is used for 2D Connected Components Labeling on CPU, while [ConnectedComponents3D](https://github.com/seung-lab/connected-components-3d) is used for 3D counterpart. Both libraries are optional. Install them for transparent CPU support. Extras are available (`opencv`, `opencv-headless` and `cc3d`).

```bash
pip install "opencv-python-headless>=4,<5" # 2D
pip install "connected-components-3d>=3,<4" # 3D
```

## Compatibility

TorchCC requires Python 3.11 or newer and PyTorch 2.10 or newer. It relies on the PyTorch stable ABI, so one wheel works with several PyTorch versions. There is one wheel per Python version and per CUDA version, like PyTorch.

| CUDA Version | PyTorch Version                        | Supported |
| ------------ | -------------------------------------- | :-------: |
| `12.6`       | `2.10`, `2.11`, `2.12`, `2.13`, `2.14` |    ✔️     |
| `12.8`       | `2.10`, `2.11`                         |    ✔️     |
| `13.0`       | `2.10`, `2.11`, `2.12`, `2.13`, `2.14` |    ✔️     |
| `13.2`       | `2.12`, `2.13`, `2.14`, `2.15`         |    ✔️     |
| `13.4`       | `2.15`                                 |    ✔️     |

<details>
  <summary><b>PyTorch Compatibility Matrix</b></summary>

| Torch Version | CUDA 12.6 | CUDA 12.8 | CUDA 13.0 | CUDA 13.2 | CUDA 13.4 | Min. Python Version | Max. Python Version | C++ Version |
| ------------- | --------- | --------- | --------- | --------- | --------- | ------------------- | ------------------- | ----------- |
| 2.10          | ☑️        | ✅        | 🟦        | ❌        | ❌        | 3.10                | 3.14                | C++17       |
| 2.11          | ☑️        | ☑️        | ✅        | ❌        | ❌        | 3.10                | 3.14                | C++17       |
| 2.12          | ☑️        | ❌        | ✅        | 🟦        | ❌        | 3.10                | 3.14                | C++17       |
| 2.13          | ☑️        | ❌        | ✅        | 🟦        | ❌        | 3.10                | 3.15                | C++20       |
| 2.14          | ☑️        | ❌        | ☑️        | ✅        | ❌        | 3.10                | 3.15                | C++20       |
| 2.15          | ❌        | ❌        | ❌        | ✅        | 🟦        | 3.11                | 3.15                | C++20       |

- ✅ : Default Stable CUDA
- ☑️ : Stable CUDA
- 🟦 : Experimental CUDA
- ❌ : Unsupported

More details : https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix

</details>

## Development

The CUDA Toolkit comes from Nix (see [`flake.nix`](./flake.nix)). Python, PyTorch and the NVIDIA runtime libraries come from PyPI through [uv](https://docs.astral.sh/uv/). There is one Nix shell and one uv dependency group per CUDA version : `cu126`, `cu128`, `cu130`, `cu132` and `cu134`.

Common tasks are available as [just](https://just.systems) recipes. Run `just` to list them. Recipes that depend on CUDA take the version as argument (default : `cu126`), for example `just build cu130`.

### Install

Create the virtual env for a CUDA version (default `cu126`) and install the git hooks.

```bash
just init cu126
```

This runs `nix develop .#cu126 -c uv sync --group cu126` and `uv run prek install`.

The `cuXXX` shells provide a host compiler for local development. The `cuXXX-ci` shells do not, so the compiler of the machine is used (this is what CI does to build wheels).

#### Why `uv` does not sync automatically

The CUDA version is a dependency group (`cuXXX`) that is not a default group. A plain `uv sync` or `uv run` only selects the default groups. It would replace the CUDA build of PyTorch by the default one from PyPI, then rebuild the extension outside of the Nix shell (no compiler, so it fails). To avoid this :

- Only `just init` (and `just build`) syncs the environment, and always inside the Nix shell.
- The `justfile` exports `UV_NO_SYNC=1`, so `uv run` in the other recipes uses the existing `.venv` as is. This variable only affects `uv run` and only recipes started through `just`. A bare `uv run` in your shell still syncs, so use `uv run --no-sync` there.
- The `ruff` hooks use `uv run --only-group dev` and the `ty` hook uses `uv run --no-sync`. They never rebuild the project.
- The `uv-sync` hook (after checkout, merge and rewrite) installs only the `dev` group with `--inexact` (`uv sync --frozen --only-group dev --inexact`). Excluding the project leaves PyTorch alone, and `--inexact` keeps the packages of the CUDA group. `--inexact` alone is not enough, as PyTorch is a direct dependency of the project and would still be replaced.

Run `just init <cuda>` again to switch CUDA version, or after a change of `uv.lock`.

### Editor

For VSCode IntelliSense, link the PyTorch and Python headers in the `lib` directory.

```bash
cd lib
ln -s ../.venv/lib/python*/site-packages/torch libtorch
ln -s "$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_path("include"))')" python
```

### Build

`just init` already builds the library in the virtual env. To build a wheel (the way CI does) :

```bash
just build cu126
```

This runs the following commands.

```bash
nix develop .#cu126-ci -c uv sync --no-default-groups --group dev --group build --group cu126 --no-install-project
nix develop .#cu126-ci -c uv build --wheel --no-build-isolation --python .venv/bin/python
```

One can define `TORCH_CUDA_ARCH_LIST` to tune the CUDA architectures the library is built for. Use `MAX_JOBS` to tune build parallelism. Both variables use defaults if not provided.

```bash
# Select the architectures
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5" just build cu126
```

The library can be built with debug mode by defined the `DEBUG_MODE=true` env variable.

### Quality

Code quality is ensured by several tools (`ruff` & `ty`).

```bash
just format # ruff format
just lint # ruff check --fix & ty check
```

These checks (and the Nix ones : `nixfmt`, `deadnix` & `statix`) also run as git hooks with [prek](https://prek.j178.dev). `just init` installs the hooks. Run them on demand with :

```bash
just check
```

### Tests

Tests use a subset of the datasets from the [YACCLAB](https://github.com/prittt/YACCLAB) project [2]. These can be downloaded with the scripts in the [`datasets`](./datasets) directory.

```bash
just data # Download and extract the 2D and 3D data (see datasets/2d.sh and datasets/3d.sh)
```

PyTest is used to run unit tests. Only the GPU code is tested (after build). CPU code is assumed correct as it lies on third-party libraries.

```bash
just test # Extra arguments go to pytest, for example : just test -k ccl2d
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

[4] Hennequin, A., Lacassagne, L., Cabaret, L., Meunier, Q., & Meunier, Q. A. (2018). [A new Direct Connected Component Labeling and Analysis Algorithms for GPUs](https://hal.science/hal-01923784/document). https://doi.org/10.1109/dasip.2018.8596835

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
