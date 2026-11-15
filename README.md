# Torch Connected Components

TorchCC is a library that brings efficient CUDA implementations of Connected Components related algorithms (such as Connected Components Labeling).
CPU support is delegated to external libraries such as OpenCV. It is built to support multiple CUDA and PyTorch versions.

> [!NOTE]
> This library is still an early prototype. It's definitely not suitable for production. Only Linux is supported at the moment. Windows can be built from source, but this is experimental and untested. There are no Windows wheels yet as this would require some more efforts. The provided instructions only describe Linux setup and builds.

## Install

### From Package Manager

You can install TorchCC by using a custom PyPI index. There is one index per supported version of CUDA. This is still experimental though.

#### pip

```bash
pip install --index-url https://bendabir.github.io/pypi/cu126 "torchcc>=0.2.0"
```

#### uv

```toml
[project]
dependencies = ["torchcc>=0.2.0"]

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
torchcc = { version = "^0.2.0", source = "bendabir-cu126" }
```

### Direct Install

> [!NOTE]
> This is a temporary solution. I might also upload artifacts to PyPI, but it means that one version of CUDA needs to be selected. It's not optimal as I'd like to uncouple CUDA versions from the library version.

1. Go to the [release page](https://github.com/Bendabir/torchcc/releases).
2. Download a Wheel file for your version of CUDA and Python. Not all versions are supported (see [the compatibility matrix](#compatibility)).
3. Install it in your env. For example : `pip install torchcc-0.2.0+cu126-cp312-cp312-linux_x86_64.whl`

### Verify a Wheel

Release wheels are signed with [Sigstore](https://docs.sigstore.dev/) (GitHub artifact attestations). To check that a wheel comes from this repository's release workflow :

```bash
gh attestation verify torchcc-0.2.0+cu126-cp312-cp312-linux_x86_64.whl --repo bendabir/torchcc
```

Only wheels from releases made after this change carry an attestation.

## Usage

All algorithms are implemented in a batch fashion for easy integration with PyTorch.

### 2D Connected Components Labeling

```python
labels = torchcc.ccl2d(masks, connectivity = 8) # masks : [N x H x W], connectivity 4 or 8
```

### 3D Connected Components Labeling

```python
labels = torchcc.ccl3d(volumes, connectivity = 26) # volumes : [N x H x W x D], connectivity 6, 18 or 26
```

On CUDA, only 26-connectivity is supported for now. Labels are unique within a volume, not consecutive.

### CPU support

CPU support is delegated to third-party libraries. [OpenCV](https://github.com/opencv/opencv-python) is used for 2D Connected Components Labeling on CPU, while [ConnectedComponents3D](https://github.com/seung-lab/connected-components-3d) is used for 3D counterpart. Both libraries are optional. Install them for transparent CPU support. Extras are available (`opencv`, `opencv-headless` and `cc3d`).

```bash
pip install "opencv-python-headless>=4,<6" # 2D
pip install "connected-components-3d>=3,<5" # 3D
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

The minimum Python version is the one of PyTorch. TorchCC itself requires Python 3.11 or newer.

More details : https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix

</details>

## Benchmarks

GPU is an NVIDIA GeForce RTX 3080 Ti. CPU is an Intel Core i7-12700KF. The speedup is the CPU time divided by the GPU time. Median time in milliseconds. Benchmark is executed in various [YACCLAB](https://github.com/prittt/YACCLAB) datasets. Figures are provided against a naive CPU execution of OpenCV CCL (2D) or ConnectedComponents3D (3D).

GPU times exclude the transfers between host and device. Run `uv run --no-sync python scripts/benchmark.py --connectivity <4|8>` (2D) or `uv run --no-sync python scripts/benchmark.py ccl3d` (3D) to reproduce (see `just data` first).

| Task | Geometry | Connectivity | Speedup (mean) | Speedup (geometric mean) |
| ---- | -------- | ------------ | -------------- | ------------------------ |
| CCL  | 2D       | 4            | 30.1           | 17.4                     |
| CCL  | 2D       | 8            | 32.6           | 18.2                     |
| CCL  | 3D       | 26           | 110.0          | 94.1                     |

<details>
  <summary><b>CCL 2D 4-connectivity Figures</b></summary>

| Dataset      | Batch | Shape     | CPU (ms) | GPU (ms) | Speedup |
| ------------ | ----: | --------- | -------: | -------: | ------: |
| 3dpes        |     1 | 576x704   |    0.360 |    0.057 |    6.4x |
| 3dpes        |     8 | 576x704   |    4.069 |    0.097 |   42.1x |
| 3dpes        |    32 | 576x704   |   12.090 |    0.248 |   48.7x |
| 3dpes        |    64 | 576x704   |   22.759 |    0.406 |   56.0x |
| fingerprints |     1 | 300x300   |    0.142 |    0.062 |    2.3x |
| fingerprints |     8 | 300x300   |    1.294 |    0.076 |   17.1x |
| fingerprints |    32 | 300x300   |    6.603 |    0.171 |   38.6x |
| fingerprints |    64 | 300x300   |   12.881 |    0.291 |   44.3x |
| hamlet       |     1 | 2125x1273 |    2.567 |    0.146 |   17.5x |
| hamlet       |     8 | 2125x1273 |   20.029 |    0.529 |   37.9x |
| hamlet       |    32 | 2125x1273 |   78.808 |    1.949 |   40.4x |
| hamlet       |    64 | 2125x1273 |  152.743 |    3.824 |   39.9x |
| medical      |     1 | 847x1920  |    1.404 |    0.116 |   12.1x |
| medical      |     8 | 847x1920  |   12.913 |    0.339 |   38.1x |
| medical      |    32 | 847x1920  |   52.696 |    1.160 |   45.4x |
| medical      |    64 | 847x1920  |   99.772 |    2.280 |   43.8x |
| mirflickr    |     1 | 375x500   |    0.134 |    0.078 |    1.7x |
| mirflickr    |     8 | 375x500   |    2.899 |    0.125 |   23.1x |
| mirflickr    |    32 | 375x500   |   11.384 |    0.256 |   44.5x |
| mirflickr    |    64 | 375x500   |   22.714 |    0.430 |   52.9x |
| random       |     1 | 32x32     |    0.008 |    0.049 |    0.2x |
| random       |     8 | 32x32     |    0.022 |    0.056 |    0.4x |
| random       |    32 | 32x32     |    0.114 |    0.073 |    1.6x |
| random       |    64 | 32x32     |    0.358 |    0.067 |    5.3x |
| tobacco800   |     1 | 1575x1200 |    1.504 |    0.102 |   14.8x |
| tobacco800   |     8 | 1575x1200 |   11.290 |    0.356 |   31.8x |
| tobacco800   |    32 | 1575x1200 |   66.421 |    1.131 |   58.7x |
| tobacco800   |    64 | 1575x1200 |  122.265 |    2.178 |   56.1x |
| xdocs        |     1 | 3381x5034 |   11.760 |    0.515 |   22.8x |
| xdocs        |     8 | 3381x5034 |  117.430 |    3.137 |   37.4x |
| xdocs        |    32 | 3381x5034 |  484.724 |   12.267 |   39.5x |
| xdocs        |    64 | 3381x5034 |  995.372 |   24.295 |   41.0x |

</details>

<details>
  <summary><b>CCL 2D 8-connectivity Figures</b></summary>

| Dataset      | Batch | Shape     | CPU (ms) | GPU (ms) | Speedup |
| ------------ | ----: | --------- | -------: | -------: | ------: |
| 3dpes        |     1 | 576x704   |    0.264 |    0.062 |    4.2x |
| 3dpes        |     8 | 576x704   |    2.022 |    0.093 |   21.7x |
| 3dpes        |    32 | 576x704   |   10.537 |    0.216 |   48.8x |
| 3dpes        |    64 | 576x704   |   22.509 |    0.387 |   58.2x |
| fingerprints |     1 | 300x300   |    0.135 |    0.065 |    2.1x |
| fingerprints |     8 | 300x300   |    1.577 |    0.111 |   14.2x |
| fingerprints |    32 | 300x300   |    5.907 |    0.148 |   39.8x |
| fingerprints |    64 | 300x300   |   11.357 |    0.184 |   61.9x |
| hamlet       |     1 | 2125x1273 |    1.846 |    0.086 |   21.4x |
| hamlet       |     8 | 2125x1273 |   17.982 |    0.399 |   45.0x |
| hamlet       |    32 | 2125x1273 |   69.041 |    1.383 |   49.9x |
| hamlet       |    64 | 2125x1273 |  150.140 |    2.811 |   53.4x |
| medical      |     1 | 847x1920  |    1.331 |    0.111 |   12.0x |
| medical      |     8 | 847x1920  |   10.278 |    0.302 |   34.0x |
| medical      |    32 | 847x1920  |   53.315 |    1.031 |   51.7x |
| medical      |    64 | 847x1920  |   96.121 |    2.049 |   46.9x |
| mirflickr    |     1 | 375x500   |    0.125 |    0.070 |    1.8x |
| mirflickr    |     8 | 375x500   |    1.959 |    0.108 |   18.1x |
| mirflickr    |    32 | 375x500   |    9.973 |    0.188 |   53.0x |
| mirflickr    |    64 | 375x500   |   18.091 |    0.322 |   56.3x |
| random       |     1 | 32x32     |    0.008 |    0.040 |    0.2x |
| random       |     8 | 32x32     |    0.020 |    0.058 |    0.3x |
| random       |    32 | 32x32     |    0.157 |    0.069 |    2.3x |
| random       |    64 | 32x32     |    0.415 |    0.077 |    5.4x |
| tobacco800   |     1 | 1575x1200 |    1.071 |    0.096 |   11.1x |
| tobacco800   |     8 | 1575x1200 |   10.529 |    0.267 |   39.5x |
| tobacco800   |    32 | 1575x1200 |   46.636 |    0.881 |   52.9x |
| tobacco800   |    64 | 1575x1200 |   91.048 |    1.722 |   52.9x |
| xdocs        |     1 | 3381x5034 |   12.579 |    0.359 |   35.0x |
| xdocs        |     8 | 3381x5034 |   87.683 |    2.116 |   41.4x |
| xdocs        |    32 | 3381x5034 |  464.194 |    8.682 |   53.5x |
| xdocs        |    64 | 3381x5034 |  990.620 |   17.851 |   55.5x |

</details>

<details>
  <summary><b>CCL 3D 26-connectivity Figures</b></summary>

| Dataset              | Batch | Shape        | CPU (ms) | GPU (ms) | Speedup |
| -------------------- | ----: | ------------ | -------: | -------: | ------: |
| hilbert              |     1 | 130x130x130  |    2.671 |    0.077 |   34.9x |
| hilbert              |     4 | 130x130x130  |   11.927 |    0.162 |   73.7x |
| hilbert              |     8 | 130x130x130  |   24.628 |    0.285 |   86.3x |
| mitochondria         |     1 | 165x768x1024 |  213.918 |    1.826 |  117.1x |
| mitochondria         |     4 | 165x768x1024 |  959.348 |    7.255 |  132.2x |
| mitochondria         |     8 | 165x768x1024 | 1863.677 |   14.496 |  128.6x |
| oasis                |     1 | 128x256x256  |   25.884 |    0.356 |   72.8x |
| oasis                |     4 | 128x256x256  |   96.221 |    1.064 |   90.4x |
| oasis                |     8 | 128x256x256  |  182.521 |    2.011 |   90.8x |
| random3D/classical   |     1 | 256x256x256  |  171.280 |    0.814 |  210.5x |
| random3D/classical   |     4 | 256x256x256  |  671.213 |    2.948 |  227.7x |
| random3D/classical   |     8 | 256x256x256  | 1368.573 |    6.011 |  227.7x |
| random3D/granularity |     1 | 256x256x256  |   10.687 |    0.241 |   44.4x |
| random3D/granularity |     4 | 256x256x256  |   44.690 |    0.772 |   57.9x |
| random3D/granularity |     8 | 256x256x256  |   83.235 |    1.514 |   55.0x |

</details>

## Development

The CUDA Toolkit comes from Nix (see [`flake.nix`](./flake.nix)). Python, PyTorch and the NVIDIA runtime libraries come from PyPI through [uv](https://docs.astral.sh/uv/). There is one Nix shell and one uv dependency group per CUDA version : `cu126`, `cu128`, `cu130`, `cu132` and `cu134`.

Common tasks are available as [just](https://just.systems) recipes. Run `just` to list them. Recipes that depend on CUDA take the version as argument (default : `cu126`), for example `just build cu130`.

### Tools

| Tool                                                                                                                                    | Usage                                                                                           | Install                                                            |
| --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [Nix](https://nixos.org)                                                                                                                | CUDA Toolkit, compilers and `uv`. Flakes must be [enabled](https://wiki.nixos.org/wiki/Flakes). | [Download](https://nixos.org/download/)                            |
| [just](https://just.systems)                                                                                                            | Task runner (see the [`justfile`](./justfile)).                                                 | [Packages](https://just.systems/man/en/packages.html)              |
| [uv](https://docs.astral.sh/uv/)                                                                                                        | Python env, dependencies and wheel build.                                                       | [Install](https://docs.astral.sh/uv/getting-started/installation/) |
| [prek](https://prek.j178.dev)                                                                                                           | Git hooks. Installed by `just init`.                                                            | Via `uv` (dev group)                                               |
| [Ruff](https://docs.astral.sh/ruff/)                                                                                                    | Python format and lint.                                                                         | Via `uv` (dev group)                                               |
| [ty](https://docs.astral.sh/ty/)                                                                                                        | Python type check.                                                                              | Via `uv` (dev group)                                               |
| [nixfmt](https://github.com/NixOS/nixfmt), [deadnix](https://github.com/astro/deadnix), [statix](https://github.com/oppiliappan/statix) | Nix format and lint. Used by the git hooks.                                                     | Must be on your `PATH` (not in the Nix shell)                      |
| [Prettier](https://prettier.io)                                                                                                         | JSON, Markdown and YAML format. Used by the git hooks.                                          | Managed by `prek`                                                  |
| [GitHub CLI](https://cli.github.com)                                                                                                    | Optional. Verify the signature of a release wheel (see [Verify a Wheel](#verify-a-wheel)).      | [Install](https://github.com/cli/cli#installation)                 |

Typical Nix usage :

```bash
nix develop .#cu126 # Shell with the CUDA 12.6 Toolkit
nix develop .#cu126 -c <command> # Run one command in that shell
```

Typical just usage :

```bash
just # List recipes
just init cu126 # Create the env
just test -k ccl2d # Run a subset of the tests
just data # Download the test datasets
```

### Install

Create the virtual env for a CUDA version (default `cu126`) and install the git hooks.

```bash
just init cu126
```

This runs `nix develop .#cu126 -c uv sync --group build --group cu126` and `uv run prek install`.

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
UV_PROJECT_ENVIRONMENT=.build nix develop .#cu126 -c uv sync --no-default-groups --group dev --group build --group cu126 --no-install-project
nix develop .#cu126 -c uv build --wheel --no-build-isolation --no-config --python .build/bin/python
```

One can define `TORCH_CUDA_ARCH_LIST` to tune the CUDA architectures the library is built for. Use `MAX_JOBS` to tune build parallelism. Both variables use defaults if not provided.

```bash
# Select the architectures
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5" just build cu126
```

The library can be built with debug mode by defining the `DEBUG_MODE=true` env variable.

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

| Task | Geometry | Connectivity | Algorithm | References | Implemented |
| ---- | -------- | ------------ | --------- | ---------- | :---------: |
| CCL  | 2D       | 4            | HA4       | [3], [4]   |     ✔️      |
| CCL  | 2D       | 8            | BUF       | [1], [3]   |     ✔️      |
| CCL  | 3D       | 6            |           |            |     ❌      |
| CCL  | 3D       | 18           |           |            |     ❌      |
| CCL  | 3D       | 26           | BUF       | [1], [3]   |     ✔️      |

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
