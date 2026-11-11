# TorchCC

CUDA extension for PyTorch. It targets Connected Components algorithms on GPU. CPU calls go to [OpenCV](https://github.com/opencv/opencv-python) or [Connected Components 3D](https://github.com/seung-lab/connected-components-3d).

## Commands

Use `just` (run `just` to list recipes). Always use the `agent-*` recipes. They give short output and save tokens. Do not use `just lint`, `just check` or `just test`. CUDA recipes take a version, default `cu126` (`cu126`, `cu128`, `cu130`, `cu132`, `cu134`).

- `just agent-check [paths]`: format check, lint and type check. It does not modify files. Default paths are `src tests`.
- `just agent-test [pytest args]`: unit tests, stop at the first failure.
- `just agent-hooks`: all git hooks. They may modify files.
- `just agent-rebuild [cuXXX]`: rebuild the extension after a change in `csrc/` or `include/`.
- `just format`: format the Python code.
- `just data`: download test datasets (needed by tests).
- `just init [cuXXX]` and `just build [cuXXX]`: create the venv, and build a wheel. Rarely needed.

## Gotchas

- Do not run a bare `uv sync`, `uv run` or `uv add`. It replaces the CUDA torch and rebuilds outside Nix. Use `just` or `uv run --no-sync`.
- `uv sync` does not rebuild after a change in `csrc/` or `include/`. Run `just agent-rebuild [cuXXX]`.
- The project is an editable install. The extension is `src/torchcc/_cuda.so`.
- Tests need a CUDA GPU. They skip without it.

## Architecture

- `include/`: C++ headers.
- `csrc/registry.cpp`: op schema (`m.def`), CUDA dispatch (`m.impl`), empty `_cuda` Python module.
- `csrc/torchcc.cu`: host code (checks, allocation, kernel launch).
- `csrc/buf.cu`: BUF kernels (`init`, `merge`, `compress`, `finalize`).
- `src/torchcc/ccl.py`: public API for Connected Components Labelling (`ccl2d`, `ccl3d`) and Python validation.
- `setup.py`: CUDA arch list per CUDA version (`CUDA_ARCHS`), `DEBUG_MODE=true`, `TORCH_TARGET_VERSION`.

## C++ and CUDA rules

- Use only `torch/csrc/stable/*` and `torch/headeronly/*` headers (stable ABI, torch 2.10 and later).
- Check inputs with `STD_TORCH_CHECK`. It raises only `RuntimeError`. Python API might need to repeat each check with the right exception (`ValueError`, `TypeError`, `NotImplementedError`).
- Use a `DeviceGuard` and the stream from `aoti_torch_get_current_cuda_stream`.
- After each kernel launch, call `STD_CUDA_KERNEL_LAUNCH_CHECK()` (`torch/csrc/stable/macros.h`). Wrap CUDA runtime calls in `STD_CUDA_CHECK`.
- Stay asynchronous. No `cudaDeviceSynchronize`, `cudaMalloc` or host copies in an op. Allocate with `torch::stable` ops (caching allocator).
- Check `x.is_contiguous()` or make it contiguous before you take the data pointer. Favor robust approaches to contiguity. Flag to the user when not possible.
- Return early for an empty input (`numel() == 0`). A grid dimension of 0 fails the launch.
- Compute sizes and overflow checks in `int64_t` before you narrow to `uint32_t` or `int32_t`.
- Keep `gridDim.y` and `gridDim.z` at 65535 or less. Use a grid-stride loop when the size is not bounded.
- Kernel pointers: `const T *__restrict__` for inputs, `T *__restrict__` for outputs. Use `constexpr` instead of `#define` for constants.
- Bounds-check every global memory access, also neighbour reads at image borders.
- Coalesce memory access: consecutive `threadIdx.x` must read consecutive addresses (innermost dimension).
- Debug a crash or a wrong result with `DEBUG_MODE=true just agent-rebuild` and `CUDA_LAUNCH_BLOCKING=1`.

## Checklists

New op:

1. Declare it in `include/torchcc.h`.
2. Add `m.def` and `m.impl` in `csrc/registry.cpp`.
3. Implement it in `csrc/`. `setup.py` finds new `.cu` and `.cpp` files itself.
4. Add the Python wrapper and `__all__` entry (`ccl.py`, `__init__.py`).
5. Add a test.

New CUDA version:

1. `flake.nix` (`toolkits`).
2. `pyproject.toml` (group, `conflicts`, `sources`, index).
3. `setup.py` (`CUDA_ARCHS`).
4. `.github/workflows/release.yaml` (matrix).
5. `README.md` (compatibility tables).

## Python conventions

- Every file starts with `from __future__ import annotations`. Files in `src/` start with a copyright docstring.
- Docstrings use the numpy style.
- Ruff uses `select = ALL`. Use `# ruff: ignore[rule-name]` (not `noqa`) to silence a rule.
- Do not silence a rule that is not in the `ignore` list of `pyproject.toml`, unless you add a comment.
- Type checker is `ty`.
- Prefer the simplest construct. A Python developer must understand it at once.
- Type all public signatures. Use `X | None`, not `Optional[X]`. Prefer `object` to `Any`.
- `# ty: ignore[...]`, `# type: ignore` and `cast()` need a comment. Say what the checker gets wrong and why you cannot fix it.
- No bare `except:` or `except Exception:` without re-raise. Wrap third-party exceptions with `raise ... from`. Put the bad values in the message.
- No mutable default arguments, no module-level mutable state, no monkey-patching.
- No magic literals. Use named constants (see `src/torchcc/constants.py`).
- Prefer early returns and guard clauses to deep nesting.
- No `utils`, `helpers` or `common` module names. No wildcard imports. Define `__all__` in public modules.

## Testing

- Test files end with `_test.py` (a hook checks it). Warnings are errors in pytest. `pytest-randomly` is on.
- Tests compare GPU labels with OpenCV after `consecutivize` (the GPU IDs are not consecutive).
- Bug fix: first write a test that fails on the current code. Then fix.
- New feature: add the test with the code.
- One behavior per test. Name tests `test_<unit>_<scenario>`.
- Use `@pytest.mark.parametrize` instead of repeated test bodies. Use `pytest.param(..., id="dash-id")` when the values are not plain scalars.
- No test classes as namespaces. Use fixtures (`tests/unit/conftest.py`).
- Assert on outputs and raised exceptions, not on internal state.
- While iterating, run `just agent-check` and `just agent-test` on the smallest scope. Before handoff, run `just agent-hooks` and then `just agent-test`.

## Changelog

- A change in `src/`, `csrc/` or `include/` needs an entry in `CHANGELOG.md` under `[Unreleased]`.
- Use the sections `Added`, `Changed`, `Fixed`, `Removed`, in alphabetical order.
- One sentence per entry. Start with a verb. Describe the impact for the user, not the implementation.
- Docs, tests, CI and tooling (`justfile`, `flake.nix`, hooks) need no entry.
