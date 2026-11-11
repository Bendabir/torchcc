# Torch Connected Components

set shell := ["bash", "-euo", "pipefail", "-c"]

# NixOS only : nix-ld exposes its libraries in NIX_LD_LIBRARY_PATH. It is prepended to any existing LD_LIBRARY_PATH.
export LD_LIBRARY_PATH := trim_start_match(trim_end_match(env('NIX_LD_LIBRARY_PATH', '') + ':' + env('LD_LIBRARY_PATH', ''), ':'), ':')

# Avoid uv to rebuild everything unnecessarily.
export UV_NO_SYNC := "1"

# CUDA version to develop and build against (see `flake.nix` and `pyproject.toml`)
cuda126 := "cu126"

# Display the list of available recipes.
[private]
default:
    @{{ just_executable() }} --list

alias help := default

# Create the virtual env for a CUDA version and install the git hooks.
[group('setup')]
init CUDA=cuda126:
    nix develop .#{{ CUDA }} -c uv sync --group build --group {{ CUDA }}
    uv run prek install

# Update the lock file
[group('setup')]
update:
    uv lock --upgrade

# Remove caches and build artifacts.
[group('setup')]
clean:
    rm -rf build dist .build .ruff_cache .pytest_cache .coverage

# Build a wheel for a CUDA version (as CI does). Honors TORCH_CUDA_ARCH_LIST, MAX_JOBS and DEBUG_MODE.
[group('build')]
build CUDA=cuda126:
    UV_PROJECT_ENVIRONMENT=.build nix develop .#{{ CUDA }} -c uv sync --no-default-groups --group dev --group build --group {{ CUDA }} --no-install-project
    nix develop .#{{ CUDA }} -c uv build --wheel --no-build-isolation --no-config --python .build/bin/python

# Format the Python code.
[group('quality')]
format:
    uv run ruff format

# Lint and type check the Python code.
[group('quality')]
lint:
    uv run ruff check --fix
    uv run ty check

# Run all git hooks (Python and Nix) on all files.
[group('quality')]
check:
    uv run prek run --all-files

# Download the test datasets.
[group('test')]
[working-directory('datasets')]
data:
    ./2d.sh
    ./3d.sh

# Run the unit tests with coverage. Extra arguments go to pytest.
[group('test')]
test *ARGS:
    uv run pytest --cov=src/torchcc {{ ARGS }} tests/unit

# Check without modifying files. Concise output for agents. Defaults to src and tests.
[group('agent')]
[private]
agent-check *PATHS:
    @uv run ruff format --check --quiet --color never {{ if PATHS == "" { "src tests" } else { PATHS } }}
    @uv run ruff check --quiet --color never --output-format concise {{ if PATHS == "" { "src tests" } else { PATHS } }}
    @uv run ty check --quiet --color never --output-format concise {{ if PATHS == "" { "src tests" } else { PATHS } }}

# Run the unit tests concisely and stop at the first failure. Extra arguments go to pytest.
[group('agent')]
[private]
agent-test *ARGS:
    @uv run pytest --quiet --tb=short --show-capture=no --no-header --color=no --maxfail=1 {{ ARGS }} tests/unit \
        | sed -E '/^[.FEsxX]+([[:space:]]+\[[[:space:]]*[0-9]+%\])?$/d'

# Run all git hooks quietly. Hooks may modify files.
[group('agent')]
[private]
agent-hooks:
    @uv run prek run --all-files --quiet --no-progress --color never

# Rebuild the extension after a change in csrc or include. Quiet unless it fails.
[group('agent')]
[private]
agent-rebuild CUDA=cuda126:
    @nix develop .#{{ CUDA }} -c uv sync --quiet --group build --group {{ CUDA }} --reinstall-package torchcc
