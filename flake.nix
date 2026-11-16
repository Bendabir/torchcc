{
  description = "Development shells for TorchCC (one per CUDA toolkit version).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Only the manylinux toolchain is used (nix-builder/pkgs/manylinux)
    kernels = {
      url = "github:huggingface/kernels";
      flake = false;
    };
  };

  # Prebuilt glibc 2.28
  nixConfig = {
    extra-substituters = [ "https://huggingface.cachix.org" ];
    extra-trusted-public-keys = [
      "huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o="
    ];
  };

  outputs =
    { nixpkgs, kernels, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # CUDA is unfree
        config.allowUnfree = true;
        overlays = [
          (_final: prev: {
            # Name used by the HF Kernels manylinux toolchain. HF pins 0.15.2 because 0.19.2 breaks cc1plus.
            # nixpkgs is back on 0.15.2, and 0.19.3 has the fix (NixOS/patchelf#672).
            patchelf-old = prev.patchelf;
          })
        ];
      };
      inherit (pkgs) lib;

      # Names match the PyTorch wheel indices and the uv dependency groups
      toolkits = {
        cu126 = pkgs.cudaPackages_12_6;
        cu128 = pkgs.cudaPackages_12_8;
        cu130 = pkgs.cudaPackages_13_0;
        cu132 = pkgs.cudaPackages_13_2;
        cu134 = pkgs.cudaPackages_13_4;
      };

      # manylinux_2_28 toolchain from Hugging Face kernels: AlmaLinux 8 gcc-toolset RPMs and glibc 2.28.
      # Same compilers and glibc as the PyTorch wheels.
      manylinux = pkgs.callPackage "${kernels}/nix-builder/pkgs/manylinux" { } {
        packageMetadata = lib.importJSON "${kernels}/nix-builder/pkgs/manylinux/manylinux-2.28-x86_64-metadata.json";
      };

      # Explicit compiler per CUDA version, within the nvcc limits
      stdenvs = {
        cu126 = manylinux.gcc13Stdenv;
        cu128 = manylinux.gcc13Stdenv;
        cu130 = manylinux.gcc14Stdenv;
        cu132 = manylinux.gcc14Stdenv;
        cu134 = manylinux.gcc14Stdenv;
      };

      # Minimal CUDA_HOME, as expected by torch.utils.cpp_extension
      cudaHome =
        { cudaPackages, stdenv }:
        let
          # nixpkgs patches the CUDA math headers for glibc >= 2.42. The patch breaks with glibc 2.28.
          withGlibc = pkg: pkg.override { glibc = stdenv.cc.libc; };
          nvcc = withGlibc cudaPackages.cuda_nvcc;
        in
        pkgs.symlinkJoin {
          name = "cuda-home-${cudaPackages.cudaMajorMinorVersion}";
          paths = [
            nvcc
            # Provides nv/target, needed by cuda_fp16.h
            cudaPackages.cccl
            (lib.getDev cudaPackages.cuda_cudart)
            (lib.getLib cudaPackages.cuda_cudart)
            (lib.getStatic cudaPackages.cuda_cudart)
          ]
          # CUDA 13 moved the crt/ headers out of cuda_nvcc.
          # CUDA 12 has an unsupported cuda_crt stub, so check availability too.
          ++ lib.optional (
            cudaPackages ? cuda_crt && lib.meta.availableOn pkgs.stdenv.hostPlatform cudaPackages.cuda_crt
          ) (withGlibc cudaPackages.cuda_crt);
          # nvcc.profile points to the nixpkgs CUDA compiler, and it wins over `-ccbin gcc`. Use ours.
          postBuild = ''
            [ -e $out/lib64 ] || ln -s lib $out/lib64
            rm $out/bin/nvcc.profile
            sed 's|^compiler-bindir.*|compiler-bindir = ${stdenv.cc}/bin|' \
              ${nvcc}/bin/nvcc.profile > $out/bin/nvcc.profile
          '';
        };

      # Python, PyTorch and NVIDIA runtime libs come from PyPI (through uv).
      # Nix only provides the CUDA toolkit.
      shell =
        { cudaPackages, stdenv }:
        pkgs.mkShell.override { inherit stdenv; } {
          packages = [ pkgs.uv ];
          CUDA_HOME = cudaHome { inherit cudaPackages stdenv; };
          # No RPATH to glibc 2.28 or the Nix store. Python loads glibc, libstdc++ and PyTorch first.
          NIX_DONT_SET_RPATH = 1;
          NIX_NO_SELF_RPATH = 1;
        };
    in
    {
      # cuXXX : manylinux_2_28 compiler supported by this CUDA version, for local development and CI.
      devShells.${system} = lib.mapAttrs (
        name: cudaPackages:
        shell {
          inherit cudaPackages;
          stdenv = stdenvs.${name};
        }
      ) toolkits;

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}
