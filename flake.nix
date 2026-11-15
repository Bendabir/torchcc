{
  description = "Development shells for TorchCC (one per CUDA toolkit version).";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        # CUDA is unfree
        config.allowUnfree = true;
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

      # Compiler overrides, when the default one of a CUDA version does not build.
      stdenvs = {
        # CUDA 12.8 fails with the default G++ 14
        cu128 = pkgs.gcc13Stdenv;
      };

      # Minimal CUDA_HOME, as expected by torch.utils.cpp_extension
      cudaHome =
        cudaPackages:
        pkgs.symlinkJoin {
          name = "cuda-home-${cudaPackages.cudaMajorMinorVersion}";
          paths = [
            cudaPackages.cuda_nvcc
            # Provides nv/target, needed by cuda_fp16.h
            cudaPackages.cccl
            (lib.getDev cudaPackages.cuda_cudart)
            (lib.getLib cudaPackages.cuda_cudart)
            (lib.getStatic cudaPackages.cuda_cudart)
          ]
          # CUDA 13 moved the crt/ headers out of cuda_nvcc
          ++ lib.optional (cudaPackages ? cuda_crt) cudaPackages.cuda_crt;
          postBuild = "[ -e $out/lib64 ] || ln -s lib $out/lib64";
        };

      # Python, PyTorch and NVIDIA runtime libs come from PyPI (through uv).
      # Nix only provides the CUDA toolkit.
      shell =
        { cudaPackages, stdenv }:
        (if stdenv != null then pkgs.mkShell.override { inherit stdenv; } else pkgs.mkShellNoCC) {
          packages = [ pkgs.uv ];
          CUDA_HOME = cudaHome cudaPackages;
        };
    in
    {
      # cuXXX : local development, with a host compiler supported by this CUDA version.
      # cuXXX-ci : no Nix compiler, so the host toolchain of the machine is used.
      #            This keeps wheels free of Nix glibc and RPATHs.
      #            Exception: versions in `stdenvs` use their Nix compiler.
      devShells.${system} = lib.concatMapAttrs (name: cudaPackages: {
        ${name} = shell {
          inherit cudaPackages;
          stdenv = stdenvs.${name} or cudaPackages.backendStdenv;
        };
        "${name}-ci" = shell {
          inherit cudaPackages;
          stdenv = stdenvs.${name} or null;
        };
      }) toolkits;

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}
