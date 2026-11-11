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

      # Minimal CUDA_HOME, as expected by torch.utils.cpp_extension
      cudaHome =
        cudaPackages:
        pkgs.symlinkJoin {
          name = "cuda-home-${cudaPackages.cudaMajorMinorVersion}";
          paths = [
            cudaPackages.cuda_nvcc
            (lib.getDev cudaPackages.cuda_cudart)
            (lib.getLib cudaPackages.cuda_cudart)
            (lib.getStatic cudaPackages.cuda_cudart)
          ];
          postBuild = "[ -e $out/lib64 ] || ln -s lib $out/lib64";
        };

      # Python, PyTorch and NVIDIA runtime libs come from PyPI (through uv).
      # Nix only provides the CUDA toolkit.
      shell =
        { cudaPackages, withCompiler }:
        (
          if withCompiler then
            pkgs.mkShell.override { stdenv = cudaPackages.backendStdenv; }
          else
            pkgs.mkShellNoCC
        )
          {
            packages = [ pkgs.uv ];
            CUDA_HOME = cudaHome cudaPackages;
          };
    in
    {
      # cuXXX : local development, with a host compiler supported by this CUDA version.
      # cuXXX-ci : no Nix compiler, so the host toolchain of the machine is used.
      #            This keeps wheels free of Nix glibc and RPATHs.
      devShells.${system} = lib.concatMapAttrs (name: cudaPackages: {
        ${name} = shell {
          inherit cudaPackages;
          withCompiler = true;
        };
        "${name}-ci" = shell {
          inherit cudaPackages;
          withCompiler = false;
        };
      }) toolkits;

      formatter.${system} = pkgs.nixfmt-rfc-style;
    };
}
