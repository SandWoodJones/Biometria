{
  description = "Projeto final Biometria UTFPR";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python312;

      wheelLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        zstd
        bzip2
        xz
        libffi
        numactl
        libdrm
        elfutils
        libsndfile
        ffmpeg.lib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python
          uv
          ffmpeg
          sox
          typst
          git
        ];

        env = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath wheelLibs;
          UV_PYTHON = "${python}/bin/python3";
          UV_PYTHON_DOWNLOADS = "never";
          HSA_OVERRIDE_GFX_VERSION = "10.3.0";
          PYTHONHASHSEED = "0";
        };

        shellHook = ''
          if [ ! -d .venv ]; then
            uv venv --quiet .venv
          fi
          source .venv/bin/activate
        '';
      };
    };
}
