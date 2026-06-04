# PythonAnalysisEinsteinToolkit

Analysis and visualization pipeline for Einstein Toolkit simulation output.

| Directory    | Contents                                                                 |
| ------------ | ------------------------------------------------------------------------ |
| `Resample/`  | MPI-parallel resampling of 2D slices onto a regular grid, saved as HDF5. |
| `Frames/`    | MPI-parallel rendering of movie frames (and mp4) from the HDF5 files.    |
| `Notebooks/` | Interactive analysis; previews frames and writes the movie config.       |
| `Archive/`   | Legacy pickle-based scripts and notebooks (superseded by the above).     |

See `Resample/README.md` and `Frames/README.md` for usage of each stage.

## Conda environment

A single environment that runs every script and notebook in the repo
(both resampling backends, frame rendering, and the notebooks):

```bash
conda create -n etk-analysis -c conda-forge python=3.11 \
    numpy scipy matplotlib h5py pyyaml mpi4py ffmpeg jupyter
conda activate etk-analysis

# kuibit (resample kuibit backend, GW-extraction notebooks) — needs Python >= 3.9:
pip install kuibit

# postcactus (resample postcactus backend, Archive scripts) — not on PyPI.
# NOTE: the resampling pipeline's startup features (simdir_pickle cache,
# simdir_exclude, scan-once-and-broadcast) require the `dev` branch of
# https://github.com/MChabanov/PyCactus — a fork of wokast/PyCactus with a
# picklable SimDir and scan-exclusion options. Install from there:
pip install "git+https://github.com/MChabanov/PyCactus.git@dev#subdirectory=PostCactus"
```

What each package is needed for:

| Package      | Used by                                                            |
| ------------ | ------------------------------------------------------------------ |
| `numpy`      | everything                                                         |
| `scipy`      | GW-extraction / omega notebooks (interpolation, integration)       |
| `matplotlib` | `Frames/make_frames.py`, notebooks, Archive movie makers           |
| `h5py`       | resamplers, `read_data.py`, `make_frames.py`                       |
| `pyyaml`     | YAML configs for `Resample/` and `Frames/`                         |
| `mpi4py`     | MPI parallelism in resamplers, `make_frames.py`, Archive scripts   |
| `ffmpeg`     | mp4 encoding, called by `make_frames.py` via subprocess            |
| `jupyter`    | the notebooks                                                      |
| `kuibit`     | `resample_2d_data_kuibit.py`, kuibit-based notebooks               |
| `postcactus` | `resample_2d_data_postcactus.py`, `Archive/` scripts and notebooks |

Verify the install:

```bash
python -c "import numpy, scipy, matplotlib, h5py, yaml, mpi4py, kuibit, postcactus; print('ok')"
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

### Notes

- **numpy version:** kuibit pins `numpy < 2`, so the `pip install kuibit` step
  downgrades the conda-installed numpy (e.g. 2.x → 1.26). This is expected and
  harmless — all scripts here work with numpy 1.26.
- **MPI:** the conda-forge `mpi4py` pulls in its own MPI (MPICH by default).
  On a cluster, use the system MPI instead: load the MPI module, skip `mpi4py`
  in the `conda create` line, and build it against the system stack with
  `pip install --no-binary mpi4py mpi4py`.
- **postcactus on newer Python:** PostCactus is pure Python and installs fine
  alongside kuibit. If it ever fails on a new Python version, fall back to the
  two-environment setup described at the end of `Resample/README.md`
  (postcactus in a Python 3.8 env, kuibit in a Python ≥ 3.9 env) — the two
  resampling backends produce identical HDF5 output, so everything downstream
  works either way.
- `Frames/make_frames.py` falls back to serial execution when `mpi4py` is not
  importable, so it also runs on machines without MPI.
