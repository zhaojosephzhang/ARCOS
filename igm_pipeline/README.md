# IGM/CGM Evolution Pipeline

This documentation corresponds to the current implementation:

- `f_igm_z_halo_paralell_MPI_type_formal_FoF.py`
- `run_IGM_z_evolve_MPI_type_formal_FoF.sh`

---

## Overview

This pipeline is designed to:

- compute the redshift evolution of `f_IGM` and `f_CGM`
- measure mass fractions of different components, including `gas`, `star`, and `BH`
- classify gas into different phases:
  - cold-dense
  - cold-diffuse
  - hot-dense
  - hot-diffuse
- compare results between the **NoBH** and **Fiducial** simulations

By default, this script uses `R200` and `M200` values provided in the FoF halo catalog.

The code includes an optional module to recompute `R200/M200`, controlled by the internal flag `if_Cal_R200`.  
However, in the current command-line interface, this option is fixed to `False`.

---

## MPI Layout

- `rank 0` processes the **NoBH** simulation
- `rank 1` processes the **Fiducial** simulation
- at least **2 MPI ranks** are required

---

## Command Line

Command format:

```bash
mpirun -np 2 python f_igm_z_halo_paralell_MPI_type_formal_FoF.py [C_R200] [logM_low] [logM_up] [logT] [logrho]

Parameters

| Parameter  | Type    | Default | Description                                                                               |
| ---------- | ------- | ------: | ----------------------------------------------------------------------------------------- |
| `C_R200`   | `int`   |     `1` | Radius multiplier in units of `R200`. For example, `1` means within `1 × R200`.           |
| `logM_low` | `float` |     `0` | Lower halo mass limit in `log10(M200 / Msun)`.                                            |
| `logM_up`  | `float` |  `15.5` | Upper halo mass limit in `log10(M200 / Msun)`.                                            |
| `logT`     | `float` |   `4.0` | Temperature threshold for gas phase classification.                                       |
| `logrho`   | `float` |   `3.5` | Density threshold (logarithmic), defined relative to the mean density (`rho / rho_mean`). |


The simplest example：

```bash
mpirun -np 2 python f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 10 15.5 4.0 3.5
```

## Batch Script

Submit the PBS job using:

```bash
qsub run_IGM_z_evolve_MPI_type_formal_FoF.sh

The script internally runs:

```bash
mpirun ${NQSV_MPIOPTS} -genv I_MPI_DEBUG 5 -np 2 -genv I_MPI_PIN_DOMAIN=node \
  python3 $PBS_O_WORKDIR/f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 10 15.5 4.0 3.5
```

## Outputs

This pipeline produces the following quantities:

- `f_IGM` (as a function of redshift)
- `f_CGM` (as a function of redshift)
- `f_star`
- `f_BH`
- `f_cold_dense`
- `f_cold_diff`
- `f_hot_dense`
- `f_hot_diff`

## Output Location
Results are written to:
- storage_path
- kdtree_output_path

## File Naming Convention
Output file names include:
- `dataset label: n (NoBH) or f (Fiducial)`
- `C_R200`
- `halo mass range`
- `logT`
- `logrho`

## Notes

- This FoF-based version directly uses`R200` values provided in the halo catalog
- The option to “recompute R200 exists internally”, but is currently disabled in the command-line interface
- Future versions may expose this option as a dedicated command-line parameter
- Both intermediate (temporary) files and final processed outputs are generated
