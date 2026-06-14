# AGORA z0 DM/EM Analysis Pipeline

This directory contains a multi-code AGORA analysis workflow for Milky-Way-like
galaxy snapshots at the `z0` release epoch.  The main pipeline computes
dispersion measure (DM), emission measure (EM), hot-gas EM, projection maps,
Mollweide sky maps, and random-observer special sightlines using a common
observer geometry across several AGORA simulation codes.

The repository is intended to support scientific comparison of simulated
electron columns and gas structure across hydrodynamic solvers.

## Scientific Purpose

The pipeline is designed to answer questions such as:

- How different AGORA codes predict Milky-Way-like DM and EM distributions.
- How much of the total DM/EM comes from the ISM, CGM, and hot gas.
- How gas morphology, stellar morphology, temperature structure, and velocity
  fields differ between codes.
- How the predicted sky maps depend on observer position and disk orientation.
- How the simulated electron-column distributions compare between full-sky
  sightlines and selected special directions.

## Main Files

| File | Purpose |
|---|---|
| `AGORA_parallel_DM_EM_projection_pipeline.py` | Main command-line pipeline for one snapshot. |
| `run_all_agora_codes_20k.sh` | Batch runner for all supported z0 datasets. |
| `AGORA_z0_run_pipeline_only.ipynb` | Lightweight notebook for launching runs only. |
| `AGORA_z0_results_viewer_only.ipynb` | Viewer notebook that reads existing `parallel_outputs/` products. |
| `AGORA_z0_adaptive_DM_EM_analysis.ipynb` | Full analysis notebook with dataset discovery, single-code running, and extra plotting panels. |
| `AGORA_EMDM_debugging.ipynb` | Debugging notebook for field, unit, geometry, and projection checks. |
| `inspect_agora_z0.ipynb` | Dataset inspection notebook. |
| `agora_z0_dataset_inspection.csv` | Tabular summary of detected datasets and available fields. |
| `agora_z0_dataset_inspection.json` | JSON version of the dataset inspection summary. |
| `AGORA_current_parameter_table.tex` | LaTeX table of current run parameters. |
| `agora_z0_dataset_inspection_table.tex` | LaTeX table of dataset inspection results. |

## Supported Codes

The current z0 workflow supports:

- `ARTI`
- `Enzo`
- `AREPO`
- `GADGET3`
- `GEAR`
- `G4Cal_Pablo`
- `CHANGA`

The batch script expects snapshots in the following default locations:

```text
ARTI/10MpcBox_csf512_04078.d
Enzo/RD0347/RD0347
AREPO/snap_336.hdf5
GADGET3/snapshot_304/snapshot_304.0.hdf5
GEAR/snapshot_0845.hdf5
CHANGA/ncal-IV.003524
G4Cal_Pablo/snapshot_034.hdf5
```

## Dependencies

The code uses Python 3 with:

- `numpy`
- `scipy`
- `matplotlib`
- `yt`
- `joblib`
- `h5py`
- `pandas` for notebook tables

The local runs have used:

```bash
/home/zhaozhang/anaconda3/bin/python
```

## Running One Dataset

Example for AREPO:

```bash
python AGORA_parallel_DM_EM_projection_pipeline.py \
  --snapshot /home/zhaozhang/local/AGORA_work/AGORA_Data/z0/AREPO/snap_336.hdf5 \
  --code AREPO \
  --outdir /home/zhaozhang/local/AGORA_work/AGORA_Data/z0/parallel_outputs/AREPO \
  --ionization-mode auto \
  --n-los 20000 \
  --n-jobs 30 \
  --random-observers 128 \
  --s-max-kpc 100 \
  --ds-kpc 0.25 \
  --R-sun-kpc 8.2 \
  --velocity-reference-source gas \
  --velocity-reference-radius-kpc 30 \
  --ism-R-kpc 20 \
  --ism-abs-z-kpc 5 \
  --hot-Tmin-K 1e6 \
  --make-projections \
  --projection-box-kpc 40 \
  --projection-npix 512 \
  --projection-max-elements 100000000 \
  --projection-kernel-max-particles 500000 \
  --projection-quiver-step 12 \
  --projection-quiver-scale 2200 \
  --projection-quiver-width 0.0022 \
  --projection-quiver-alpha 0.75 \
  --make-mollweide \
  --make-hot-em-diagnostics
```

To run in the background:

```bash
nohup python AGORA_parallel_DM_EM_projection_pipeline.py \
  --snapshot /path/to/snapshot \
  --code AREPO \
  --outdir parallel_outputs/AREPO \
  --ionization-mode auto \
  --n-los 20000 \
  --n-jobs 30 \
  --random-observers 128 \
  --s-max-kpc 100 \
  --ds-kpc 0.25 \
  --R-sun-kpc 8.2 \
  --velocity-reference-source gas \
  --velocity-reference-radius-kpc 30 \
  --ism-R-kpc 20 \
  --ism-abs-z-kpc 5 \
  --hot-Tmin-K 1e6 \
  --make-projections \
  --projection-box-kpc 40 \
  --projection-npix 512 \
  --projection-max-elements 100000000 \
  --projection-kernel-max-particles 500000 \
  --projection-quiver-step 12 \
  --make-mollweide \
  --make-hot-em-diagnostics \
  > single_logs/AREPO_20k_pipeline.log 2>&1 &
```

## Running All Codes

Run all supported datasets:

```bash
bash run_all_agora_codes_20k.sh
```

Run selected datasets:

```bash
bash run_all_agora_codes_20k.sh ARTI AREPO GEAR
```

Detached background run:

```bash
setsid -f /bin/bash -c \
'WAIT_FOR_PID= /bin/bash run_all_agora_codes_20k.sh \
> batch_logs_20k/nohup_all_codes_20k.log 2>&1 < /dev/null'
```

Batch status and logs are written to:

```text
batch_logs_20k/batch_status.log
batch_logs_20k/<CODE>.log
```

## Current Batch Defaults

These defaults are synchronized with the analysis notebooks:

| Parameter | Value | Meaning |
|---|---:|---|
| `N_LOS` | `20000` | Number of all-sky sightlines. |
| `N_JOBS` | `30` | joblib worker count. |
| `RANDOM_OBSERVERS` | `128` | Random Solar azimuths for special LOS tests. |
| `S_MAX_KPC` | `100` | Maximum LOS integration distance. |
| `DS_KPC` | `0.25` | LOS integration step. |
| `R_SUN_KPC` | `8.2` | Solar radius in the face-on frame. |
| `VELOCITY_REFERENCE_SOURCE` | `gas` | Bulk velocity reference for velocity maps. |
| `VELOCITY_REFERENCE_RADIUS_KPC` | `30.0` | Radius for bulk velocity subtraction. |
| `ISM_R_KPC` | `20.0` | ISM cylindrical radius. |
| `ISM_ABS_Z_KPC` | `5.0` | ISM half-height. |
| `HOT_TMIN_K` | `1.0e6` | Hot-gas temperature threshold. |
| `PROJECTION_BOX_KPC` | `40` | Projection width, normally `[-20,20] kpc`. |
| `PROJECTION_NPIX` | `512` | Projection pixels per side. |
| `PROJECTION_MAX_ELEMENTS` | `100000000` | Maximum loaded elements for projection. |
| `PROJECTION_KERNEL_MAX_PARTICLES` | `500000` | Maximum gas particles deposited with kernel per map. |
| `PROJECTION_QUIVER_STEP` | `12` | Velocity quiver stride. |
| `PROJECTION_LOS_HALF_THICKNESS_KPC` | `5.0` | Projection half-thickness; empty means full-column projection. |
| `UNIT_BASE` | `auto` | Unit handling for HDF5 snapshots. |
| `INTEGRATION_BACKEND` | `auto` | Particle or grid-ray backend selection. |
| `PARTICLE_INTERPOLATION` | `auto` | M6 kernel when available, otherwise inverse-distance. |
| `CENTER_MODE` | `stellar_com` | Galaxy center estimate. |
| `DISK_NORMAL_SOURCE` | `stars` | Disk-normal angular momentum source. |
| `IONIZATION_MODE` | `auto` | Prefer real electron/ion fields, then temperature-based fallback. |

## Important Pipeline Arguments

### Dataset and Units

| Argument | Description |
|---|---|
| `--snapshot` | Path to the snapshot or dataset entry point. |
| `--code` | Code label, e.g. `ARTI`, `ENZO`, `AREPO`, `GADGET-3`, `GEAR`, `CHANGA`. |
| `--outdir` | Output directory. |
| `--unit-base` | `auto`, `none`, `agora_gadget`, or `agora_arepo`. |
| `--tipsy-length-unit-kpc` | Explicit Tipsy length code unit in physical kpc. Needed for CHANGA if no parameter file is available. |
| `--tipsy-mass-unit-msun` | Explicit Tipsy mass code unit in Msun. Needed for CHANGA if no parameter file is available. |
| `--tipsy-time-unit-s` | Explicit Tipsy time code unit in seconds. Needed for physical Tipsy velocities. |

### Geometry

| Argument | Description |
|---|---|
| `--center-mode` | `stellar_com`, `gas_com`, `star_gas_com`, `gas_density_peak`, or `manual`. |
| `--center-kpc x y z` | Manual center in kpc. |
| `--disk-normal-source` | `stars`, `cold_gas`, `gas`, `star_gas`, or `manual`. |
| `--manual-disk-normal x y z` | Manual disk-normal vector. |
| `--angular-momentum-radius-kpc` | Radius for disk angular momentum. |
| `--velocity-reference-source` | `none`, `stars`, `gas`, or `star_gas`. |
| `--velocity-reference-radius-kpc` | Radius for bulk velocity subtraction. |

### LOS Integration

| Argument | Description |
|---|---|
| `--n-los` | Number of all-sky sightlines. |
| `--s-max-kpc` | Maximum path length. |
| `--ds-kpc` | Step size along each LOS. |
| `--n-jobs` | Number of parallel workers. |
| `--integration-backend` | `auto`, `particle`, or `gridray`. |
| `--particle-interpolation` | `auto`, `inverse_distance`, or `m6_kernel`. |
| `--max-kernel-radius-kpc` | Maximum particle search radius. |

### Gas Physics

| Argument | Description |
|---|---|
| `--ionization-mode` | `auto`, `fully_ionized`, `hydrogen_only`, `temperature_cut`, or `temperature_weighted`. |
| `--X-H`, `--Y-He` | Hydrogen and helium mass fractions. |
| `--ism-R-kpc` | ISM radius for component decomposition. |
| `--ism-abs-z-kpc` | ISM half-height for component decomposition. |
| `--cgm-inner-r-kpc` | CGM inner radius. |
| `--cgm-outer-r-kpc` | CGM outer radius. |
| `--hot-Tmin-K` | Hot-gas temperature threshold. |

### Projection Products

| Argument | Description |
|---|---|
| `--make-projections` | Write gas, stellar, temperature, and velocity projection products. |
| `--projection-box-kpc` | Full projected map width. |
| `--projection-npix` | Pixels per side. |
| `--projection-max-elements` | Maximum gas/star elements loaded for projection. |
| `--projection-los-half-thickness-kpc` | Optional projection half-thickness. Omit for full-column projection. |
| `--projection-kernel-max-particles` | Maximum gas particles deposited with M6 kernel per map. |
| `--projection-quiver-step` | Velocity quiver spacing. |
| `--projection-quiver-scale` | Quiver scale. |
| `--projection-quiver-width` | Quiver width. |
| `--projection-quiver-alpha` | Quiver transparency. |

### Extra Diagnostics

| Argument | Description |
|---|---|
| `--make-mollweide` | Write all-sky Mollweide maps for DM, EM, and hot EM. |
| `--make-hot-em-diagnostics` | Write hot EM versus Galactic-center angle and polar sky view. |
| `--random-observers` | Number of random observer azimuths for special LOS tests. |

## Output Structure

Each production run writes to:

```text
parallel_outputs/<CODE>/
```

Core LOS products:

| File | Description |
|---|---|
| `MWlike_4pi_DM_EM_sightlines.csv` | Main LOS table with DM, EM, component DM/EM, hot EM, and sky geometry. |
| `MWlike_4pi_DM_EM_sightlines.npz` | Same arrays in NumPy format. |
| `MWlike_4pi_DM_EM_summary.json` | Summary statistics. |
| `MWlike_4pi_DM_EM_metadata.json` | Run metadata, geometry, unit, field, and backend information. |
| `parallel_pipeline_metadata.json` | Top-level pipeline configuration and metadata. |

Projection products:

| File | Description |
|---|---|
| `projection_maps_face_edge.npz` | Reusable arrays for gas, stars, temperature, and velocity maps. |
| `projection_maps_face_edge_metadata.json` | Projection settings and method metadata. |
| `projection_gas_surface_density_face_edge.png` | Gas surface-density face-on/edge-on map. |
| `projection_stellar_surface_density_face_edge.png` | Stellar surface-density face-on/edge-on map. |
| `projection_gas_temperature_face_edge.png` | Gas temperature face-on/edge-on map. |
| `projection_gas_velocity_face_edge.png` | LOS velocity map with in-plane velocity quivers. |

All-sky and hot-gas products:

| File | Description |
|---|---|
| `MWlike_4pi_DM_total_pc_cm3_mollweide.png` | All-sky total DM map. |
| `MWlike_4pi_EM_ne2_total_pc_cm6_mollweide.png` | All-sky total EM map. |
| `MWlike_4pi_EM_ne2_hot_pc_cm6_mollweide.png` | All-sky hot EM map. |
| `MWlike_4pi_hot_EM_vs_GC_angle.png` | Hot EM versus angle from Galactic center. |
| `MWlike_4pi_hot_EM_GC_polar.png` | Hot EM polar sky view around the Galactic center. |

Random-observer products:

| File | Description |
|---|---|
| `random_observer_special_los_DM_EM.csv` | Toward-center, anti-center, and vertical LOS for random Solar azimuths. |
| `random_observer_special_los_summary.json` | Summary statistics for those special LOS tests. |

Viewer figures are saved under:

```text
parallel_outputs/viewer_figures/
```

Common viewer products include:

- `comparison_DM_EM_hotEM_median_p16_p84.png`
- `LOS_pdf_histograms_across_codes.png`
- `DM_pdf_parametric_fits.png`
- `across_codes_Gas_projection.png`
- `across_codes_Stellar_projection.png`
- `across_codes_Temperature_projection.png`
- `across_codes_Velocity_projection.png`
- `across_codes_nested_projection_atlas.png`
- `stellar_mock_observation_style_atlas.png`
- `<CODE>_mollweide_diagnostics.png`
- `<CODE>_hot_em_diagnostics.png`
- `<CODE>_DM_EM_component_distributions.png`

## Projection and Kernel Notes

For particle codes, the LOS backend uses M6 smoothing-length interpolation when
gas mass, density, and smoothing length are available.  Otherwise it falls back
to inverse-distance interpolation.

Projection maps use the following conventions:

- Gas surface density uses M6 smoothing-length deposition when possible.
- Velocity maps are direct mass-weighted particle/cell bin averages.  Velocity
  is a vector field and is not deposited as a conserved scalar.
- Temperature maps are mass-weighted maps.
- `--projection-los-half-thickness-kpc` controls slab projection thickness.  If
  omitted, the projection is full-column.

## Ionization Logic

With `--ionization-mode auto`, the code attempts to:

1. Use real electron/ion fields when available and usable.
2. Use a temperature-weighted ionization model when temperature exists and no
   reliable electron field is available.
3. Fall back to fully ionized gas when no better information exists.

The actual field or fallback used is recorded in the metadata JSON files.

## Dataset Inspection

The inspection files:

```text
agora_z0_dataset_inspection.csv
agora_z0_dataset_inspection.json
```

summarize the detected snapshots, yt dataset classes, units, particle types,
field availability, gas/star position and velocity fields, and recommended
backends.  These files are useful before production runs and before debugging a
new dataset.

## Known Caveats

### CHANGA / Tipsy Units

CHANGA is a Tipsy dataset.  A yt-reported value such as:

```text
length_unit = 1.0 kpc
mass_unit   = 1.0 Msun
```

does not necessarily prove that the raw Tipsy code units are physically correct.
Current CHANGA tests show an unphysical domain width of order `10^6-10^7 kpc`,
so CHANGA production results should not be treated as final until the correct
Tipsy `dKpcUnit`, `dMsolUnit`, and optional time unit are supplied.

The local file:

```text
CHANGA/ncal-IV.003524-hsml
```

is a yt-generated smoothing-length sidecar, not an original AGORA download.
The current reader skips two leading float values and then reads one smoothing
length per gas particle.

### AREPO Smoothing Length

AREPO contains both `PartType0/SubfindHsml` and an external sidecar:

```text
AREPO/snap_336.hsml.hdf5:/PartType0/SmoothingLength
```

For gas projection and interpolation, the external sidecar smoothing length is
preferred.

### Debug Outputs

Directories under `parallel_outputs/` with names containing `_smoke`, `_debug`,
or `_fix` are development/test products.  The viewer notebook filters these out
for production comparison plots when `is_production_output_dir(...)` is
available.

## Recommended GitHub Layout

Raw AGORA snapshots and generated outputs are large and should usually not be
committed directly.  For a public GitHub repository, consider excluding:

```text
ARTI/
AREPO/
CHANGA/
Enzo/
G4Cal_Pablo/
GADGET3/
GEAR/
parallel_outputs/
batch_logs_20k/
single_logs/
__pycache__/
*.ewah
*.kdtree
*.hdf5
*.DAT
```

Keep the pipeline, notebooks, README, small inspection summaries, and LaTeX
tables in Git.  Store raw data and heavy outputs on a data server, Git LFS,
Zenodo, or another archival storage service.

