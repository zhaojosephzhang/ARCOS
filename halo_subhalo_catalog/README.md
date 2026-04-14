# Halo And Subhalo Catalog Extraction

This directory contains a utility script for extracting physical halo and subhalo catalogs from Gadget-style `fof_subhalo_tab` outputs.

## Main Script

`FoF_halo_catalog_extraction.py`

The script currently supports both:

- FoF halo catalog extraction
- Subhalo catalog extraction

## What The Script Does

The script reads:

- snapshot header information from `snapshot_XXX(.i).hdf5`
- group and subhalo data from `fof_subhalo_tab_XXX(.i).hdf5`

It restores physical units using the scale factor `a` and Hubble parameter `h`, and writes two output catalogs:

- `halo_catalog_physical_020.h5`
- `subhalo_catalog_physical_020.h5`

## Inputs

The script currently uses hard-coded paths in the `__main__` block:

```python
snapshot_base = "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v2/output/snapdir_020/snapshot_020"
group_base = "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v2/output/groups_020/fof_subhalo_tab_020"
```

Meaning:

- `snapshot_base`: snapshot file prefix used to read `Time` and `HubbleParam`
- `group_base`: FoF/subhalo catalog file prefix used to read halo and subhalo datasets

The script automatically supports either:

- single-file catalogs: `xxx.hdf5`
- multi-file catalogs: `xxx.0.hdf5`, `xxx.1.hdf5`, ...

## Halo Catalog Output

The halo catalog contains:

- `GroupPos`
- `GroupMass`
- `M200`
- `R200`
- `GroupGasMass`
- `GroupDMMass`
- `GroupStellarMass`
- `GroupBHMass`

## Subhalo Catalog Output

The subhalo catalog contains:

- `SubhaloPos`
- `SubhaloMass`
- `SubhaloGroupNr`
- `SubhaloGasMass`
- `SubhaloDMMass`
- `SubhaloStellarMass`
- `SubhaloBHMass`

## Mass Components

Mass components are extracted from `MassType[:, i]` using the usual Gadget ordering:

- `0`: gas
- `1`: dark matter
- `4`: stars
- `5`: black holes

Applied datasets:

- `GroupMassType` for halo component masses
- `SubhaloMassType` for subhalo component masses

## Units

The output catalogs are stored in physical units:

- mass unit: `1e10 M_sun`
- length unit: `Mpc`

These units are also written into the output HDF5 attributes.

## Running

Run directly with:

```bash
python FoF_halo_catalog_extraction.py
```

or through the existing batch script:

```bash
bash joblist_Halo_extraction.sh
```

## Validation Script

The extracted catalogs can be checked with:

`plot_halo_catalog_validation.py`

This validator supports both halo and subhalo catalogs and can generate:

- 3D position distribution plots
- mass-bin count distributions
- median radius trends for halo catalogs
- mass-fraction evolution plots for gas, DM, stellar, and BH components

## Notes

- The current extraction script keeps the original halo functionality and extends it with subhalo support.
- For subhalos, the stellar mass is taken from `SubhaloMassType[:, 4]`.
- For halos, the component masses are taken from `GroupMassType`.
