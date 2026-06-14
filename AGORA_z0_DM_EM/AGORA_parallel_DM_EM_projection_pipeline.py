#!/usr/bin/env python3
"""
MWlike_3D_DM_EM_sightlines_AGORA_multicode.py

Multi-code AGORA framework for MW-like 3D DM/EM sightline analysis.

What this script is designed to do
----------------------------------
1. Load AGORA yt-readable datasets from different codes.
2. Use code-specific field adapters for particle/SPH, moving-mesh, and grid/AMR data.
3. Determine the galaxy center.
4. Compute the disk angular-momentum direction from inner stars or cold gas.
5. Define a face-on disk coordinate system without collapsing the 3D data.
6. Place a Solar-like observer in the face-on frame.
7. Shoot 4pi LOS directions and compute DM/EM with ISM/CGM/hot-gas decomposition.

Supported code families
-----------------------
SPH / particle-like backend:
    GADGET-3, GADGET-4, GEAR, GIZMO, CHANGA, GASOLINE
    Uses gas element positions + KDTree interpolation by default. Internally these
    are treated by backend family rather than by exact code name.

Grid/AMR backend:
    ART-I, ENZO, RAMSES
    Uses yt arbitrary rays through the native mesh/cells.

Moving-mesh / quasi-particle backend:
    AREPO
    Default is particle-like KDTree interpolation because AGORA AREPO outputs are often
    handled through PartType0-style fields in yt. You can force grid rays with
    --integration-backend gridray if your yt frontend exposes fluid mesh fields.

Important scientific caveats
----------------------------
- The default particle backend uses --particle-interpolation auto. It uses the M6
  smoothing-length kernel when particle mass, density, and smoothing length exist,
  otherwise it falls back to k-nearest inverse-distance interpolation.
- Electron density is code/field dependent. The script prefers HII/HeII/HeIII
  ion fractions or number densities, then explicit electron number density.
  Placeholder abundance floors, common in some GADGET-family snapshots, are
  ignored. With --ionization-mode auto, missing/floor ionization information
  falls back to a temperature-weighted model when temperature exists, otherwise
  to a fully ionized model.
- For grid/AMR runs, yt ray integration is much closer to a native 3D LOS integral.

Example
-------
python MWlike_3D_DM_EM_sightlines_AGORA_multicode.py \
  --snapshot /path/to/snapshot_or_info_file \
  --code RAMSES \
  --integration-backend auto \
  --outdir ./MW_DM_EM_RAMSES \
  --n-los 4096 \
  --s-max-kpc 250 \
  --ds-kpc 0.25 \
  --R-sun-kpc 8.2 \
  --disk-normal-source stars

For GADGET-3/GADGET-4/GIZMO/GEAR and AREPO:
  keep the default --unit-base auto. It reads the HDF5 Units group when present
  before calling yt. Use --unit-base none only when you intentionally want yt's
  frontend defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple, List, Any

import numpy as np

plt = None
LogNorm = None
TwoSlopeNorm = None

def ensure_matplotlib():
    global plt, LogNorm, TwoSlopeNorm
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from matplotlib.colors import LogNorm as _LogNorm, TwoSlopeNorm as _TwoSlopeNorm
        plt = _plt
        LogNorm = _LogNorm
        TwoSlopeNorm = _TwoSlopeNorm
    return plt

try:
    import yt
    from yt.data_objects.particle_filters import add_particle_filter
except Exception:  # pragma: no cover
    yt = None
    add_particle_filter = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

# cgs constants
M_P_G = 1.67262192369e-24
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.98847e33

AGORA_GADGET_UNIT_BASE = {
    "UnitLength_in_cm": 3.08568e24,      # Mpc for the AGORA Gadget3 HDF5 snapshots inspected here
    "UnitMass_in_g": 1.989e43,           # 1e10 Msun
    "UnitVelocity_in_cm_per_s": 100000,  # km/s
}

AGORA_AREPO_UNIT_BASE = {
    "UnitLength_in_cm": 3.08568e24,      # Mpc
    "UnitMass_in_g": 1.989e43,
    "UnitVelocity_in_cm_per_s": 100000,
}

# -----------------------------------------------------------------------------
# AGORA code adapter configuration
# -----------------------------------------------------------------------------

AGORA_CODE_CONFIG: Dict[str, Dict[str, Any]] = {
    "ART-I": {
        "family": "grid",
        "gas_types": ["gas"],
        "star_types": ["stars", "star", "all"],
        "gas_mass_names": ["cell_mass", "mass"],
        "star_mass_names": ["particle_mass", "Mass", "Masses"],
        "density_names": ["density"],
        "temperature_names": ["temperature", "Temperature"],
        "velocity_names": [("velocity_x", "velocity_y", "velocity_z"), ("x-velocity", "y-velocity", "z-velocity")],
        "electron_names": ["El_number_density", "electron_number_density", "Electron_Number_Density"],
        "nh_names": ["H_number_density", "H_nuclei_density", "number_density"],
    },
    "RAMSES": {
        "family": "grid",
        "gas_types": ["gas"],
        "star_types": ["star", "Stars", "stars", "all"],
        "gas_mass_names": ["cell_mass", "mass"],
        "star_mass_names": ["particle_mass", "Mass", "Masses"],
        "density_names": ["density"],
        "temperature_names": ["temperature", "Temperature"],
        "velocity_names": [("velocity_x", "velocity_y", "velocity_z"), ("x-velocity", "y-velocity", "z-velocity")],
        "electron_names": ["electron_number_density", "El_number_density", "Electron_Number_Density"],
        "nh_names": ["H_number_density", "H_nuclei_density", "number_density"],
    },
    "ENZO": {
        "family": "grid",
        "gas_types": ["gas"],
        "star_types": ["Stars", "stars", "star", "all"],
        "gas_mass_names": ["cell_mass", "mass"],
        "star_mass_names": ["particle_mass", "Mass", "Masses"],
        "density_names": ["density"],
        "temperature_names": ["temperature", "Temperature"],
        "velocity_names": [("velocity_x", "velocity_y", "velocity_z"), ("x-velocity", "y-velocity", "z-velocity")],
        "electron_names": ["Electron_Density", "electron_density", "electron_number_density", "El_number_density"],
        "nh_names": ["H_number_density", "H_nuclei_density", "number_density"],
    },
    "GADGET-3": {
        "family": "particle",
        "gas_types": ["PartType0", "Gas", "gas"],
        "star_types": ["PartType4", "Stars", "stars"],
        "gas_mass_names": ["Masses", "Mass", "particle_mass"],
        "star_mass_names": ["Masses", "Mass", "particle_mass"],
        "density_names": ["Density", "density", "particle_density"],
        "temperature_names": ["Temperature", "temperature", "Temperature1", "GrackleTemperature"],
        "velocity_vector_names": ["particle_velocity", "Velocities"],
        "position_vector_names": ["particle_position", "Coordinates"],
        "smoothing_length_names": ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"],
        "electron_names": ["electron_number_density", "Electron_Density", "electron_density", "El_number_density", "ElectronAbundance", "electron_abundance"],
        "hii_names": ["HII", "H_p1_fraction", "HII_Fraction", "HydrogenIonizedFraction"],
        "heii_names": ["HeII", "He_p1_fraction", "HeII_Fraction"],
        "heiii_names": ["HeIII", "He_p2_fraction", "HeIII_Fraction"],
        "nh_names": ["H_number_density", "NeutralHydrogenAbundance"],
    },
    "GADGET-4": {
        # GADGET-4 is treated as the same SPH/particle backend as GADGET-3 for
        # this LOS analysis.  The exact HDF5/yt field names may still vary
        # between datasets, so keep multiple aliases below.
        "family": "particle",
        "gas_types": ["PartType0", "Gas", "gas"],
        "star_types": ["PartType4", "Stars", "stars"],
        "gas_mass_names": ["Masses", "Mass", "particle_mass"],
        "star_mass_names": ["Masses", "Mass", "particle_mass"],
        "density_names": ["Density", "density", "particle_density"],
        "temperature_names": ["Temperature", "temperature", "Temperature1", "GrackleTemperature"],
        "velocity_vector_names": ["particle_velocity", "Velocities"],
        "position_vector_names": ["particle_position", "Coordinates"],
        "smoothing_length_names": ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"],
        "electron_names": ["electron_number_density", "Electron_Density", "electron_density", "El_number_density", "ElectronAbundance", "electron_abundance"],
        "hii_names": ["HII", "H_p1_fraction", "HII_Fraction", "HydrogenIonizedFraction"],
        "heii_names": ["HeII", "He_p1_fraction", "HeII_Fraction"],
        "heiii_names": ["HeIII", "He_p2_fraction", "HeIII_Fraction"],
        "nh_names": ["H_number_density", "NeutralHydrogenAbundance"],
    },
    "GEAR": {
        "family": "particle",
        "gas_types": ["PartType0", "Gas", "gas"],
        "star_types": ["PartType1", "PartType4", "Stars", "stars"],
        "gas_mass_names": ["Masses", "Mass", "particle_mass"],
        "star_mass_names": ["Masses", "Mass", "particle_mass"],
        "density_names": ["Density", "density", "particle_density"],
        "temperature_names": ["Temperature", "temperature", "GrackleTemperature"],
        "velocity_vector_names": ["particle_velocity", "Velocities"],
        "position_vector_names": ["particle_position", "Coordinates"],
        "smoothing_length_names": ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"],
        "electron_names": ["electron_number_density", "Electron_Density", "electron_density", "El_number_density", "ElectronAbundance", "electron_abundance"],
        "hii_names": ["HII", "H_p1_fraction", "HII_Fraction", "HydrogenIonizedFraction"],
        "heii_names": ["HeII", "He_p1_fraction", "HeII_Fraction"],
        "heiii_names": ["HeIII", "He_p2_fraction", "HeIII_Fraction"],
        "nh_names": ["H_number_density", "NeutralHydrogenAbundance"],
    },
    "GIZMO": {
        "family": "particle",
        "gas_types": ["PartType0", "Gas", "gas"],
        "star_types": ["PartType4", "Stars", "stars"],
        "gas_mass_names": ["Masses", "Mass", "particle_mass"],
        "star_mass_names": ["Masses", "Mass", "particle_mass"],
        "density_names": ["Density", "density", "particle_density"],
        "temperature_names": ["Temperature", "temperature", "GrackleTemperature"],
        "velocity_vector_names": ["particle_velocity", "Velocities"],
        "position_vector_names": ["particle_position", "Coordinates"],
        "smoothing_length_names": ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"],
        "electron_names": ["electron_number_density", "Electron_Density", "electron_density", "El_number_density", "ElectronAbundance", "electron_abundance"],
        "hii_names": ["HII", "H_p1_fraction", "HII_Fraction", "HydrogenIonizedFraction"],
        "heii_names": ["HeII", "He_p1_fraction", "HeII_Fraction"],
        "heiii_names": ["HeIII", "He_p2_fraction", "HeIII_Fraction"],
        "nh_names": ["H_number_density", "NeutralHydrogenAbundance"],
    },
    "CHANGA": {
        "family": "particle",
        "gas_types": ["Gas", "gas", "PartType0"],
        "star_types": ["stars", "Stars", "PartType4", "all"],
        "gas_mass_names": ["Mass", "Masses", "particle_mass"],
        "star_mass_names": ["Mass", "Masses", "particle_mass"],
        "density_names": ["density", "Density", "particle_density"],
        "temperature_names": ["temperature", "Temperature", "GrackleTemperature"],
        "velocity_vector_names": ["particle_velocity", "Velocities"],
        "position_vector_names": ["particle_position", "Coordinates"],
        "smoothing_length_names": ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"],
        "electron_names": ["electron_number_density", "Electron_Density", "electron_density", "El_number_density", "ElectronAbundance", "electron_abundance"],
        "hii_names": ["HII", "H_p1_fraction", "HII_Fraction", "HydrogenIonizedFraction"],
        "heii_names": ["HeII", "He_p1_fraction", "HeII_Fraction"],
        "heiii_names": ["HeIII", "He_p2_fraction", "HeIII_Fraction"],
        "nh_names": ["H_number_density", "NeutralHydrogenAbundance"],
    },
    "GASOLINE": {
        "family": "particle",
        "gas_types": ["Gas", "gas", "PartType0"],
        "star_types": ["stars", "Stars", "PartType4", "all"],
        "gas_mass_names": ["Mass", "Masses", "particle_mass"],
        "star_mass_names": ["Mass", "Masses", "particle_mass"],
        "density_names": ["density", "Density", "particle_density"],
        "temperature_names": ["temperature", "Temperature", "GrackleTemperature"],
        "velocity_vector_names": ["particle_velocity", "Velocities"],
        "position_vector_names": ["particle_position", "Coordinates"],
        "smoothing_length_names": ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"],
        "electron_names": ["electron_number_density", "Electron_Density", "electron_density", "El_number_density", "ElectronAbundance", "electron_abundance"],
        "hii_names": ["HII", "H_p1_fraction", "HII_Fraction", "HydrogenIonizedFraction"],
        "heii_names": ["HeII", "He_p1_fraction", "HeII_Fraction"],
        "heiii_names": ["HeIII", "He_p2_fraction", "HeIII_Fraction"],
        "nh_names": ["H_number_density", "NeutralHydrogenAbundance"],
    },
    "AREPO": {
        "family": "moving_mesh",
        "gas_types": ["PartType0", "Gas", "gas"],
        "star_types": ["PartType4", "Stars", "stars"],
        "gas_mass_names": ["Masses", "Mass", "particle_mass"],
        "star_mass_names": ["Masses", "Mass", "particle_mass"],
        "density_names": ["Density", "density", "particle_density"],
        "temperature_names": ["Temperature", "temperature", "GrackleTemperature"],
        "velocity_vector_names": ["particle_velocity", "Velocities"],
        "position_vector_names": ["particle_position", "Coordinates"],
        "smoothing_length_names": ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"],
        "electron_names": ["electron_number_density", "Electron_Density", "electron_density", "El_number_density", "ElectronAbundance", "electron_abundance"],
        "hii_names": ["HII", "H_p1_fraction", "HII_Fraction", "HydrogenIonizedFraction"],
        "heii_names": ["HeII", "He_p1_fraction", "HeII_Fraction"],
        "heiii_names": ["HeIII", "He_p2_fraction", "HeIII_Fraction"],
        "nh_names": ["H_number_density", "NeutralHydrogenAbundance"],
    },
}

# Backend grouping.  The analysis logic should depend on the data structure
# rather than on the exact simulation-code name.
GRID_CODES = {k for k, v in AGORA_CODE_CONFIG.items() if v["family"] == "grid"}
PARTICLE_CODES = {k for k, v in AGORA_CODE_CONFIG.items() if v["family"] in ("particle", "moving_mesh")}
SPH_CODES = {k for k, v in AGORA_CODE_CONFIG.items() if v["family"] == "particle"}
MOVING_MESH_CODES = {k for k, v in AGORA_CODE_CONFIG.items() if v["family"] == "moving_mesh"}


def code_backend_family(code_cfg: Dict[str, Any]) -> str:
    """Return a physics/data-structure family label used for reporting."""
    fam = code_cfg.get("family", "particle")
    if fam == "grid":
        return "AMR/grid"
    if fam == "moving_mesh":
        return "moving-mesh"
    if fam == "particle":
        return "SPH/particle"
    return str(fam)


def normalize_code(code: str) -> str:
    if code is None or code.lower() == "auto":
        return "AUTO"
    c = code.upper().replace("_", "-")
    aliases = {
        # Keep explicit major versions.  Bare "GADGET" is mapped to GADGET-4
        # because the current CROCODILE/Osaka-style analysis is GADGET-4 based.
        # Use --code GADGET-3 explicitly for old AGORA comparison runs.
        "GADGET": "GADGET-4",
        "GADGET4": "GADGET-4",
        "GADGET-4": "GADGET-4",
        "G4CAL": "GADGET-4",
        "G4CAL_PABLO": "GADGET-4",
        "G4CAL-PABLO": "GADGET-4",
        "GADGET3": "GADGET-3",
        "GADGET-3": "GADGET-3",
        "ART": "ART-I",
        "ARTI": "ART-I",
        "ART-I": "ART-I",
        "CHANGA": "CHANGA",
        "CHARANGA": "CHANGA",
        "GIZMO": "GIZMO",
        "GEAR": "GEAR",
        "RAMSES": "RAMSES",
        "ENZO": "ENZO",
        "AREPO": "AREPO",
        "GASOLINE": "GASOLINE",
    }
    return aliases.get(c, c)




def infer_code_from_snapshot_path(path: str) -> str:
    """Infer AGORA code from common data-folder names when --code auto is used."""
    parts = [p.upper().replace('-', '_') for p in Path(path).expanduser().parts]
    path_text = str(path).upper().replace('-', '_')
    folder_map = {
        "ARTI": "ART-I",
        "ART_I": "ART-I",
        "ENZO": "ENZO",
        "AREPO": "AREPO",
        "GADGET3": "GADGET-3",
        "GADGET_3": "GADGET-3",
        "GEAR": "GEAR",
        "CHANGA": "CHANGA",
        "G4CAL_PABLO": "GADGET-4",
        "G4CAL": "GADGET-4",
        "GADGET4": "GADGET-4",
        "GADGET_4": "GADGET-4",
    }
    for key, value in folder_map.items():
        if key in parts or f"/{key}/" in path_text:
            return value
    return "AUTO"


def default_config_for_unknown(code: str, ds) -> Dict[str, Any]:
    # Fallback: infer family from particle types. If no particle gas type exists, use grid.
    ptypes = set(getattr(ds, "particle_types", []))
    if {"PartType0", "Gas", "gas"}.intersection(ptypes):
        base = dict(AGORA_CODE_CONFIG["GIZMO"])
        base["family"] = "particle"
        return base
    return dict(AGORA_CODE_CONFIG["RAMSES"])


@dataclass
class Config:
    snapshot: str
    code: str = "auto"
    outdir: str = "MW_DM_EM_output"
    unit_base: str = "auto"  # auto, none, agora_gadget, agora_arepo
    tipsy_length_unit_kpc: Optional[float] = None
    tipsy_mass_unit_msun: Optional[float] = None
    tipsy_time_unit_s: Optional[float] = None
    integration_backend: str = "auto"  # auto, particle, gridray

    gas_type: str = "auto"
    star_type: str = "auto"
    prefer_electron_field: bool = True

    center_mode: str = "stellar_com"  # stellar_com, gas_com, star_gas_com, gas_density_peak, manual
    center_kpc: Optional[Tuple[float, float, float]] = None
    center_radius_kpc: float = 30.0
    velocity_reference_source: str = "stars"  # none, stars, gas, star_gas
    velocity_reference_radius_kpc: float = 30.0

    disk_normal_source: str = "stars"  # stars, cold_gas, gas, star_gas, manual
    manual_disk_normal: Optional[Tuple[float, float, float]] = None
    angular_momentum_radius_kpc: float = 20.0
    cold_gas_Tmax_K: float = 3.0e4

    R_sun_kpc: float = 8.2
    phi_sun_deg: float = 0.0
    z_sun_kpc: float = 0.020

    n_los: int = 1024
    s_max_kpc: float = 250.0
    ds_kpc: float = 0.25
    chunk_los: int = 64

    # Particle backend interpolation
    n_ngb: int = 32
    max_kernel_radius_kpc: float = 5.0
    particle_interpolation: str = "auto"  # auto, inverse_distance, m6_kernel

    # Sampling for center/angle momentum if all_data is too large
    max_elements_for_geometry: int = 200_000_000
    seed: int = 12345

    X_H: float = 0.76
    Y_He: float = 0.24
    ionization_mode: str = "auto"  # auto, fully_ionized, hydrogen_only, temperature_cut, temperature_weighted
    ionized_Tmin_K: float = 1.0e4
    ionized_Tmid_K: float = 1.0e4
    ionized_logT_width: float = 0.25

    ism_R_kpc: float = 20.0
    ism_abs_z_kpc: float = 5.0
    cgm_inner_r_kpc: float = 20.0
    cgm_outer_r_kpc: float = 250.0
    hot_Tmin_K: float = 1.0e6

    random_rotate_directions: bool = False

    # Projection maps: half-thickness along the line of sight in the face-on frame.
    # None means full column projection.
    projection_los_half_thickness_kpc: Optional[float] = None


# -----------------------------------------------------------------------------
# yt field helpers
# -----------------------------------------------------------------------------

def unit_base_from_hdf5_units(path: str) -> Optional[Dict[str, float]]:
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(path, "r") as f:
            if "Units" not in f:
                return None
            attrs = f["Units"].attrs
            required = ["UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s"]
            if not all(k in attrs for k in required):
                return None
            return {k: float(attrs[k]) for k in required}
    except Exception:
        return None


def unit_base_from_hdf5_field_attrs(path: str) -> Optional[Dict[str, float]]:
    """Infer Gadget/Arepo unit_base from per-field HDF5 to_cgs attrs.

    Some AGORA Gadget-family snapshots do not have a top-level Units group, but
    their fields still carry conversion attrs. Without passing these to yt, yt
    assumes coordinates are kpc/h; for Mpc/h snapshots that shrinks positions by
    1000.
    """
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(path, "r") as f:
            if "PartType0" not in f:
                return None
            p0 = f["PartType0"]
            fields = {
                "UnitLength_in_cm": "Coordinates",
                "UnitMass_in_g": "Masses",
                "UnitVelocity_in_cm_per_s": "Velocities",
            }
            unit_base = {}
            for key, field_name in fields.items():
                if field_name not in p0 or "to_cgs" not in p0[field_name].attrs:
                    return None
                unit_base[key] = float(p0[field_name].attrs["to_cgs"])
            if any(v <= 0 for v in unit_base.values()):
                return None
            return unit_base
    except Exception:
        return None


def load_dataset(path: str, code: str = "auto", unit_base: str = "auto",
                 tipsy_length_unit_kpc: Optional[float] = None,
                 tipsy_mass_unit_msun: Optional[float] = None,
                 tipsy_time_unit_s: Optional[float] = None):
    if yt is None:
        raise ImportError("yt is not installed. Please run this script in your AGORA/yt environment.")
    kwargs = {}
    if unit_base == "auto":
        file_units = unit_base_from_hdf5_units(path)
        source = "HDF5 Units group"
        if file_units is None:
            file_units = unit_base_from_hdf5_field_attrs(path)
            source = "HDF5 field attrs"
        if file_units is not None:
            kwargs["unit_base"] = file_units
            print(f"Using {source} as yt unit_base:", file_units)
    elif unit_base == "agora_gadget":
        kwargs["unit_base"] = AGORA_GADGET_UNIT_BASE
    elif unit_base == "agora_arepo":
        kwargs["unit_base"] = AGORA_AREPO_UNIT_BASE
    elif unit_base == "none":
        pass
    else:
        raise ValueError(f"Unknown --unit-base: {unit_base}")
    tipsy_unit_base = {}
    if tipsy_length_unit_kpc is not None:
        tipsy_unit_base["length"] = (float(tipsy_length_unit_kpc), "kpc")
    if tipsy_mass_unit_msun is not None:
        tipsy_unit_base["mass"] = (float(tipsy_mass_unit_msun), "Msun")
    if tipsy_time_unit_s is not None:
        tipsy_unit_base["time"] = (float(tipsy_time_unit_s), "s")
    if tipsy_unit_base:
        kwargs["unit_base"] = tipsy_unit_base
        print("Using explicit Tipsy unit_base:", tipsy_unit_base)
    ds = yt.load(path, **kwargs)
    if normalize_code(code) == "AREPO" and hasattr(ds, "gen_hsmls"):
        # yt's Gadget frontend may try to generate/delete snap_*.hsml.hdf5
        # while merely building field_list. AREPO does not need this generated
        # SPH smoothing length for our density/cell-volume based analysis.
        ds.gen_hsmls = False
    print("Loaded dataset:", ds)
    print("Dataset type:", getattr(ds, "dataset_type", "unknown"))
    print("Particle types:", getattr(ds, "particle_types", None))
    return ds


def validate_dataset_units(ds, code: str, cfg: Config) -> None:
    normalized = normalize_code(code)
    if normalized != "CHANGA" or getattr(ds, "dataset_type", None) != "tipsy":
        return
    explicit_tipsy_units = (
        cfg.tipsy_length_unit_kpc is not None
        or cfg.tipsy_mass_unit_msun is not None
        or cfg.tipsy_time_unit_s is not None
    )
    length_unit = str(getattr(ds, "length_unit", ""))
    mass_unit = str(getattr(ds, "mass_unit", ""))
    has_param_units = any(k in getattr(ds, "parameters", {}) for k in ("dKpcUnit", "dMsolUnit"))
    if not explicit_tipsy_units and not has_param_units and length_unit == "1.0 kpc" and mass_unit == "1.0 Msun":
        raise RuntimeError(
            "CHANGA Tipsy snapshot was loaded with yt default units "
            "(length_unit=1 kpc, mass_unit=1 Msun). This makes positions, "
            "masses, DM, EM, and projections wrong. Provide the original "
            "Tipsy units with --tipsy-length-unit-kpc and --tipsy-mass-unit-msun "
            "(and --tipsy-time-unit-s if velocity units are needed), or supply a "
            "Tipsy parameter file/metadata before running production analysis."
        )


def field_exists(ds, field: Tuple[str, str]) -> bool:
    return field in ds.field_list or field in ds.derived_field_list


def first_existing_field(ds, ftypes: Sequence[str], names: Sequence[str]) -> Optional[Tuple[str, str]]:
    for ft in ftypes:
        for name in names:
            f = (ft, name)
            if field_exists(ds, f):
                return f
    return None


def choose_ftype(ds, user_value: str, candidates: Sequence[str], required_names: Sequence[str] = ()) -> Optional[str]:
    if user_value != "auto":
        return user_value
    for ft in candidates:
        if not required_names:
            if any(f[0] == ft for f in ds.field_list + ds.derived_field_list):
                return ft
        else:
            if any(field_exists(ds, (ft, n)) for n in required_names):
                return ft
    for ft in candidates:
        if ft in getattr(ds, "particle_types", []):
            return ft
    return None


def get_array(data, field: Tuple[str, str], unit: Optional[str] = None) -> np.ndarray:
    arr = data[field]
    if unit is not None:
        arr = arr.to(unit)
    return np.asarray(arr, dtype=np.float64)


def read_arepo_sidecar_smoothing_length(ds, ftype: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[str, str]]]:
    """Read AREPO snap_*.hsml.hdf5 smoothing lengths before SubfindHsml."""
    if ftype not in {"PartType0", "gas"}:
        return None, None
    filename = getattr(ds, "parameter_filename", None)
    if filename is None:
        return None, None
    snap_path = Path(filename)
    sidecar = snap_path.with_name(f"{snap_path.stem}.hsml.hdf5")
    if not sidecar.exists():
        return None, None
    try:
        import h5py
        with h5py.File(sidecar, "r") as f:
            key = "PartType0/SmoothingLength"
            if key not in f:
                return None, None
            hsml = np.asarray(f[key], dtype=np.float64)
        if hsml.size == 0:
            return None, None
        unit = getattr(ds, "length_unit", None)
        if unit is not None:
            factor = float(unit.to("kpc"))
            hsml = hsml * factor
        return hsml, ("PartType0", "SmoothingLength_sidecar")
    except Exception as exc:
        print(f"WARNING: cannot read AREPO sidecar smoothing length {sidecar}: {exc!r}")
        return None, None


def read_tipsy_sidecar_smoothing_length(ds, ftype: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[str, str]]]:
    """Read CHANGA/Tipsy *-hsml sidecar files when yt cannot load hsml.

    AGORA CHANGA distributes smoothing lengths as a companion binary file
    named like ``ncal-IV.003524-hsml``.  It is big-endian float32 and, for the
    z=0 file used here, contains two leading header values before one hsml per
    gas particle.
    """
    if getattr(ds, "dataset_type", None) != "tipsy" or ftype not in {"Gas", "gas"}:
        return None, None
    filename = getattr(ds, "parameter_filename", None)
    if filename is None:
        return None, None
    snap_path = Path(filename)
    candidates = [snap_path.with_name(f"{snap_path.name}-hsml"), snap_path.with_suffix(snap_path.suffix + ".hsml")]
    sidecar = next((path for path in candidates if path.exists()), None)
    if sidecar is None:
        return None, None
    try:
        hsml = np.fromfile(sidecar, dtype=">f4").astype(np.float64)
        gas_count = None
        counts = getattr(ds, "particle_type_counts", None)
        if isinstance(counts, dict):
            gas_count = counts.get("Gas") or counts.get("gas")
        if gas_count is not None:
            gas_count = int(gas_count)
            if hsml.size == gas_count + 2:
                hsml = hsml[2:]
            elif hsml.size != gas_count:
                print(f"WARNING: Tipsy hsml sidecar {sidecar} has {hsml.size} values; expected {gas_count} or {gas_count + 2}.")
                return None, None
        if hsml.size == 0 or not np.all(np.isfinite(hsml)) or np.any(hsml <= 0):
            print(f"WARNING: Tipsy hsml sidecar {sidecar} contains invalid smoothing lengths.")
            return None, None
        unit = getattr(ds, "length_unit", None)
        if unit is not None:
            hsml = hsml * float(unit.to("kpc"))
        return hsml, ("Gas", "smoothing_length_sidecar")
    except Exception as exc:
        print(f"WARNING: cannot read Tipsy sidecar smoothing length {sidecar}: {exc!r}")
        return None, None


def random_subsample(*arrays: Optional[np.ndarray], max_n: int, seed: int = 12345):
    base = next((a for a in arrays if a is not None), None)
    if base is None or len(base) <= max_n:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base), size=max_n, replace=False)
    return tuple(None if a is None else a[idx] for a in arrays)


def get_positions(data, ds, ftype: str) -> np.ndarray:
    xyz = [(ftype, "particle_position_x"), (ftype, "particle_position_y"), (ftype, "particle_position_z")]
    if all(field_exists(ds, f) for f in xyz):
        return np.vstack([get_array(data, f, "kpc") for f in xyz]).T
    for vname in ["particle_position", "Coordinates"]:
        f = (ftype, vname)
        if field_exists(ds, f):
            return get_array(data, f, "kpc")
    # Fluid/grid positions
    xyz = [(ftype, "x"), (ftype, "y"), (ftype, "z")]
    if all(field_exists(ds, f) for f in xyz):
        return np.vstack([get_array(data, f, "kpc") for f in xyz]).T
    xyz = [("index", "x"), ("index", "y"), ("index", "z")]
    if all(field_exists(ds, f) for f in xyz):
        return np.vstack([get_array(data, f, "kpc") for f in xyz]).T
    raise RuntimeError(f"Cannot find 3D positions for field/particle type {ftype}.")


def get_velocities(data, ds, ftype: str, code_cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    xyz = [(ftype, "particle_velocity_x"), (ftype, "particle_velocity_y"), (ftype, "particle_velocity_z")]
    if all(field_exists(ds, f) for f in xyz):
        return np.vstack([get_array(data, f, "km/s") for f in xyz]).T
    for vname in code_cfg.get("velocity_vector_names", ["particle_velocity", "Velocities"]):
        f = (ftype, vname)
        if field_exists(ds, f):
            return get_array(data, f, "km/s")
    for names in code_cfg.get("velocity_names", []):
        fields = [(ftype, names[0]), (ftype, names[1]), (ftype, names[2])]
        if all(field_exists(ds, f) for f in fields):
            return np.vstack([get_array(data, f, "km/s") for f in fields]).T
    # Generic gas velocity fallback
    for names in [("velocity_x", "velocity_y", "velocity_z"), ("x-velocity", "y-velocity", "z-velocity")]:
        fields = [(ftype, names[0]), (ftype, names[1]), (ftype, names[2])]
        if all(field_exists(ds, f) for f in fields):
            return np.vstack([get_array(data, f, "km/s") for f in fields]).T
    return None


def get_mass(data, ds, ftype: str, names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        f = (ftype, name)
        if field_exists(ds, f):
            return get_array(data, f, "Msun")
    return None


def enzo_stellar_particle_ftype(ds) -> Optional[str]:
    for ftype in ["io", "all", "nbody"]:
        if field_exists(ds, (ftype, "particle_type")) and all(
            field_exists(ds, (ftype, name))
            for name in ["particle_position_x", "particle_position_y", "particle_position_z", "particle_mass"]
        ):
            return ftype
    return None


def load_enzo_stellar_particle_arrays(data, ds, code_cfg: Dict[str, Any], max_n: int, seed: int):
    """Load Enzo star particles from the aggregate particle container.

    Enzo stores DM and star particles together in io/all.  In this AGORA
    snapshot, particle_type==2 has positive creation_time and is the stellar
    population; particle_type 1 and 4 are non-stellar components.
    """
    ftype = enzo_stellar_particle_ftype(ds)
    if ftype is None:
        return None, None, None, None
    ptype = np.asarray(data[(ftype, "particle_type")])
    mask = ptype == 2
    if not np.any(mask) and field_exists(ds, (ftype, "creation_time")):
        creation_time = np.asarray(data[(ftype, "creation_time")])
        mask = creation_time > 0
    n_star = int(np.count_nonzero(mask))
    if n_star == 0:
        return None, None, None, ftype
    idx = np.flatnonzero(mask)
    if idx.size > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_n, replace=False))
    pos = np.vstack([
        get_array(data, (ftype, "particle_position_x"), "kpc")[idx],
        get_array(data, (ftype, "particle_position_y"), "kpc")[idx],
        get_array(data, (ftype, "particle_position_z"), "kpc")[idx],
    ]).T
    vel = None
    vel_fields = [(ftype, "particle_velocity_x"), (ftype, "particle_velocity_y"), (ftype, "particle_velocity_z")]
    if all(field_exists(ds, f) for f in vel_fields):
        vel = np.vstack([get_array(data, f, "km/s")[idx] for f in vel_fields]).T
    mass = get_array(data, (ftype, "particle_mass"), "Msun")[idx]
    return pos, vel, mass, ftype


def get_smoothing_length(data, ds, ftype: str, names: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[Tuple[str, str]]]:
    sidecar_hsml, sidecar_field = read_arepo_sidecar_smoothing_length(ds, ftype)
    if sidecar_hsml is not None:
        return sidecar_hsml, sidecar_field
    sidecar_hsml, sidecar_field = read_tipsy_sidecar_smoothing_length(ds, ftype)
    if sidecar_hsml is not None:
        return sidecar_hsml, sidecar_field
    candidates = []
    for ft in [ftype, "gas"]:
        if ft is None:
            continue
        for name in names:
            f = (ft, name)
            if f not in candidates:
                candidates.append(f)
    for f in candidates:
        if not field_exists(ds, f):
            continue
        try:
            return get_array(data, f, "kpc"), f
        except Exception as exc_unit:
            try:
                return np.asarray(data[f], dtype=np.float64), f
            except Exception as exc_raw:
                print(f"WARNING: cannot read smoothing length field {f}; skipping it. "
                      f"unit error={exc_unit!r}; raw error={exc_raw!r}")
                continue
    return None, None


def get_density_cgs(data, ds, ftypes: Sequence[str], names: Sequence[str]) -> Tuple[np.ndarray, Tuple[str, str]]:
    f = first_existing_field(ds, ftypes, names)
    if f is None:
        raise RuntimeError(f"Cannot find density field. Tried ftypes={ftypes}, names={names}")
    return get_array(data, f, "g/cm**3"), f


def get_temperature(data, ds, ftypes: Sequence[str], names: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[Tuple[str, str]]]:
    f = first_existing_field(ds, ftypes, names)
    if f is None:
        return None, None
    try:
        return get_array(data, f, "K"), f
    except Exception:
        print(f"WARNING: reading {f} as raw temperature values; assuming K.")
        return np.asarray(data[f], dtype=np.float64), f


def get_scalar_field(data, ds, ftypes: Sequence[str], names: Sequence[str], unit: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[str, str]]]:
    f = first_existing_field(ds, ftypes, names)
    if f is None:
        return None, None
    try:
        return get_array(data, f, unit), f
    except Exception:
        return np.asarray(data[f], dtype=np.float64), f


def add_common_derived_fields(ds, code: str, code_cfg: Dict[str, Any]):
    """Add only safe, generic fields. We avoid overriding code-specific temperature unless necessary."""
    gas_types = code_cfg.get("gas_types", ["gas"])
    density_field = first_existing_field(ds, gas_types, code_cfg.get("density_names", ["density"]))
    if density_field is not None:
        out_field = (density_field[0], "density_squared")
        if not field_exists(ds, out_field):
            def _density_squared(field, data):
                return data[density_field] ** 2
            try:
                ds.add_field(out_field, function=_density_squared, sampling_type="local", units="g**2/cm**6")
            except Exception:
                try:
                    ds.add_field(out_field, function=_density_squared, particle_type=(code_cfg["family"] != "grid"), units="g**2/cm**6")
                except Exception as exc:
                    print("WARNING: could not add density_squared:", exc)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def weighted_mean(x: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if x is None or len(x) == 0:
        raise ValueError("Empty array in weighted_mean.")
    if w is None:
        return np.nanmean(x, axis=0)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(w) & np.all(np.isfinite(x), axis=1)
    if np.count_nonzero(ok) == 0 or np.nansum(w[ok]) <= 0:
        return np.nanmean(x, axis=0)
    return np.nansum(x[ok] * w[ok, None], axis=0) / np.nansum(w[ok])


def determine_center(cfg: Config, gas_pos: Optional[np.ndarray], gas_mass: Optional[np.ndarray], gas_rho: Optional[np.ndarray],
                     star_pos: Optional[np.ndarray], star_mass: Optional[np.ndarray]) -> np.ndarray:
    if cfg.center_mode == "manual":
        if cfg.center_kpc is None:
            raise ValueError("--center-mode manual requires --center-kpc x y z")
        return np.asarray(cfg.center_kpc, dtype=np.float64)

    if cfg.center_mode == "stellar_com" and star_pos is not None and len(star_pos) > 0:
        c0 = weighted_mean(star_pos, star_mass)
        r = np.linalg.norm(star_pos - c0[None, :], axis=1)
        m = r < cfg.center_radius_kpc
        if np.count_nonzero(m) >= 10:
            return weighted_mean(star_pos[m], None if star_mass is None else star_mass[m])
        return c0

    if cfg.center_mode == "gas_com" and gas_pos is not None and len(gas_pos) > 0:
        return weighted_mean(gas_pos, gas_mass)

    if cfg.center_mode == "star_gas_com":
        chunks = []
        weights = []
        if star_pos is not None and len(star_pos) > 0:
            chunks.append(star_pos)
            weights.append(np.ones(len(star_pos), dtype=np.float64) if star_mass is None else star_mass)
        if gas_pos is not None and len(gas_pos) > 0:
            chunks.append(gas_pos)
            weights.append(np.ones(len(gas_pos), dtype=np.float64) if gas_mass is None else gas_mass)
        if chunks:
            pos = np.vstack(chunks)
            mass = np.concatenate(weights)
            c0 = weighted_mean(pos, mass)
            r = np.linalg.norm(pos - c0[None, :], axis=1)
            m = r < cfg.center_radius_kpc
            if np.count_nonzero(m) >= 10:
                return weighted_mean(pos[m], mass[m])
            return c0

    if cfg.center_mode == "gas_density_peak" and gas_pos is not None and gas_rho is not None:
        return gas_pos[int(np.nanargmax(gas_rho))]

    if star_pos is not None and len(star_pos) > 0:
        return weighted_mean(star_pos, star_mass)
    if gas_pos is not None and len(gas_pos) > 0:
        return weighted_mean(gas_pos, gas_mass)
    raise RuntimeError("Cannot determine center: no valid gas or star positions.")


def compute_angular_momentum_normal(cfg: Config,
                                    gas_pos: Optional[np.ndarray], gas_vel: Optional[np.ndarray], gas_mass: Optional[np.ndarray], gas_temp: Optional[np.ndarray],
                                    star_pos: Optional[np.ndarray], star_vel: Optional[np.ndarray], star_mass: Optional[np.ndarray],
                                    center: np.ndarray) -> np.ndarray:
    if cfg.disk_normal_source == "manual":
        if cfg.manual_disk_normal is None:
            raise ValueError("--disk-normal-source manual requires --manual-disk-normal x y z")
        n = np.asarray(cfg.manual_disk_normal, dtype=np.float64)
        return n / np.linalg.norm(n)

    if cfg.disk_normal_source == "stars" and star_pos is not None and star_vel is not None and len(star_pos) > 0:
        pos, vel, mass = star_pos, star_vel, star_mass
        mask = np.linalg.norm(pos - center[None, :], axis=1) < cfg.angular_momentum_radius_kpc
    elif cfg.disk_normal_source in ("cold_gas", "gas") and gas_pos is not None and gas_vel is not None:
        pos, vel, mass = gas_pos, gas_vel, gas_mass
        mask = np.linalg.norm(pos - center[None, :], axis=1) < cfg.angular_momentum_radius_kpc
        if cfg.disk_normal_source == "cold_gas" and gas_temp is not None:
            mask &= gas_temp < cfg.cold_gas_Tmax_K
    elif cfg.disk_normal_source == "star_gas":
        pos_chunks = []
        vel_chunks = []
        mass_chunks = []
        if star_pos is not None and star_vel is not None and len(star_pos) > 0:
            m_star = np.linalg.norm(star_pos - center[None, :], axis=1) < cfg.angular_momentum_radius_kpc
            if np.count_nonzero(m_star) > 0:
                pos_chunks.append(star_pos[m_star])
                vel_chunks.append(star_vel[m_star])
                mass_chunks.append(np.ones(np.count_nonzero(m_star), dtype=np.float64) if star_mass is None else star_mass[m_star])
        if gas_pos is not None and gas_vel is not None and len(gas_pos) > 0:
            m_gas = np.linalg.norm(gas_pos - center[None, :], axis=1) < cfg.angular_momentum_radius_kpc
            if np.count_nonzero(m_gas) > 0:
                pos_chunks.append(gas_pos[m_gas])
                vel_chunks.append(gas_vel[m_gas])
                mass_chunks.append(np.ones(np.count_nonzero(m_gas), dtype=np.float64) if gas_mass is None else gas_mass[m_gas])
        if not pos_chunks:
            print("WARNING: Cannot compute star+gas angular momentum. Using z-axis.")
            return np.array([0.0, 0.0, 1.0])
        pos = np.vstack(pos_chunks)
        vel = np.vstack(vel_chunks)
        mass = np.concatenate(mass_chunks)
        mask = np.ones(len(pos), dtype=bool)
    else:
        print("WARNING: Cannot compute angular momentum from requested source. Using z-axis.")
        return np.array([0.0, 0.0, 1.0])

    if np.count_nonzero(mask) < 10:
        print("WARNING: Too few elements for angular momentum. Using z-axis.")
        return np.array([0.0, 0.0, 1.0])

    p = pos[mask] - center[None, :]
    v = vel[mask]
    w = np.ones(len(p)) if mass is None else mass[mask]
    v_bulk = weighted_mean(v, w)
    L = np.nansum(w[:, None] * np.cross(p, v - v_bulk[None, :]), axis=0)
    norm = np.linalg.norm(L)
    if not np.isfinite(norm) or norm == 0:
        print("WARNING: Bad angular momentum vector. Using z-axis.")
        return np.array([0.0, 0.0, 1.0])
    n = L / norm
    if n[2] < 0:
        n = -n
    return n


def compute_bulk_velocity(cfg: Config,
                          gas_pos: Optional[np.ndarray], gas_vel: Optional[np.ndarray], gas_mass: Optional[np.ndarray],
                          star_pos: Optional[np.ndarray], star_vel: Optional[np.ndarray], star_mass: Optional[np.ndarray],
                          center: np.ndarray) -> np.ndarray:
    """Mass-weighted reference velocity for subtracting halo/galaxy bulk motion."""
    if cfg.velocity_reference_source == "none":
        return np.zeros(3, dtype=np.float64)

    pos_chunks = []
    vel_chunks = []
    weight_chunks = []

    def add_component(pos, vel, mass):
        if pos is None or vel is None or len(pos) == 0:
            return
        mask = np.linalg.norm(pos - center[None, :], axis=1) < cfg.velocity_reference_radius_kpc
        if np.count_nonzero(mask) == 0:
            return
        pos_chunks.append(pos[mask])
        vel_chunks.append(vel[mask])
        weight_chunks.append(np.ones(np.count_nonzero(mask), dtype=np.float64) if mass is None else mass[mask])

    if cfg.velocity_reference_source in ("stars", "star_gas"):
        add_component(star_pos, star_vel, star_mass)
    if cfg.velocity_reference_source in ("gas", "star_gas"):
        add_component(gas_pos, gas_vel, gas_mass)

    # If the requested stellar reference is unavailable, fall back to gas rather
    # than leaving raw simulation-frame velocities in the projection maps.
    if not vel_chunks and cfg.velocity_reference_source == "stars":
        add_component(gas_pos, gas_vel, gas_mass)

    if not vel_chunks:
        print("WARNING: Cannot determine bulk velocity. Using zero velocity offset.")
        return np.zeros(3, dtype=np.float64)

    vel = np.vstack(vel_chunks)
    weights = np.concatenate(weight_chunks)
    return weighted_mean(vel, weights)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64); a /= np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64); b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, 1.0):
        return np.eye(3)
    if np.isclose(c, -1.0):
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis); v /= np.linalg.norm(v)
        K = skew(v)
        return np.eye(3) + 2.0 * (K @ K)
    s = np.linalg.norm(v)
    K = skew(v)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def observer_position_faceon(cfg: Config) -> np.ndarray:
    phi = np.deg2rad(cfg.phi_sun_deg)
    return np.array([cfg.R_sun_kpc * np.cos(phi), cfg.R_sun_kpc * np.sin(phi), cfg.z_sun_kpc], dtype=np.float64)


def original_to_faceon(pos: np.ndarray, center: np.ndarray, R_faceon: np.ndarray) -> np.ndarray:
    return (R_faceon @ (pos - center[None, :]).T).T


def faceon_to_original(pos_faceon: np.ndarray, center: np.ndarray, R_faceon: np.ndarray) -> np.ndarray:
    return (R_faceon.T @ pos_faceon.T).T + center[None, :]


def dataset_unit_metadata(ds) -> Dict[str, Any]:
    """Small unit audit block saved with outputs for cosmological snapshots."""
    meta = {
        "length_unit": str(getattr(ds, "length_unit", None)),
        "mass_unit": str(getattr(ds, "mass_unit", None)),
        "time_unit": str(getattr(ds, "time_unit", None)),
        "velocity_unit": str(getattr(ds, "velocity_unit", None)),
        "current_redshift": None,
        "scale_factor": None,
        "hubble_constant": None,
        "domain_width_kpc_physical": None,
    }
    z = getattr(ds, "current_redshift", None)
    if z is not None:
        try:
            z = float(z)
            meta["current_redshift"] = z
            meta["scale_factor"] = 1.0 / (1.0 + z)
        except Exception:
            meta["current_redshift"] = str(z)
    h = getattr(ds, "hubble_constant", None)
    if h is not None:
        try:
            meta["hubble_constant"] = float(h)
        except Exception:
            meta["hubble_constant"] = str(h)
    try:
        meta["domain_width_kpc_physical"] = np.asarray(ds.domain_width.to("kpc"), dtype=float).tolist()
    except Exception:
        pass
    return meta


# -----------------------------------------------------------------------------
# Density / ionization
# -----------------------------------------------------------------------------

def ne_nh_from_rho(cfg: Config, rho_cgs: np.ndarray, temp: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    nH = cfg.X_H * rho_cgs / M_P_G
    ne_full = (cfg.X_H + 0.5 * cfg.Y_He) * rho_cgs / M_P_G
    mode = cfg.ionization_mode
    if mode == "auto":
        mode = "temperature_weighted" if temp is not None else "fully_ionized"
    if mode == "fully_ionized":
        ne = ne_full
    elif mode == "hydrogen_only":
        ne = nH.copy()
    elif mode == "temperature_cut":
        if temp is None:
            raise ValueError("ionization_mode=temperature_cut requires temperature.")
        ne = np.where(temp >= cfg.ionized_Tmin_K, ne_full, 0.0)
    elif mode == "temperature_weighted":
        if temp is None:
            raise ValueError("ionization_mode=temperature_weighted requires temperature.")
        logT = np.log10(np.clip(temp, 1.0, None))
        logT0 = np.log10(cfg.ionized_Tmid_K)
        width = max(float(cfg.ionized_logT_width), 1.0e-6)
        ion_frac = 1.0 / (1.0 + np.exp(-(logT - logT0) / width))
        ne = ion_frac * ne_full
    else:
        raise ValueError(f"Unknown ionization_mode: {cfg.ionization_mode}")
    return ne.astype(np.float64), nH.astype(np.float64)


def _field_values_are_abundance(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return np.nanmin(finite) >= -1.0e-6 and np.nanpercentile(finite, 99.9) <= 1.5


def _looks_like_abundance_floor(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return np.nanpercentile(finite, 99.9) < 1.0e-8


def _number_density_or_fraction_to_density(values: np.ndarray, species_density: np.ndarray) -> np.ndarray:
    if _field_values_are_abundance(values):
        return np.clip(values, 0.0, None) * species_density
    return np.clip(values, 0.0, None)


def _get_species_density(data, ds, gas_ftype: str, names: Sequence[str], species_density: np.ndarray):
    values, field = get_scalar_field(data, ds, [gas_ftype, "gas"], names, "cm**-3")
    if values is None:
        return None, None
    if _field_values_are_abundance(values) and _looks_like_abundance_floor(values):
        print(f"WARNING: ignoring {field} because it looks like an ion-abundance floor: "
              f"min={np.nanmin(values):.3e}, median={np.nanmedian(values):.3e}, max={np.nanmax(values):.3e}")
        return None, None
    return _number_density_or_fraction_to_density(values, species_density), field


def build_ne_from_ion_fields(cfg: Config, ds, data, gas_ftype: str, code_cfg: Dict[str, Any], rho: np.ndarray):
    nH = cfg.X_H * rho / M_P_G
    nHe = cfg.Y_He * rho / (4.0 * M_P_G)
    nHII, f_HII = _get_species_density(data, ds, gas_ftype, code_cfg.get("hii_names", []), nH)
    nHeII, f_HeII = _get_species_density(data, ds, gas_ftype, code_cfg.get("heii_names", []), nHe)
    nHeIII, f_HeIII = _get_species_density(data, ds, gas_ftype, code_cfg.get("heiii_names", []), nHe)
    if nHII is None and nHeII is None and nHeIII is None:
        return None, []
    ne = np.zeros_like(rho, dtype=np.float64)
    fields = []
    if nHII is not None:
        ne += nHII
        fields.append(f_HII)
    if nHeII is not None:
        ne += nHeII
        fields.append(f_HeII)
    if nHeIII is not None:
        ne += 2.0 * nHeIII
        fields.append(f_HeIII)
    return ne, fields


def build_particle_ne_nh(cfg: Config, ds, data, gas_ftype: str, code_cfg: Dict[str, Any], rho: np.ndarray, temp: Optional[np.ndarray]):
    ne_model, nH = ne_nh_from_rho(cfg, rho, temp)

    # Prefer ion species when present. This is required for Gadget-family
    # snapshots where ElectronAbundance can be only a floor value.
    ne_ions, ion_fields = build_ne_from_ion_fields(cfg, ds, data, gas_ftype, code_cfg, rho)
    if ne_ions is not None:
        return ne_ions, nH, tuple(ion_fields)

    # Prefer a direct electron number density field if available. Abundance-like
    # fields are only used when they are not a near-zero placeholder floor.
    ne_field = None
    ne_direct = None
    if cfg.prefer_electron_field:
        ne_direct, ne_field = get_scalar_field(data, ds, [gas_ftype, "gas"], code_cfg.get("electron_names", []), "cm**-3")
        if ne_direct is not None:
            # If field is an abundance rather than a number density, infer by dimension/range heuristics.
            if np.nanmedian(ne_direct) < 5.0 and np.nanmax(ne_direct) <= 10.0:
                # Some Gadget-family snapshots contain a placeholder ElectronAbundance
                # floor (for example all gas at ~1e-20). Treat that as unavailable
                # instead of suppressing ne by many orders of magnitude.
                if _looks_like_abundance_floor(ne_direct):
                    print(f"WARNING: ignoring {ne_field} because it looks like an electron-abundance floor: "
                          f"min={np.nanmin(ne_direct):.3e}, median={np.nanmedian(ne_direct):.3e}, max={np.nanmax(ne_direct):.3e}")
                    ne_direct = None
                    ne_field = None
                else:
                    nH_tmp = cfg.X_H * rho / M_P_G
                    ne_direct = ne_direct * nH_tmp
    ne = ne_model if ne_direct is None else ne_direct
    return ne, nH, ne_field


# -----------------------------------------------------------------------------
# Direction sampling
# -----------------------------------------------------------------------------

def fibonacci_sphere(n: int, random_rotate: bool = False, seed: int = 12345) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * np.pi * i / golden
    dirs = np.vstack([r * np.cos(phi), r * np.sin(phi), z]).T
    if random_rotate:
        rng = np.random.default_rng(seed)
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        angle = rng.uniform(0, 2 * np.pi)
        K = skew(a)
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        dirs = (R @ dirs.T).T
    return dirs


def lon_lat_from_dirs(dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.rad2deg(np.arctan2(dirs[:, 1], dirs[:, 0]))
    lon = (lon + 360.0) % 360.0
    lat = np.rad2deg(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
    return lon, lat


def add_observer_angle_columns(data: Dict[str, np.ndarray], observer_faceon: np.ndarray) -> None:
    """Add angular distance from the Galactic-center direction as seen by the observer."""
    dirs = np.vstack([
        np.asarray(data["dir_x_faceon"], dtype=np.float64),
        np.asarray(data["dir_y_faceon"], dtype=np.float64),
        np.asarray(data["dir_z_faceon"], dtype=np.float64),
    ]).T
    to_gc = -np.asarray(observer_faceon, dtype=np.float64)
    norm = np.linalg.norm(to_gc)
    if not np.isfinite(norm) or norm == 0:
        return
    to_gc /= norm
    cosang = np.clip(np.sum(dirs * to_gc[None, :], axis=1), -1.0, 1.0)
    data["angle_from_galactic_center_deg"] = np.rad2deg(np.arccos(cosang))


def sky_position_angle_about_gc(data: Dict[str, np.ndarray], observer_faceon: np.ndarray) -> np.ndarray:
    """Position angle around the Galactic-center direction for polar sky plots."""
    dirs = np.vstack([
        np.asarray(data["dir_x_faceon"], dtype=np.float64),
        np.asarray(data["dir_y_faceon"], dtype=np.float64),
        np.asarray(data["dir_z_faceon"], dtype=np.float64),
    ]).T
    g = -np.asarray(observer_faceon, dtype=np.float64)
    g /= np.linalg.norm(g)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e1 = z - np.dot(z, g) * g
    if np.linalg.norm(e1) < 1.0e-8:
        e1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    return np.arctan2(np.sum(dirs * e2[None, :], axis=1), np.sum(dirs * e1[None, :], axis=1))


def init_output(cfg: Config, dirs: np.ndarray) -> Dict[str, np.ndarray]:
    lon, lat = lon_lat_from_dirs(dirs)
    return {
        "los_id": np.arange(cfg.n_los, dtype=np.int64),
        "l_deg": lon,
        "b_deg": lat,
        "dir_x_faceon": dirs[:, 0],
        "dir_y_faceon": dirs[:, 1],
        "dir_z_faceon": dirs[:, 2],
        "DM_total_pc_cm3": np.zeros(cfg.n_los),
        "DM_ISM_pc_cm3": np.zeros(cfg.n_los),
        "DM_CGM_pc_cm3": np.zeros(cfg.n_los),
        "DM_hot_pc_cm3": np.zeros(cfg.n_los),
        "EM_ne2_total_pc_cm6": np.zeros(cfg.n_los),
        "EM_ne_nH_total_pc_cm6": np.zeros(cfg.n_los),
        "EM_ne2_ISM_pc_cm6": np.zeros(cfg.n_los),
        "EM_ne2_CGM_pc_cm6": np.zeros(cfg.n_los),
        "EM_ne2_hot_pc_cm6": np.zeros(cfg.n_los),
    }


def masks_for_points_faceon(points: np.ndarray, temp: Optional[np.ndarray], cfg: Config):
    r_sph = np.linalg.norm(points, axis=-1)
    R_cyl = np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)
    abs_z = np.abs(points[..., 2])
    ism = (R_cyl <= cfg.ism_R_kpc) & (abs_z <= cfg.ism_abs_z_kpc)
    cgm = (r_sph >= cfg.cgm_inner_r_kpc) & (r_sph <= cfg.cgm_outer_r_kpc)
    hot = np.zeros_like(ism, dtype=bool) if temp is None else (temp >= cfg.hot_Tmin_K)
    return ism, cgm, hot


# -----------------------------------------------------------------------------
# Particle backend
# -----------------------------------------------------------------------------

def build_tree(gas_pos_faceon: np.ndarray):
    if cKDTree is None:
        raise ImportError("scipy is required for particle/KDTree interpolation.")
    return cKDTree(gas_pos_faceon)


def m6_kernel(dist_cm: np.ndarray, h_cm: np.ndarray, D: int = 3) -> np.ndarray:
    """Vectorized M6 spline kernel in the same piecewise form as the user's M6()."""
    dist_cm = np.asarray(dist_cm, dtype=np.float64)
    h_cm = np.asarray(h_cm, dtype=np.float64)
    W = np.zeros_like(dist_cm, dtype=np.float64)
    ok = np.isfinite(dist_cm) & np.isfinite(h_cm) & (h_cm > 0)
    if not np.any(ok):
        return W
    q = np.zeros_like(dist_cm, dtype=np.float64)
    q[ok] = dist_cm[ok] / h_cm[ok]
    sigma = np.array([1.0 / 120.0 * 3.0, 7.0 / (478.0 * np.pi) * 3.0**2, 1.0 / (120.0 * np.pi) * 3.0**3])
    w = np.zeros_like(dist_cm, dtype=np.float64)
    m1 = ok & (q >= 0.0) & (q < 1.0 / 3.0)
    m2 = ok & (q >= 1.0 / 3.0) & (q < 2.0 / 3.0)
    m3 = ok & (q >= 2.0 / 3.0) & (q < 1.0)
    w[m1] = 3.0**5 * (1.0 - q[m1])**5 - 6.0 * 3.0**5 * (2.0 / 3.0 - q[m1])**5 + 15.0 * 3.0**5 * (1.0 / 3.0 - q[m1])**5
    w[m2] = 3.0**5 * (1.0 - q[m2])**5 - 6.0 * 3.0**5 * (2.0 / 3.0 - q[m2])**5
    w[m3] = 3.0**5 * (1.0 - q[m3])**5
    W[ok] = h_cm[ok] ** (-D) * sigma[D - 1] * w[ok]
    return W


def _estimate_inverse_distance(points: np.ndarray, dist: np.ndarray, idx: np.ndarray, ne: np.ndarray,
                               nH: np.ndarray, temp: Optional[np.ndarray], cfg: Config,
                               rows: Optional[np.ndarray] = None):
    if rows is None:
        rows = np.arange(len(points))
    eps = 1e-6
    w = 1.0 / (dist[rows] + eps) ** 2
    nearest = dist[rows, 0]
    valid = np.isfinite(nearest) & (nearest <= cfg.max_kernel_radius_kpc)
    w[~valid, :] = 0.0
    sw = np.sum(w, axis=1)
    good = sw > 0
    ne_part = np.zeros(len(rows), dtype=np.float64)
    nH_part = np.zeros(len(rows), dtype=np.float64)
    T_part = np.full(len(rows), np.nan, dtype=np.float64)
    if np.any(good):
        wg = w[good] / sw[good, None]
        idg = idx[rows][good]
        ne_part[good] = np.sum(wg * ne[idg], axis=1)
        nH_part[good] = np.sum(wg * nH[idg], axis=1)
        if temp is not None:
            T_part[good] = np.sum(wg * temp[idg], axis=1)
    return rows, ne_part, nH_part, T_part, good


def estimate_ne_nh_T_at_points(points: np.ndarray, tree, gas_pos: np.ndarray,
                                ne: np.ndarray, nH: np.ndarray, temp: Optional[np.ndarray], cfg: Config,
                                gas_mass: Optional[np.ndarray] = None, gas_rho: Optional[np.ndarray] = None,
                                gas_hsml: Optional[np.ndarray] = None):
    k = min(cfg.n_ngb, len(gas_pos))
    dist, idx = tree.query(points, k=k, workers=-1)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    ne_p = np.zeros(len(points), dtype=np.float64)
    nH_p = np.zeros(len(points), dtype=np.float64)
    T_p = np.full(len(points), np.nan, dtype=np.float64)

    want_m6 = cfg.particle_interpolation in ("auto", "m6_kernel")
    have_m6 = gas_mass is not None and gas_rho is not None and gas_hsml is not None
    use_m6 = want_m6 and have_m6
    if cfg.particle_interpolation == "m6_kernel" and not have_m6:
        raise RuntimeError("--particle-interpolation m6_kernel requires gas mass, density, and smoothing length fields.")

    if use_m6:
        idn = idx
        h_kpc = np.asarray(gas_hsml[idn], dtype=np.float64)
        rho_cgs = np.asarray(gas_rho[idn], dtype=np.float64)
        mass_g = np.asarray(gas_mass[idn], dtype=np.float64) * MSUN_G
        W = m6_kernel(dist * KPC_CM, h_kpc * KPC_CM, D=3)
        vol_weight = np.where(rho_cgs > 0, mass_g / rho_cgs, 0.0) * W
        # Keep the original max radius as a safety cap, while the M6 support itself enforces r < h.
        nearest = dist[:, 0]
        safe = np.isfinite(nearest) & (nearest <= cfg.max_kernel_radius_kpc)
        vol_weight[~safe, :] = 0.0
        sw = np.sum(vol_weight, axis=1)
        good = sw > 0
        if np.any(good):
            idg = idn[good]
            wg = vol_weight[good]
            ne_p[good] = np.sum(ne[idg] * wg, axis=1)
            nH_p[good] = np.sum(nH[idg] * wg, axis=1)
            if temp is not None:
                T_p[good] = np.sum(temp[idg] * wg, axis=1) / sw[good]
        if cfg.particle_interpolation == "auto" and np.any(~good):
            rows, ne_fb, nH_fb, T_fb, _ = _estimate_inverse_distance(points, dist, idx, ne, nH, temp, cfg, rows=np.where(~good)[0])
            ne_p[rows] = ne_fb
            nH_p[rows] = nH_fb
            T_p[rows] = T_fb
        return ne_p, nH_p, T_p

    _, ne_part, nH_part, T_part, _ = _estimate_inverse_distance(points, dist, idx, ne, nH, temp, cfg)
    return ne_part, nH_part, T_part


def integrate_particle_sightlines(cfg: Config, gas_pos_faceon: np.ndarray, ne: np.ndarray, nH: np.ndarray,
                                  temp: Optional[np.ndarray], observer_faceon: np.ndarray, dirs_faceon: np.ndarray,
                                  gas_mass: Optional[np.ndarray] = None, gas_rho: Optional[np.ndarray] = None,
                                  gas_hsml: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    tree = build_tree(gas_pos_faceon)
    s_grid = np.arange(0.0, cfg.s_max_kpc + 0.5 * cfg.ds_kpc, cfg.ds_kpc, dtype=np.float64)
    dl_pc = cfg.ds_kpc * 1.0e3
    out = init_output(cfg, dirs_faceon)
    for start in range(0, cfg.n_los, cfg.chunk_los):
        end = min(cfg.n_los, start + cfg.chunk_los)
        d = dirs_faceon[start:end]
        points = observer_faceon[None, None, :] + d[:, None, :] * s_grid[None, :, None]
        flat = points.reshape(-1, 3)
        ne_p, nH_p, T_p = estimate_ne_nh_T_at_points(flat, tree, gas_pos_faceon, ne, nH, temp, cfg, gas_mass=gas_mass, gas_rho=gas_rho, gas_hsml=gas_hsml)
        shape = (end - start, len(s_grid))
        ne_p = ne_p.reshape(shape)
        nH_p = nH_p.reshape(shape)
        T_p = T_p.reshape(shape)
        ism, cgm, hot = masks_for_points_faceon(points, T_p if temp is not None else None, cfg)
        dm = ne_p * dl_pc
        em_ne2 = ne_p * ne_p * dl_pc
        em_nenh = ne_p * nH_p * dl_pc
        sl = slice(start, end)
        out["DM_total_pc_cm3"][sl] = np.sum(dm, axis=1)
        out["DM_ISM_pc_cm3"][sl] = np.sum(np.where(ism, dm, 0.0), axis=1)
        out["DM_CGM_pc_cm3"][sl] = np.sum(np.where(cgm, dm, 0.0), axis=1)
        out["DM_hot_pc_cm3"][sl] = np.sum(np.where(hot, dm, 0.0), axis=1)
        out["EM_ne2_total_pc_cm6"][sl] = np.sum(em_ne2, axis=1)
        out["EM_ne_nH_total_pc_cm6"][sl] = np.sum(em_nenh, axis=1)
        out["EM_ne2_ISM_pc_cm6"][sl] = np.sum(np.where(ism, em_ne2, 0.0), axis=1)
        out["EM_ne2_CGM_pc_cm6"][sl] = np.sum(np.where(cgm, em_ne2, 0.0), axis=1)
        out["EM_ne2_hot_pc_cm6"][sl] = np.sum(np.where(hot, em_ne2, 0.0), axis=1)
        print(f"[particle] Integrated LOS {start} - {end} / {cfg.n_los}")
    return out


# -----------------------------------------------------------------------------
# Grid/AMR backend
# -----------------------------------------------------------------------------

def field_value_or_model_on_ray(cfg: Config, ray, ds, code_cfg: Dict[str, Any], rho_field: Tuple[str, str], temp_field: Optional[Tuple[str, str]]):
    rho = np.asarray(ray[rho_field].to("g/cm**3"), dtype=np.float64)
    temp = None if temp_field is None else np.asarray(ray[temp_field].to("K"), dtype=np.float64)
    ne = None
    ne_field = None
    ne_ions, ion_fields = build_ne_from_ion_fields(cfg, ds, ray, rho_field[0], code_cfg, rho)
    if ne_ions is not None:
        ne = ne_ions
        ne_field = tuple(ion_fields)
    if cfg.prefer_electron_field:
        f = first_existing_field(ds, code_cfg.get("gas_types", ["gas"]), code_cfg.get("electron_names", []))
        if ne is None and f is not None:
            try:
                ne = np.asarray(ray[f].to("cm**-3"), dtype=np.float64)
                ne_field = f
            except Exception:
                val = np.asarray(ray[f], dtype=np.float64)
                # abundance fallback
                if np.nanmedian(val) < 5.0 and np.nanmax(val) <= 10.0:
                    if _looks_like_abundance_floor(val):
                        print(f"WARNING: ignoring {f} because it looks like an electron-abundance floor: "
                              f"min={np.nanmin(val):.3e}, median={np.nanmedian(val):.3e}, max={np.nanmax(val):.3e}")
                    else:
                        ne = val * cfg.X_H * rho / M_P_G
                        ne_field = f
                else:
                    ne = val
                    ne_field = f
    ne_model, nH = ne_nh_from_rho(cfg, rho, temp)
    if ne is None:
        ne = ne_model
    return ne, nH, temp, ne_field


def ray_dl_pc(ray, start_kpc: np.ndarray, end_kpc: np.ndarray, expected_n: int) -> np.ndarray:
    # yt arbitrary rays usually include ('index','dts'), fractional cell path lengths.
    # Some ART/Enzo rays expose the field but return an empty array; in that case keep
    # array lengths aligned with the sampled fluid fields and use equal path lengths.
    length_pc = np.linalg.norm(end_kpc - start_kpc) * 1.0e3
    if ("index", "dts") in ray.ds.field_list or ("index", "dts") in ray.ds.derived_field_list:
        try:
            dts = np.asarray(ray[("index", "dts")], dtype=np.float64)
            if len(dts) == expected_n:
                return dts * length_pc
        except Exception:
            pass
    return np.full(expected_n, length_pc / max(expected_n, 1), dtype=np.float64)


def integrate_gridray_sightlines(cfg: Config, ds, code_cfg: Dict[str, Any], center_original_kpc: np.ndarray,
                                 R_faceon: np.ndarray, observer_faceon: np.ndarray, dirs_faceon: np.ndarray,
                                 rho_field: Tuple[str, str], temp_field: Optional[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    out = init_output(cfg, dirs_faceon)
    observer_original = faceon_to_original(observer_faceon[None, :], center_original_kpc, R_faceon)[0]
    for i, d_faceon in enumerate(dirs_faceon):
        start_faceon = observer_faceon
        end_faceon = observer_faceon + d_faceon * cfg.s_max_kpc
        start_original = observer_original
        end_original = faceon_to_original(end_faceon[None, :], center_original_kpc, R_faceon)[0]
        ray = ds.ray(ds.arr(start_original, "kpc"), ds.arr(end_original, "kpc"))
        ne, nH, T, _ = field_value_or_model_on_ray(cfg, ray, ds, code_cfg, rho_field, temp_field)
        dl_pc = ray_dl_pc(ray, start_original, end_original, len(ne))
        # Recover positions along ray for ISM/CGM masks.
        if ("index", "x") in ds.field_list or ("index", "x") in ds.derived_field_list:
            pos_original = np.vstack([
                np.asarray(ray[("index", "x")].to("kpc"), dtype=np.float64),
                np.asarray(ray[("index", "y")].to("kpc"), dtype=np.float64),
                np.asarray(ray[("index", "z")].to("kpc"), dtype=np.float64),
            ]).T
            pos_faceon = original_to_faceon(pos_original, center_original_kpc, R_faceon)
        else:
            # Fallback: use midpoint samples along parametric ray.
            t = np.linspace(0.0, 1.0, len(ne), endpoint=False) + 0.5 / max(len(ne), 1)
            pos_faceon = start_faceon[None, :] + t[:, None] * (end_faceon - start_faceon)[None, :]
        # Sort by t if available, keeping arrays aligned.
        if ("index", "t") in ds.field_list or ("index", "t") in ds.derived_field_list:
            t_values = np.asarray(ray[("index", "t")], dtype=np.float64)
            if len(t_values) == len(ne):
                order = np.argsort(t_values)
                ne, nH, dl_pc, pos_faceon = ne[order], nH[order], dl_pc[order], pos_faceon[order]
                if T is not None:
                    T = T[order]
        ism, cgm, hot = masks_for_points_faceon(pos_faceon, T, cfg)
        dm = ne * dl_pc
        em_ne2 = ne * ne * dl_pc
        em_nenh = ne * nH * dl_pc
        out["DM_total_pc_cm3"][i] = np.sum(dm)
        out["DM_ISM_pc_cm3"][i] = np.sum(np.where(ism, dm, 0.0))
        out["DM_CGM_pc_cm3"][i] = np.sum(np.where(cgm, dm, 0.0))
        out["DM_hot_pc_cm3"][i] = np.sum(np.where(hot, dm, 0.0))
        out["EM_ne2_total_pc_cm6"][i] = np.sum(em_ne2)
        out["EM_ne_nH_total_pc_cm6"][i] = np.sum(em_nenh)
        out["EM_ne2_ISM_pc_cm6"][i] = np.sum(np.where(ism, em_ne2, 0.0))
        out["EM_ne2_CGM_pc_cm6"][i] = np.sum(np.where(cgm, em_ne2, 0.0))
        out["EM_ne2_hot_pc_cm6"][i] = np.sum(np.where(hot, em_ne2, 0.0))
        if (i + 1) % max(1, cfg.chunk_los) == 0 or i == cfg.n_los - 1:
            print(f"[gridray] Integrated LOS {i + 1} / {cfg.n_los}")
    return out


# -----------------------------------------------------------------------------
# Loading arrays for geometry and particle backend
# -----------------------------------------------------------------------------

def load_geometry_arrays(cfg: Config, ds, code_cfg: Dict[str, Any]):
    ad = ds.all_data()
    gas_ftype = choose_ftype(ds, cfg.gas_type, code_cfg.get("gas_types", ["gas"]), code_cfg.get("density_names", ["density"]))
    if gas_ftype is None:
        raise RuntimeError("Could not determine gas field/particle type. Use --gas-type explicitly.")
    star_ftype = choose_ftype(ds, cfg.star_type, code_cfg.get("star_types", []), ["particle_position_x", "particle_position", "Coordinates", "x"])

    print("Using gas ftype:", gas_ftype)
    print("Using star ftype:", star_ftype)

    gas_pos = gas_vel = gas_mass = gas_rho = gas_temp = None
    star_pos = star_vel = star_mass = None
    rho_field = temp_field = None

    # For both grid and particle, we load a possibly subsampled geometry set.
    try:
        gas_pos = get_positions(ad, ds, gas_ftype)
        gas_vel = get_velocities(ad, ds, gas_ftype, code_cfg)
        gas_mass = get_mass(ad, ds, gas_ftype, code_cfg.get("gas_mass_names", ["cell_mass", "Masses", "Mass"]))
        gas_rho, rho_field = get_density_cgs(ad, ds, [gas_ftype, "gas"], code_cfg.get("density_names", ["density"]))
        gas_temp, temp_field = get_temperature(ad, ds, [gas_ftype, "gas"], code_cfg.get("temperature_names", ["temperature"]))
        gas_pos, gas_vel, gas_mass, gas_rho, gas_temp = random_subsample(
            gas_pos, gas_vel, gas_mass, gas_rho, gas_temp, max_n=cfg.max_elements_for_geometry, seed=cfg.seed
        )
        print(f"Loaded gas geometry elements: {len(gas_pos)}")
    except Exception as exc:
        print("WARNING: failed to load gas geometry arrays:", exc)

    if normalize_code(cfg.code) == "ENZO":
        try:
            star_pos, star_vel, star_mass, enzo_star_ftype = load_enzo_stellar_particle_arrays(
                ad, ds, code_cfg, cfg.max_elements_for_geometry, cfg.seed + 1
            )
            if star_pos is not None:
                star_ftype = enzo_star_ftype
                print(f"Loaded Enzo stellar particle geometry elements: {len(star_pos)} from {enzo_star_ftype} particle_type==2")
        except Exception as exc:
            print("WARNING: failed to load Enzo stellar particle geometry arrays:", exc)
            star_pos = star_vel = star_mass = None
    elif star_ftype is not None:
        try:
            star_pos = get_positions(ad, ds, star_ftype)
            star_vel = get_velocities(ad, ds, star_ftype, code_cfg)
            star_mass = get_mass(ad, ds, star_ftype, code_cfg.get("star_mass_names", ["particle_mass", "Masses", "Mass"]))
            star_pos, star_vel, star_mass = random_subsample(star_pos, star_vel, star_mass, max_n=cfg.max_elements_for_geometry, seed=cfg.seed + 1)
            print(f"Loaded star geometry elements: {len(star_pos)}")
        except Exception as exc:
            print("WARNING: failed to load star geometry arrays:", exc)
            star_pos = star_vel = star_mass = None

    return {
        "gas_ftype": gas_ftype,
        "star_ftype": star_ftype,
        "gas_pos": gas_pos,
        "gas_vel": gas_vel,
        "gas_mass": gas_mass,
        "gas_rho": gas_rho,
        "gas_temp": gas_temp,
        "star_pos": star_pos,
        "star_vel": star_vel,
        "star_mass": star_mass,
        "rho_field": rho_field,
        "temp_field": temp_field,
    }


def apply_dataset_specific_geometry_fallbacks(cfg: Config, code: str, arrays: Dict[str, Any]) -> Config:
    """Avoid treating aggregate particle containers as stellar disks."""
    local_cfg = Config(**asdict(cfg))
    normalized = normalize_code(code)
    star_ftype = arrays.get("star_ftype")

    if normalized == "ENZO" and arrays.get("star_pos") is None:
        print("WARNING: Enzo stellar particles could not be isolated; falling back to gas geometry.")
        arrays["star_ftype"] = None
        arrays["star_vel"] = None
        arrays["star_mass"] = None
        if local_cfg.center_mode == "stellar_com":
            print("WARNING: switching Enzo center_mode from stellar_com to gas_density_peak.")
            local_cfg.center_mode = "gas_density_peak"
        if local_cfg.disk_normal_source == "stars":
            print("WARNING: switching Enzo disk_normal_source from stars to cold_gas.")
            local_cfg.disk_normal_source = "cold_gas"

    if normalized == "AREPO" and local_cfg.center_mode == "stellar_com":
        # The AGORA AREPO z=0 file is a full cosmological box.  A COM over all
        # stars lands between structures rather than on the target galaxy, which
        # makes the 40 kpc projection window empty.  The gas-density peak is a
        # robust galaxy center for this snapshot and keeps manual runs usable.
        print("WARNING: switching AREPO center_mode from stellar_com to gas_density_peak for full-box snapshot.")
        local_cfg.center_mode = "gas_density_peak"

    return local_cfg


def load_full_particle_gas_for_backend(cfg: Config, ds, code_cfg: Dict[str, Any], gas_ftype: str):
    ad = ds.all_data()
    gas_pos = get_positions(ad, ds, gas_ftype)
    rho, rho_field = get_density_cgs(ad, ds, [gas_ftype, "gas"], code_cfg.get("density_names", ["density"]))
    temp, temp_field = get_temperature(ad, ds, [gas_ftype, "gas"], code_cfg.get("temperature_names", ["temperature"]))
    ne, nH, ne_field = build_particle_ne_nh(cfg, ds, ad, gas_ftype, code_cfg, rho, temp)
    gas_mass = get_mass(ad, ds, gas_ftype, code_cfg.get("gas_mass_names", ["Masses", "Mass", "particle_mass"]))
    hsml, hsml_field = get_smoothing_length(ad, ds, gas_ftype, code_cfg.get("smoothing_length_names", ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"]))
    return gas_pos, rho, temp, ne, nH, rho_field, temp_field, ne_field, gas_mass, hsml, hsml_field


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def write_csv(path: str, data: Dict[str, np.ndarray]) -> None:
    keys = list(data.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(len(data[keys[0]])):
            writer.writerow([data[k][i] for k in keys])


def summarize_results(data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    summary = {}
    skip = {"los_id", "l_deg", "b_deg", "dir_x_faceon", "dir_y_faceon", "dir_z_faceon"}
    for key, arr in data.items():
        if key in skip:
            continue
        summary[key] = {
            "mean": float(np.nanmean(arr)),
            "median": float(np.nanmedian(arr)),
            "p16": float(np.nanpercentile(arr, 16)),
            "p84": float(np.nanpercentile(arr, 84)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }
    return summary


def save_outputs(cfg: Config, data: Dict[str, np.ndarray], metadata: Dict) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    csv_path = os.path.join(cfg.outdir, "MWlike_4pi_DM_EM_sightlines.csv")
    npz_path = os.path.join(cfg.outdir, "MWlike_4pi_DM_EM_sightlines.npz")
    meta_path = os.path.join(cfg.outdir, "MWlike_4pi_DM_EM_metadata.json")
    summary_path = os.path.join(cfg.outdir, "MWlike_4pi_DM_EM_summary.json")
    write_csv(csv_path, data)
    np.savez_compressed(npz_path, **data)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summarize_results(data), f, indent=2)
    print("Saved:")
    print(" ", csv_path)
    print(" ", npz_path)
    print(" ", meta_path)
    print(" ", summary_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_vec3(values: Optional[Sequence[str]]) -> Optional[Tuple[float, float, float]]:
    if values is None:
        return None
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Need exactly three numbers.")
    return (float(values[0]), float(values[1]), float(values[2]))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-code AGORA MW-like 3D DM/EM 4pi sightline framework")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--code", default="auto", help="ART-I/RAMSES/ENZO/GADGET-3/GADGET-4/GEAR/GIZMO/CHANGA/AREPO/GASOLINE")
    p.add_argument("--outdir", default="MW_DM_EM_output")
    p.add_argument("--unit-base", choices=["auto", "none", "agora_gadget", "agora_arepo"], default="auto",
                   help="auto reads an HDF5 Units group when present; none uses yt defaults.")
    p.add_argument("--tipsy-length-unit-kpc", type=float, default=None,
                   help="Explicit Tipsy length code unit in physical kpc, required for CHANGA if no parameter file is available.")
    p.add_argument("--tipsy-mass-unit-msun", type=float, default=None,
                   help="Explicit Tipsy mass code unit in Msun, required for CHANGA if no parameter file is available.")
    p.add_argument("--tipsy-time-unit-s", type=float, default=None,
                   help="Explicit Tipsy time code unit in seconds; sets velocity_unit=length/time for Tipsy data.")
    p.add_argument("--integration-backend", choices=["auto", "particle", "gridray"], default="auto")
    p.add_argument("--gas-type", default="auto")
    p.add_argument("--star-type", default="auto")
    p.add_argument("--no-prefer-electron-field", dest="prefer_electron_field", action="store_false")

    p.add_argument("--center-mode", choices=["stellar_com", "gas_com", "star_gas_com", "gas_density_peak", "manual"], default="stellar_com")
    p.add_argument("--center-kpc", nargs=3)
    p.add_argument("--center-radius-kpc", type=float, default=30.0)
    p.add_argument("--velocity-reference-source", choices=["none", "stars", "gas", "star_gas"], default="stars",
                   help="Bulk velocity source subtracted from projection velocity maps.")
    p.add_argument("--velocity-reference-radius-kpc", type=float, default=30.0,
                   help="Radius around the center for computing the bulk velocity reference.")

    p.add_argument("--disk-normal-source", choices=["stars", "cold_gas", "gas", "star_gas", "manual"], default="stars")
    p.add_argument("--manual-disk-normal", nargs=3)
    p.add_argument("--angular-momentum-radius-kpc", type=float, default=20.0)
    p.add_argument("--cold-gas-Tmax-K", type=float, default=3.0e4)

    p.add_argument("--R-sun-kpc", type=float, default=8.2)
    p.add_argument("--phi-sun-deg", type=float, default=0.0)
    p.add_argument("--z-sun-kpc", type=float, default=0.020)

    p.add_argument("--n-los", type=int, default=1024, help="Number of 4pi sightlines.")
    p.add_argument("--nside", type=int, default=None, help="Deprecated alias of --n-los for compatibility.")
    p.add_argument("--s-max-kpc", type=float, default=250.0)
    p.add_argument("--ds-kpc", type=float, default=0.25)
    p.add_argument("--chunk-los", type=int, default=64)

    p.add_argument("--n-ngb", type=int, default=32)
    p.add_argument("--max-kernel-radius-kpc", type=float, default=5.0)
    p.add_argument("--particle-interpolation", choices=["auto", "inverse_distance", "m6_kernel"], default="auto",
                   help="Particle LOS interpolation. auto uses M6 smoothing-length kernel when mass/rho/hsml exist, otherwise inverse_distance.")
    p.add_argument("--max-elements-for-geometry", type=int, default=2_000_000)

    p.add_argument("--X-H", type=float, default=0.76)
    p.add_argument("--Y-He", type=float, default=0.24)
    p.add_argument("--ionization-mode", choices=["auto", "fully_ionized", "hydrogen_only", "temperature_cut", "temperature_weighted"], default="auto",
                   help="auto uses real ion/electron fields when available; otherwise temperature_weighted if temperature exists, else fully_ionized.")
    p.add_argument("--ionized-Tmin-K", type=float, default=1.0e4)
    p.add_argument("--ionized-Tmid-K", type=float, default=1.0e4,
                   help="Midpoint temperature for --ionization-mode temperature_weighted.")
    p.add_argument("--ionized-logT-width", type=float, default=0.25,
                   help="Log10 temperature width for the smooth temperature-weighted ionization transition.")

    p.add_argument("--ism-R-kpc", type=float, default=20.0)
    p.add_argument("--ism-abs-z-kpc", type=float, default=5.0)
    p.add_argument("--cgm-inner-r-kpc", type=float, default=20.0)
    p.add_argument("--cgm-outer-r-kpc", type=float, default=250.0)
    p.add_argument("--hot-Tmin-K", type=float, default=1.0e6)

    p.add_argument("--random-rotate-directions", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    return p


def config_from_args(args) -> Config:
    n_los = args.n_los if args.nside is None else args.nside
    return Config(
        snapshot=args.snapshot,
        code=args.code,
        outdir=args.outdir,
        unit_base=args.unit_base,
        tipsy_length_unit_kpc=args.tipsy_length_unit_kpc,
        tipsy_mass_unit_msun=args.tipsy_mass_unit_msun,
        tipsy_time_unit_s=args.tipsy_time_unit_s,
        integration_backend=args.integration_backend,
        gas_type=args.gas_type,
        star_type=args.star_type,
        prefer_electron_field=args.prefer_electron_field,
        center_mode=args.center_mode,
        center_kpc=parse_vec3(args.center_kpc),
        center_radius_kpc=args.center_radius_kpc,
        velocity_reference_source=args.velocity_reference_source,
        velocity_reference_radius_kpc=args.velocity_reference_radius_kpc,
        disk_normal_source=args.disk_normal_source,
        manual_disk_normal=parse_vec3(args.manual_disk_normal),
        angular_momentum_radius_kpc=args.angular_momentum_radius_kpc,
        cold_gas_Tmax_K=args.cold_gas_Tmax_K,
        R_sun_kpc=args.R_sun_kpc,
        phi_sun_deg=args.phi_sun_deg,
        z_sun_kpc=args.z_sun_kpc,
        n_los=n_los,
        s_max_kpc=args.s_max_kpc,
        ds_kpc=args.ds_kpc,
        chunk_los=args.chunk_los,
        n_ngb=args.n_ngb,
        max_kernel_radius_kpc=args.max_kernel_radius_kpc,
        particle_interpolation=args.particle_interpolation,
        max_elements_for_geometry=args.max_elements_for_geometry,
        X_H=args.X_H,
        Y_He=args.Y_He,
        ionization_mode=args.ionization_mode,
        ionized_Tmin_K=args.ionized_Tmin_K,
        ionized_Tmid_K=args.ionized_Tmid_K,
        ionized_logT_width=args.ionized_logT_width,
        ism_R_kpc=args.ism_R_kpc,
        ism_abs_z_kpc=args.ism_abs_z_kpc,
        cgm_inner_r_kpc=args.cgm_inner_r_kpc,
        cgm_outer_r_kpc=args.cgm_outer_r_kpc,
        hot_Tmin_K=args.hot_Tmin_K,
        random_rotate_directions=args.random_rotate_directions,
        seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = make_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring unknown arguments:", unknown)
    cfg = config_from_args(args)
    code = normalize_code(cfg.code)
    if code == "AUTO":
        code = infer_code_from_snapshot_path(cfg.snapshot)
        print("Inferred code from snapshot path:", code)

    ds = load_dataset(
        cfg.snapshot, code, cfg.unit_base,
        cfg.tipsy_length_unit_kpc, cfg.tipsy_mass_unit_msun, cfg.tipsy_time_unit_s,
    )
    validate_dataset_units(ds, code, cfg)
    code_cfg = AGORA_CODE_CONFIG.get(code, default_config_for_unknown(code, ds))
    print("Normalized code:", code)
    print("Code family:", code_cfg["family"], "(", code_backend_family(code_cfg), ")")
    add_common_derived_fields(ds, code, code_cfg)

    arrays = load_geometry_arrays(cfg, ds, code_cfg)
    cfg = apply_dataset_specific_geometry_fallbacks(cfg, code, arrays)
    center = determine_center(cfg, arrays["gas_pos"], arrays["gas_mass"], arrays["gas_rho"], arrays["star_pos"], arrays["star_mass"])
    print("Center [kpc, original frame] =", center)

    disk_normal = compute_angular_momentum_normal(
        cfg,
        arrays["gas_pos"], arrays["gas_vel"], arrays["gas_mass"], arrays["gas_temp"],
        arrays["star_pos"], arrays["star_vel"], arrays["star_mass"],
        center,
    )
    print("Disk normal [original frame] =", disk_normal)
    R_faceon = rotation_matrix_from_vectors(disk_normal, np.array([0.0, 0.0, 1.0]))
    observer_faceon = observer_position_faceon(cfg)
    dirs_faceon = fibonacci_sphere(cfg.n_los, cfg.random_rotate_directions, cfg.seed)
    print("Observer [kpc, face-on frame] =", observer_faceon)

    backend = cfg.integration_backend
    if backend == "auto":
        backend = "gridray" if code_cfg["family"] == "grid" else "particle"
    print("Integration backend:", backend)

    ne_field_used = None
    if backend == "particle":
        gas_pos, rho, temp, ne, nH, rho_field, temp_field, ne_field_used, gas_mass_full, gas_hsml, hsml_field = load_full_particle_gas_for_backend(
            cfg, ds, code_cfg, arrays["gas_ftype"]
        )
        gas_pos_faceon = original_to_faceon(gas_pos, center, R_faceon)
        print(f"Full gas elements for particle backend: {len(gas_pos_faceon)}")
        print("Density field used:", rho_field)
        print("Temperature field used:", temp_field)
        print("Electron field used:", ne_field_used)
        print("Smoothing length field used:", hsml_field)
        print("Particle interpolation:", cfg.particle_interpolation)
        data = integrate_particle_sightlines(cfg, gas_pos_faceon, ne, nH, temp, observer_faceon, dirs_faceon,
                                             gas_mass=gas_mass_full, gas_rho=rho, gas_hsml=gas_hsml)
    elif backend == "gridray":
        rho_field = arrays["rho_field"]
        temp_field = arrays["temp_field"]
        if rho_field is None:
            # Re-find directly without needing geometry arrays.
            rho_field = first_existing_field(ds, code_cfg.get("gas_types", ["gas"]), code_cfg.get("density_names", ["density"]))
        if temp_field is None:
            temp_field = first_existing_field(ds, code_cfg.get("gas_types", ["gas"]), code_cfg.get("temperature_names", ["temperature"]))
        if rho_field is None:
            raise RuntimeError("gridray backend requires a gas density field.")
        print("Density field used:", rho_field)
        print("Temperature field used:", temp_field)
        data = integrate_gridray_sightlines(cfg, ds, code_cfg, center, R_faceon, observer_faceon, dirs_faceon, rho_field, temp_field)
    else:
        raise ValueError(f"Unknown integration backend: {backend}")

    metadata = {
        "config": asdict(cfg),
        "dataset": str(ds),
        "dataset_units": dataset_unit_metadata(ds),
        "normalized_code": code,
        "code_family": code_cfg["family"],
        "backend_family_label": code_backend_family(code_cfg),
        "integration_backend": backend,
        "gas_ftype": arrays["gas_ftype"],
        "star_ftype": arrays["star_ftype"],
        "density_field": str(arrays["rho_field"]),
        "temperature_field": str(arrays["temp_field"]),
        "electron_field_used_particle_backend": str(ne_field_used),
        "smoothing_length_field_used_particle_backend": str(locals().get("hsml_field", None)),
        "particle_interpolation": cfg.particle_interpolation,
        "center_kpc_original_frame": center.tolist(),
        "disk_normal_original_frame": disk_normal.tolist(),
        "rotation_matrix_original_to_faceon": R_faceon.tolist(),
        "observer_kpc_faceon_frame": observer_faceon.tolist(),
        "notes": {
            "DM_units": "pc cm^-3",
            "EM_units": "pc cm^-6",
            "faceon_frame": "x-y is disk plane and z is disk angular-momentum axis",
            "ISM_mask": "R_cyl <= ism_R_kpc and |z| <= ism_abs_z_kpc in face-on frame",
            "CGM_mask": "cgm_inner_r_kpc <= r_sph <= cgm_outer_r_kpc in face-on frame",
            "particle_backend": "k-nearest inverse-distance-squared interpolation, not strict SPH kernel",
            "gridray_backend": "yt arbitrary ray integration through native grid/AMR cells",
        },
    }
    save_outputs(cfg, data, metadata)




# =============================================================================
# Parallel pipeline, projection maps, Mollweide plots, and random observer tests
# =============================================================================

_base_make_parser = make_parser
_base_config_from_args = config_from_args


def parallel_available(n_jobs: int) -> bool:
    return Parallel is not None and delayed is not None and n_jobs not in (0, 1, -0)


def chunk_slices(n: int, chunk: int):
    chunk = max(1, int(chunk))
    for start in range(0, n, chunk):
        yield start, min(n, start + chunk)


def _particle_los_chunk(start: int, end: int, cfg: Config, tree, gas_pos_faceon: np.ndarray, ne: np.ndarray, nH: np.ndarray,
                        temp: Optional[np.ndarray], observer_faceon: np.ndarray, dirs_faceon: np.ndarray,
                        gas_mass: Optional[np.ndarray] = None, gas_rho: Optional[np.ndarray] = None, gas_hsml: Optional[np.ndarray] = None):
    s_grid = np.arange(0.0, cfg.s_max_kpc + 0.5 * cfg.ds_kpc, cfg.ds_kpc, dtype=np.float64)
    dl_pc = cfg.ds_kpc * 1.0e3
    d = dirs_faceon[start:end]
    points = observer_faceon[None, None, :] + d[:, None, :] * s_grid[None, :, None]
    flat = points.reshape(-1, 3)
    ne_p, nH_p, T_p = estimate_ne_nh_T_at_points(flat, tree, gas_pos_faceon, ne, nH, temp, cfg, gas_mass=gas_mass, gas_rho=gas_rho, gas_hsml=gas_hsml)
    shape = (end - start, len(s_grid))
    ne_p = ne_p.reshape(shape)
    nH_p = nH_p.reshape(shape)
    T_p = T_p.reshape(shape)
    ism, cgm, hot = masks_for_points_faceon(points, T_p if temp is not None else None, cfg)
    dm = ne_p * dl_pc
    em_ne2 = ne_p * ne_p * dl_pc
    em_nenh = ne_p * nH_p * dl_pc
    return start, end, {
        "DM_total_pc_cm3": np.sum(dm, axis=1),
        "DM_ISM_pc_cm3": np.sum(np.where(ism, dm, 0.0), axis=1),
        "DM_CGM_pc_cm3": np.sum(np.where(cgm, dm, 0.0), axis=1),
        "DM_hot_pc_cm3": np.sum(np.where(hot, dm, 0.0), axis=1),
        "EM_ne2_total_pc_cm6": np.sum(em_ne2, axis=1),
        "EM_ne_nH_total_pc_cm6": np.sum(em_nenh, axis=1),
        "EM_ne2_ISM_pc_cm6": np.sum(np.where(ism, em_ne2, 0.0), axis=1),
        "EM_ne2_CGM_pc_cm6": np.sum(np.where(cgm, em_ne2, 0.0), axis=1),
        "EM_ne2_hot_pc_cm6": np.sum(np.where(hot, em_ne2, 0.0), axis=1),
    }


def integrate_particle_sightlines_parallel(cfg: Config, gas_pos_faceon: np.ndarray, ne: np.ndarray, nH: np.ndarray,
                                           temp: Optional[np.ndarray], observer_faceon: np.ndarray,
                                           dirs_faceon: np.ndarray, n_jobs: int = 1,
                                           los_chunk: Optional[int] = None,
                                           gas_mass: Optional[np.ndarray] = None, gas_rho: Optional[np.ndarray] = None,
                                           gas_hsml: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    out = init_output(cfg, dirs_faceon)
    chunk = los_chunk or cfg.chunk_los
    jobs = list(chunk_slices(cfg.n_los, chunk))
    t0 = time.time()
    tree = build_tree(gas_pos_faceon)
    if parallel_available(n_jobs):
        results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads", verbose=5)(
            delayed(_particle_los_chunk)(s, e, cfg, tree, gas_pos_faceon, ne, nH, temp, observer_faceon, dirs_faceon, gas_mass, gas_rho, gas_hsml)
            for s, e in jobs
        )
    else:
        results = [_particle_los_chunk(s, e, cfg, tree, gas_pos_faceon, ne, nH, temp, observer_faceon, dirs_faceon, gas_mass, gas_rho, gas_hsml) for s, e in jobs]
    for s, e, block in results:
        for key, val in block.items():
            out[key][s:e] = val
        print(f"[particle/joblib] Integrated LOS {e} / {cfg.n_los}")
    print(f"[particle/joblib] total integration time: {(time.time() - t0) / 60.0:.2f} min")
    return out


def _gridray_one_los(i: int, cfg: Config, ds, code_cfg: Dict[str, Any], center_original_kpc: np.ndarray,
                     R_faceon: np.ndarray, observer_faceon: np.ndarray, dirs_faceon: np.ndarray,
                     rho_field: Tuple[str, str], temp_field: Optional[Tuple[str, str]]):
    d_faceon = dirs_faceon[i]
    observer_original = faceon_to_original(observer_faceon[None, :], center_original_kpc, R_faceon)[0]
    start_faceon = observer_faceon
    end_faceon = observer_faceon + d_faceon * cfg.s_max_kpc
    start_original = observer_original
    end_original = faceon_to_original(end_faceon[None, :], center_original_kpc, R_faceon)[0]
    ray = ds.ray(ds.arr(start_original, "kpc"), ds.arr(end_original, "kpc"))
    ne, nH, T, _ = field_value_or_model_on_ray(cfg, ray, ds, code_cfg, rho_field, temp_field)
    dl_pc = ray_dl_pc(ray, start_original, end_original, len(ne))
    if ("index", "x") in ds.field_list or ("index", "x") in ds.derived_field_list:
        pos_original = np.vstack([
            np.asarray(ray[("index", "x")].to("kpc"), dtype=np.float64),
            np.asarray(ray[("index", "y")].to("kpc"), dtype=np.float64),
            np.asarray(ray[("index", "z")].to("kpc"), dtype=np.float64),
        ]).T
        pos_faceon = original_to_faceon(pos_original, center_original_kpc, R_faceon)
    else:
        t = np.linspace(0.0, 1.0, len(ne), endpoint=False) + 0.5 / max(len(ne), 1)
        pos_faceon = start_faceon[None, :] + t[:, None] * (end_faceon - start_faceon)[None, :]
    if ("index", "t") in ds.field_list or ("index", "t") in ds.derived_field_list:
        t_values = np.asarray(ray[("index", "t")], dtype=np.float64)
        if len(t_values) == len(ne):
            order = np.argsort(t_values)
            ne, nH, dl_pc, pos_faceon = ne[order], nH[order], dl_pc[order], pos_faceon[order]
            if T is not None:
                T = T[order]
    ism, cgm, hot = masks_for_points_faceon(pos_faceon, T, cfg)
    dm = ne * dl_pc
    em_ne2 = ne * ne * dl_pc
    em_nenh = ne * nH * dl_pc
    return i, {
        "DM_total_pc_cm3": float(np.sum(dm)),
        "DM_ISM_pc_cm3": float(np.sum(np.where(ism, dm, 0.0))),
        "DM_CGM_pc_cm3": float(np.sum(np.where(cgm, dm, 0.0))),
        "DM_hot_pc_cm3": float(np.sum(np.where(hot, dm, 0.0))),
        "EM_ne2_total_pc_cm6": float(np.sum(em_ne2)),
        "EM_ne_nH_total_pc_cm6": float(np.sum(em_nenh)),
        "EM_ne2_ISM_pc_cm6": float(np.sum(np.where(ism, em_ne2, 0.0))),
        "EM_ne2_CGM_pc_cm6": float(np.sum(np.where(cgm, em_ne2, 0.0))),
        "EM_ne2_hot_pc_cm6": float(np.sum(np.where(hot, em_ne2, 0.0))),
    }


def integrate_gridray_sightlines_parallel(cfg: Config, ds, code_cfg: Dict[str, Any], center_original_kpc: np.ndarray,
                                          R_faceon: np.ndarray, observer_faceon: np.ndarray, dirs_faceon: np.ndarray,
                                          rho_field: Tuple[str, str], temp_field: Optional[Tuple[str, str]],
                                          n_jobs: int = 1) -> Dict[str, np.ndarray]:
    out = init_output(cfg, dirs_faceon)
    t0 = time.time()
    if parallel_available(n_jobs):
        results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads", verbose=5)(
            delayed(_gridray_one_los)(i, cfg, ds, code_cfg, center_original_kpc, R_faceon, observer_faceon, dirs_faceon, rho_field, temp_field)
            for i in range(cfg.n_los)
        )
    else:
        results = [_gridray_one_los(i, cfg, ds, code_cfg, center_original_kpc, R_faceon, observer_faceon, dirs_faceon, rho_field, temp_field)
                   for i in range(cfg.n_los)]
    for i, block in results:
        for key, val in block.items():
            out[key][i] = val
        if (i + 1) % max(1, cfg.chunk_los) == 0 or i == cfg.n_los - 1:
            print(f"[gridray/joblib] Integrated LOS {i + 1} / {cfg.n_los}")
    print(f"[gridray/joblib] total integration time: {(time.time() - t0) / 60.0:.2f} min")
    return out


def finite_percentile_limits(arr, pmin=5, pmax=99):
    vals = np.asarray(arr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    return float(np.nanpercentile(vals, pmin)), float(np.nanpercentile(vals, pmax))


def map_coords(pos, vel=None, view="face-on"):
    if view == "face-on":
        x, y = pos[:, 0], pos[:, 1]
        vx = None if vel is None else vel[:, 0]
        vy = None if vel is None else vel[:, 1]
        vlos = None if vel is None else vel[:, 2]
        xlabel, ylabel = "x_faceon [kpc]", "y_faceon [kpc]"
    elif view == "edge-on":
        x, y = pos[:, 0], pos[:, 2]
        vx = None if vel is None else vel[:, 0]
        vy = None if vel is None else vel[:, 2]
        vlos = None if vel is None else vel[:, 1]
        xlabel, ylabel = "x_faceon [kpc]", "z_faceon [kpc]"
    else:
        raise ValueError("view must be face-on or edge-on")
    return x, y, vx, vy, vlos, xlabel, ylabel


def _deposit_projected_kernel(x: np.ndarray, y: np.ndarray, weights: np.ndarray, hsml: np.ndarray,
                              box_kpc: float, npix: int, rng_seed: int = 12345,
                              max_particles: int = 500_000, max_radius_pixels: int = 12):
    """Deposit particles to a 2D grid with the same M6 kernel family as LOS interpolation.

    This is the projected 2D analog of the LOS particle interpolation kernel:
    ``m6_kernel(r, h, D=2)`` is evaluated on pixel centers and multiplied by
    pixel area, so the accumulated map is mass per pixel.  Dividing by pixel
    area later gives surface density.

    If the selected particle count is larger than max_particles, a random
    subset is used and weights are scaled to conserve total mass in expectation.
    """
    n_input = int(x.size)
    if n_input == 0:
        return np.zeros((npix, npix), dtype=np.float64), n_input, 0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    hsml = np.asarray(hsml, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & np.isfinite(hsml) & (weights > 0) & (hsml > 0)
    x, y, weights, hsml = x[finite], y[finite], weights[finite], hsml[finite]
    if x.size == 0:
        return np.zeros((npix, npix), dtype=np.float64), n_input, 0
    if max_particles is not None and max_particles > 0 and x.size > max_particles:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(x.size, size=int(max_particles), replace=False)
        scale = x.size / float(max_particles)
        x, y, weights, hsml = x[idx], y[idx], weights[idx] * scale, hsml[idx]
    half = box_kpc / 2.0
    pix = box_kpc / npix
    pixel_area_cm2 = (pix * KPC_CM) ** 2
    # Avoid delta-function maps for sub-pixel hsml and cap very broad kernels
    # for plotting cost.  The M6 support in m6_kernel is q < 1.
    h_eff = np.clip(hsml, 0.5 * pix, max_radius_pixels * pix)
    ix_c = np.floor((x + half) / pix).astype(np.int64)
    iy_c = np.floor((y + half) / pix).astype(np.int64)
    out = np.zeros((npix, npix), dtype=np.float64)
    for xi, yi, wi, hi, ix0, iy0 in zip(x, y, weights, h_eff, ix_c, iy_c):
        r_pix = max(1, int(np.ceil(hi / pix)))
        ix_min = max(0, ix0 - r_pix)
        ix_max = min(npix - 1, ix0 + r_pix)
        iy_min = max(0, iy0 - r_pix)
        iy_max = min(npix - 1, iy0 + r_pix)
        xs = -half + (np.arange(ix_min, ix_max + 1) + 0.5) * pix
        ys = -half + (np.arange(iy_min, iy_max + 1) + 0.5) * pix
        dx = xs[:, None] - xi
        dy = ys[None, :] - yi
        r_cm = np.sqrt(dx * dx + dy * dy) * KPC_CM
        h_cm = np.full_like(r_cm, hi * KPC_CM)
        kw = m6_kernel(r_cm, h_cm, D=2)
        if not np.any(kw > 0):
            if 0 <= ix0 < npix and 0 <= iy0 < npix:
                out[ix0, iy0] += wi
            continue
        out[ix_min:ix_max + 1, iy_min:iy_max + 1] += wi * kw * pixel_area_cm2
    return out, n_input, int(x.size)


def make_2d_maps(pos, mass=None, temp=None, vel=None, hsml=None, view="face-on", box_kpc=60.0, npix=256,
                 los_half_thickness_kpc: Optional[float] = None, use_kernel: bool = False,
                 kernel_max_particles: int = 500_000, kernel_seed: int = 12345):
    x, y, vx, vy, vlos, xlabel, ylabel = map_coords(pos, vel, view)
    half = box_kpc / 2.0
    if view == "face-on":
        los_coord = pos[:, 2]
    elif view == "edge-on":
        los_coord = pos[:, 1]
    else:
        raise ValueError("view must be face-on or edge-on")
    mask = np.isfinite(x) & np.isfinite(y) & (x >= -half) & (x <= half) & (y >= -half) & (y <= half)
    if los_half_thickness_kpc is not None:
        los_half = float(los_half_thickness_kpc)
        mask &= np.isfinite(los_coord) & (np.abs(los_coord) <= los_half)
    x = x[mask]
    y = y[mask]
    w_mass = np.ones_like(x, dtype=np.float64) if mass is None else np.asarray(mass)[mask]
    hsml_sel = None if hsml is None else np.asarray(hsml, dtype=np.float64)[mask]
    extent_range = [[-half, half], [-half, half]]
    area = (box_kpc / npix) ** 2
    method = "histogram2d"
    kernel_n_deposited = None

    def direct_particle_weighted_map(values):
        vals = np.asarray(values)[mask]
        good = np.isfinite(vals) & np.isfinite(w_mass) & (w_mass > 0)
        if not np.any(good):
            return np.full((npix, npix), np.nan, dtype=np.float64)
        num, _, _ = np.histogram2d(x[good], y[good], bins=[npix, npix], range=extent_range, weights=w_mass[good] * vals[good])
        den, _, _ = np.histogram2d(x[good], y[good], bins=[npix, npix], range=extent_range, weights=w_mass[good])
        return num / np.where(den > 0, den, np.nan)

    can_kernel = use_kernel and hsml_sel is not None and mass is not None and x.size > 0
    if can_kernel:
        mass_map, _, kernel_n_deposited = _deposit_projected_kernel(
            x, y, w_mass, hsml_sel, box_kpc, npix, rng_seed=kernel_seed,
            max_particles=kernel_max_particles,
        )
        method = "m6_2d_kernel"
        velocity_method = "direct_particle_bin_mass_weighted"
        def kernel_weighted_map(values):
            vals = np.asarray(values)[mask]
            num, _, _ = _deposit_projected_kernel(
                x, y, w_mass * vals, hsml_sel, box_kpc, npix, rng_seed=kernel_seed,
                max_particles=kernel_max_particles,
            )
            den = mass_map
            return num / np.where(den > 0, den, np.nan)
    else:
        mass_map, _, _ = np.histogram2d(x, y, bins=[npix, npix], range=extent_range, weights=w_mass)
        method = "histogram2d"
        velocity_method = "direct_particle_bin_mass_weighted"
        kernel_weighted_map = direct_particle_weighted_map

    sigma = mass_map / area
    temp_map = None if temp is None else kernel_weighted_map(np.asarray(temp))
    vx_map = vy_map = vlos_map = None
    if vel is not None:
        vx_all, vy_all, vlos_all = vx, vy, vlos
        # Velocity is a vector field, not a conserved projected scalar.  Do not
        # kernel-smooth it with gas mass; use the true particle/cell velocities
        # averaged only within the displayed pixel bins.
        vx_map = direct_particle_weighted_map(np.asarray(vx_all))
        vy_map = direct_particle_weighted_map(np.asarray(vy_all))
        vlos_map = direct_particle_weighted_map(np.asarray(vlos_all))
    return {
        "sigma": sigma.T,
        "temperature": None if temp_map is None else temp_map.T,
        "vx": None if vx_map is None else vx_map.T,
        "vy": None if vy_map is None else vy_map.T,
        "vlos": None if vlos_map is None else vlos_map.T,
        "extent": [-half, half, -half, half],
        "xlabel": xlabel,
        "ylabel": ylabel,
        "n_used": int(np.count_nonzero(mask)),
        "projection_method": method,
        "velocity_projection_method": velocity_method,
        "kernel_n_deposited": kernel_n_deposited,
        "los_half_thickness_kpc": None if los_half_thickness_kpc is None else float(los_half_thickness_kpc),
    }


def load_projection_arrays(cfg: Config, ds, code_cfg: Dict[str, Any], arrays: Dict[str, Any], center: np.ndarray, R_faceon: np.ndarray,
                           max_elements: int, bulk_velocity_original: Optional[np.ndarray] = None):
    ad = ds.all_data()
    gas_ftype = arrays["gas_ftype"]
    star_ftype = arrays["star_ftype"]
    gas = {"pos": None, "vel": None, "mass": None, "temp": None, "hsml": None, "hsml_field": None}
    stars = {"pos": None, "vel": None, "mass": None}
    try:
        gas["pos"] = get_positions(ad, ds, gas_ftype)
        gas["vel"] = get_velocities(ad, ds, gas_ftype, code_cfg)
        gas["mass"] = get_mass(ad, ds, gas_ftype, code_cfg.get("gas_mass_names", ["cell_mass", "Masses", "Mass"]))
        gas["temp"], _ = get_temperature(ad, ds, [gas_ftype, "gas"], code_cfg.get("temperature_names", ["temperature"]))
        gas["hsml"], gas["hsml_field"] = get_smoothing_length(
            ad, ds, gas_ftype, code_cfg.get("smoothing_length_names", ["SmoothingLength", "Smoothing_Length", "smoothing_length", "SmoothingLengths", "SubfindHsml", "hsml", "H_sml"])
        )
        if gas["hsml_field"] is not None:
            print("Projection gas smoothing length field used:", gas["hsml_field"])
        gas["pos"], gas["vel"], gas["mass"], gas["temp"], gas["hsml"] = random_subsample(
            gas["pos"], gas["vel"], gas["mass"], gas["temp"], gas["hsml"], max_n=max_elements, seed=cfg.seed + 10
        )
        gas["pos"] = original_to_faceon(gas["pos"], center, R_faceon)
        if gas["vel"] is not None:
            if bulk_velocity_original is not None:
                gas["vel"] = gas["vel"] - bulk_velocity_original[None, :]
            gas["vel"] = (R_faceon @ gas["vel"].T).T
    except Exception as exc:
        print("WARNING: cannot load gas projection arrays:", exc)
    if normalize_code(cfg.code) == "ENZO":
        try:
            stars["pos"], stars["vel"], stars["mass"], enzo_star_ftype = load_enzo_stellar_particle_arrays(
                ad, ds, code_cfg, max_elements, cfg.seed + 11
            )
            if stars["pos"] is not None:
                print(f"Loaded Enzo stellar projection elements: {len(stars['pos'])} from {enzo_star_ftype} particle_type==2")
                stars["pos"] = original_to_faceon(stars["pos"], center, R_faceon)
                if stars["vel"] is not None:
                    if bulk_velocity_original is not None:
                        stars["vel"] = stars["vel"] - bulk_velocity_original[None, :]
                    stars["vel"] = (R_faceon @ stars["vel"].T).T
        except Exception as exc:
            print("WARNING: cannot load Enzo stellar projection arrays:", exc)
    elif star_ftype is not None:
        try:
            stars["pos"] = get_positions(ad, ds, star_ftype)
            stars["vel"] = get_velocities(ad, ds, star_ftype, code_cfg)
            stars["mass"] = get_mass(ad, ds, star_ftype, code_cfg.get("star_mass_names", ["particle_mass", "Masses", "Mass"]))
            stars["pos"], stars["vel"], stars["mass"] = random_subsample(
                stars["pos"], stars["vel"], stars["mass"], max_n=max_elements, seed=cfg.seed + 11
            )
            stars["pos"] = original_to_faceon(stars["pos"], center, R_faceon)
            if stars["vel"] is not None:
                if bulk_velocity_original is not None:
                    stars["vel"] = stars["vel"] - bulk_velocity_original[None, :]
                stars["vel"] = (R_faceon @ stars["vel"].T).T
        except Exception as exc:
            print("WARNING: cannot load star projection arrays:", exc)
    return gas, stars


def plot_stacked_quantity(fig_path: Path, maps_face, maps_edge, quantity: str, title: str, cmap: str, log10: bool = True,
                          cbar_label: str = "", pmin: float = 5, pmax: float = 99):
    ensure_matplotlib()
    arrs = [maps_face[quantity], maps_edge[quantity]]
    plot_arrs = []
    for arr in arrs:
        if arr is None:
            plot_arrs.append(None)
        elif log10:
            plot_arrs.append(np.log10(np.where(arr > 0, arr, np.nan)))
        else:
            plot_arrs.append(arr)
    finite_chunks = [a[np.isfinite(a)].ravel() for a in plot_arrs if a is not None and np.any(np.isfinite(a))]
    vals = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    vmin, vmax = (None, None) if vals.size == 0 else (np.nanpercentile(vals, pmin), np.nanpercentile(vals, pmax))
    if vals.size == 0:
        print(f"WARNING: {title} projection has no finite {quantity} values; writing placeholder map: {fig_path}")
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 11.0), constrained_layout=True)
    for ax, maps, arr, view in zip(axes, [maps_face, maps_edge], plot_arrs, ["face-on", "edge-on"]):
        im = ax.imshow(arr, origin="lower", extent=maps["extent"], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title} ({view}), N={maps['n_used']}")
        ax.set_xlabel(maps["xlabel"])
        ax.set_ylabel(maps["ylabel"])
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_velocity_stacked(fig_path: Path, maps_face, maps_edge, quiver_step: int = 10,
                          quiver_scale: float = 1800.0, quiver_width: float = 0.0022,
                          quiver_alpha: float = 0.75):
    ensure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 11.0), constrained_layout=True)
    for ax, maps, view in zip(axes, [maps_face, maps_edge], ["face-on", "edge-on"]):
        bg = np.log10(np.where(maps["sigma"] > 0, maps["sigma"], np.nan))
        vlos = maps["vlos"]
        vmax = np.nanpercentile(np.abs(vlos[np.isfinite(vlos)]), 95) if vlos is not None and np.any(np.isfinite(vlos)) else 1.0
        ax.imshow(bg, origin="lower", extent=maps["extent"], cmap="Greys", interpolation="nearest", alpha=0.45)
        im = ax.imshow(vlos, origin="lower", extent=maps["extent"], cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax), interpolation="nearest", alpha=0.85)
        if maps["vx"] is not None and maps["vy"] is not None:
            ny, nx = maps["vx"].shape
            xs = np.linspace(maps["extent"][0], maps["extent"][1], nx)
            ys = np.linspace(maps["extent"][2], maps["extent"][3], ny)
            X, Y = np.meshgrid(xs, ys)
            sl = slice(None, None, quiver_step)
            ax.quiver(X[sl, sl], Y[sl, sl], maps["vx"][sl, sl], maps["vy"][sl, sl], color="black",
                      scale=quiver_scale, width=quiver_width, alpha=quiver_alpha)
        ax.set_title(f"Gas velocity field ({view}), N={maps['n_used']}")
        ax.set_xlabel(maps["xlabel"])
        ax.set_ylabel(maps["ylabel"])
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="LOS velocity [km/s]")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def save_projection_map_data(outdir: Path, gas_face: Dict[str, Any], gas_edge: Dict[str, Any],
                             star_face: Optional[Dict[str, Any]], star_edge: Optional[Dict[str, Any]],
                             box_kpc: float, npix: int, max_elements: int, quiver_step: int,
                             projection_los_half_thickness_kpc: Optional[float] = None,
                             projection_kernel_max_particles: int = 500_000,
                             quiver_scale: float = 1800.0, quiver_width: float = 0.0022,
                             quiver_alpha: float = 0.75,
                             bulk_velocity_original: Optional[np.ndarray] = None,
                             bulk_velocity_faceon: Optional[np.ndarray] = None):
    """Save the arrays behind the face-on/edge-on projection plots."""
    arrays_to_save: Dict[str, np.ndarray] = {
        "gas_face_sigma": np.asarray(gas_face["sigma"], dtype=np.float64),
        "gas_edge_sigma": np.asarray(gas_edge["sigma"], dtype=np.float64),
        "gas_face_extent": np.asarray(gas_face["extent"], dtype=np.float64),
        "gas_edge_extent": np.asarray(gas_edge["extent"], dtype=np.float64),
        "gas_face_n_used": np.asarray(gas_face["n_used"], dtype=np.int64),
        "gas_edge_n_used": np.asarray(gas_edge["n_used"], dtype=np.int64),
    }
    for prefix, maps in [("gas_face", gas_face), ("gas_edge", gas_edge)]:
        for key in ["temperature", "vx", "vy", "vlos"]:
            if maps.get(key) is not None:
                arrays_to_save[f"{prefix}_{key}"] = np.asarray(maps[key], dtype=np.float64)
    if star_face is not None and star_edge is not None:
        arrays_to_save.update({
            "star_face_sigma": np.asarray(star_face["sigma"], dtype=np.float64),
            "star_edge_sigma": np.asarray(star_edge["sigma"], dtype=np.float64),
            "star_face_extent": np.asarray(star_face["extent"], dtype=np.float64),
            "star_edge_extent": np.asarray(star_edge["extent"], dtype=np.float64),
            "star_face_n_used": np.asarray(star_face["n_used"], dtype=np.int64),
            "star_edge_n_used": np.asarray(star_edge["n_used"], dtype=np.int64),
        })
    np.savez_compressed(outdir / "projection_maps_face_edge.npz", **arrays_to_save)
    metadata = {
        "description": "Arrays used to make face-on/edge-on projection PNGs.",
        "box_kpc": box_kpc,
        "npix": npix,
        "max_elements": max_elements,
        "quiver_step": quiver_step,
        "projection_los_half_thickness_kpc": projection_los_half_thickness_kpc,
        "projection_los_note": "None means full-column projection. If set, face-on keeps |z_faceon| <= value and edge-on keeps |y_faceon| <= value.",
        "gas_projection_method_face": gas_face.get("projection_method"),
        "gas_projection_method_edge": gas_edge.get("projection_method"),
        "gas_kernel_max_particles": projection_kernel_max_particles,
        "gas_kernel_n_deposited_face": gas_face.get("kernel_n_deposited"),
        "gas_kernel_n_deposited_edge": gas_edge.get("kernel_n_deposited"),
        "gas_velocity_projection_method_face": gas_face.get("velocity_projection_method"),
        "gas_velocity_projection_method_edge": gas_edge.get("velocity_projection_method"),
        "quiver_scale": quiver_scale,
        "quiver_width": quiver_width,
        "quiver_alpha": quiver_alpha,
        "surface_density_units": "mass per kpc^2 in the mass units supplied by yt for the selected dataset",
        "temperature_units": "K",
        "velocity_units": "km/s",
        "velocity_reference_note": "Projection gas/star velocities have this bulk velocity subtracted before face-on rotation.",
        "bulk_velocity_original_km_s": None if bulk_velocity_original is None else np.asarray(bulk_velocity_original, dtype=float).tolist(),
        "bulk_velocity_faceon_km_s": None if bulk_velocity_faceon is None else np.asarray(bulk_velocity_faceon, dtype=float).tolist(),
        "extent_order": "[xmin, xmax, ymin, ymax] in kpc",
    }
    with open(outdir / "projection_maps_face_edge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def make_projection_products(outdir: Path, cfg: Config, ds, code_cfg: Dict[str, Any], arrays: Dict[str, Any],
                             center: np.ndarray, R_faceon: np.ndarray, box_kpc: float, npix: int,
                             max_elements: int, quiver_step: int, projection_los_half_thickness_kpc: Optional[float] = None,
                             projection_kernel_max_particles: int = 500_000,
                             quiver_scale: float = 1800.0,
                             quiver_width: float = 0.0022, quiver_alpha: float = 0.75,
                             bulk_velocity_original: Optional[np.ndarray] = None):
    gas, stars = load_projection_arrays(cfg, ds, code_cfg, arrays, center, R_faceon, max_elements, bulk_velocity_original)
    if gas["pos"] is None:
        print("WARNING: projection maps skipped because gas positions are unavailable")
        return
    use_gas_kernel = code_cfg.get("family") in {"particle", "moving_mesh"} and gas.get("hsml") is not None
    if use_gas_kernel:
        print(f"Using M6 smoothing-length kernel projection for gas maps; max deposited particles per map = {projection_kernel_max_particles}")
    gas_face = make_2d_maps(gas["pos"], mass=gas["mass"], temp=gas["temp"], vel=gas["vel"], hsml=gas.get("hsml"),
                            view="face-on", box_kpc=box_kpc, npix=npix,
                            los_half_thickness_kpc=projection_los_half_thickness_kpc, use_kernel=use_gas_kernel,
                            kernel_max_particles=projection_kernel_max_particles, kernel_seed=cfg.seed + 20)
    gas_edge = make_2d_maps(gas["pos"], mass=gas["mass"], temp=gas["temp"], vel=gas["vel"], hsml=gas.get("hsml"),
                            view="edge-on", box_kpc=box_kpc, npix=npix,
                            los_half_thickness_kpc=projection_los_half_thickness_kpc, use_kernel=use_gas_kernel,
                            kernel_max_particles=projection_kernel_max_particles, kernel_seed=cfg.seed + 21)
    plot_stacked_quantity(outdir / "projection_gas_surface_density_face_edge.png", gas_face, gas_edge, "sigma", "Gas surface density", "magma", True, "log gas surface density")
    if gas_face["temperature"] is not None:
        plot_stacked_quantity(outdir / "projection_gas_temperature_face_edge.png", gas_face, gas_edge, "temperature", "Mass-weighted gas temperature", "turbo", True, "log T [K]", pmin=2, pmax=98)
    if gas_face["vx"] is not None:
        plot_velocity_stacked(outdir / "projection_gas_velocity_face_edge.png", gas_face, gas_edge,
                              quiver_step=quiver_step, quiver_scale=quiver_scale,
                              quiver_width=quiver_width, quiver_alpha=quiver_alpha)
    star_face = star_edge = None
    if stars["pos"] is not None:
        star_face = make_2d_maps(stars["pos"], mass=stars["mass"], view="face-on", box_kpc=box_kpc, npix=npix,
                                 los_half_thickness_kpc=projection_los_half_thickness_kpc)
        star_edge = make_2d_maps(stars["pos"], mass=stars["mass"], view="edge-on", box_kpc=box_kpc, npix=npix,
                                 los_half_thickness_kpc=projection_los_half_thickness_kpc)
        plot_stacked_quantity(outdir / "projection_stellar_surface_density_face_edge.png", star_face, star_edge, "sigma", "Stellar surface density", "inferno", True, "log stellar surface density")
    bulk_faceon = None if bulk_velocity_original is None else (R_faceon @ np.asarray(bulk_velocity_original, dtype=float))
    save_projection_map_data(outdir, gas_face, gas_edge, star_face, star_edge, box_kpc, npix, max_elements,
                             quiver_step, projection_los_half_thickness_kpc, projection_kernel_max_particles,
                             quiver_scale, quiver_width, quiver_alpha,
                             bulk_velocity_original=bulk_velocity_original, bulk_velocity_faceon=bulk_faceon)


def plot_mollweide(data: Dict[str, np.ndarray], outdir: Path, prefix: str = "MWlike_4pi"):
    ensure_matplotlib()
    lon = np.deg2rad(np.asarray(data["l_deg"]) - 180.0)
    lat = np.deg2rad(np.asarray(data["b_deg"]))
    for key, label in [
        ("DM_total_pc_cm3", r"DM [pc cm$^{-3}$]"),
        ("EM_ne2_total_pc_cm6", r"EM [pc cm$^{-6}$]"),
        ("EM_ne2_hot_pc_cm6", r"Hot EM [pc cm$^{-6}$]"),
    ]:
        if key not in data:
            continue
        vals = np.asarray(data[key], dtype=np.float64)
        positive = vals[np.isfinite(vals) & (vals > 0)]
        fig = plt.figure(figsize=(10, 5.2), constrained_layout=True)
        ax = fig.add_subplot(111, projection="mollweide")
        if positive.size > 0:
            norm = LogNorm(vmin=np.nanpercentile(positive, 2), vmax=np.nanpercentile(positive, 98))
        else:
            norm = None
        sc = ax.scatter(lon, lat, c=vals, s=14, cmap="viridis" if key.startswith("DM") else "magma", norm=norm)
        ax.grid(True, alpha=0.35)
        ax.set_title(key)
        fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, label=label)
        fig.savefig(outdir / f"{prefix}_{key}_mollweide.png", dpi=180)
        plt.close(fig)


def plot_hot_em_diagnostics(data: Dict[str, np.ndarray], outdir: Path, observer_faceon: np.ndarray,
                            hot_Tmin_K: float, prefix: str = "MWlike_4pi"):
    ensure_matplotlib()
    if "angle_from_galactic_center_deg" not in data:
        add_observer_angle_columns(data, observer_faceon)
    angle = np.asarray(data["angle_from_galactic_center_deg"], dtype=np.float64)
    em = np.asarray(data["EM_ne2_hot_pc_cm6"], dtype=np.float64)
    em_obs_unit = 1.0e3 * em  # plotted as 10^-3 cm^-6 pc, matching common X-ray EM figures.
    ok = np.isfinite(angle) & np.isfinite(em_obs_unit) & (em_obs_unit >= 0)
    angle = angle[ok]
    em_obs_unit = em_obs_unit[ok]

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.scatter(angle, em_obs_unit, s=22, color="#d99a2b", alpha=0.75, edgecolor="none", label="LOS samples")
    if len(angle) >= 8:
        bins = np.linspace(0.0, 180.0, 19)
        centers = 0.5 * (bins[:-1] + bins[1:])
        med = np.full(len(centers), np.nan)
        lo = np.full(len(centers), np.nan)
        hi = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            m = (angle >= bins[i]) & (angle < bins[i + 1])
            if np.count_nonzero(m) > 0:
                med[i] = np.nanmedian(em_obs_unit[m])
                lo[i] = np.nanpercentile(em_obs_unit[m], 16)
                hi[i] = np.nanpercentile(em_obs_unit[m], 84)
        good = np.isfinite(med)
        ax.plot(centers[good], med[good], color="black", lw=2.0, label="binned median")
        ax.fill_between(centers[good], lo[good], hi[good], color="black", alpha=0.18, linewidth=0, label="16-84%")
    ax.set_xlabel("Angle from Galactic centre [deg]")
    ax.set_ylabel(r"Hot EM [$10^{-3}$ cm$^{-6}$ pc]")
    ax.set_title(rf"Hot-gas EM profile ($T \geq {hot_Tmin_K:.1e}$ K)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(outdir / f"{prefix}_hot_EM_vs_GC_angle.png", dpi=180)
    plt.close(fig)

    pa = sky_position_angle_about_gc(data, observer_faceon)[ok]
    radius = angle
    positive = em_obs_unit[np.isfinite(em_obs_unit) & (em_obs_unit > 0)]
    fig = plt.figure(figsize=(7.0, 6.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    if positive.size > 0:
        vmin = max(np.nanpercentile(positive, 2), 1.0e-12)
        vmax = max(np.nanpercentile(positive, 98), vmin * 1.01)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    sc = ax.scatter(pa, radius, c=em_obs_unit, s=62, cmap="YlOrRd", norm=norm,
                    edgecolor="0.25", linewidth=0.35)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 180)
    ax.set_rticks([30, 60, 90, 120, 150, 180])
    ax.set_title(rf"Hot EM sky view around Galactic centre ($T \geq {hot_Tmin_K:.1e}$ K)")
    fig.colorbar(sc, ax=ax, pad=0.08, label=r"Hot EM [$10^{-3}$ cm$^{-6}$ pc]")
    fig.savefig(outdir / f"{prefix}_hot_EM_GC_polar.png", dpi=180)
    plt.close(fig)


def special_observer_dirs(phi_deg: float):
    phi = np.deg2rad(phi_deg)
    radial_out = np.array([np.cos(phi), np.sin(phi), 0.0], dtype=np.float64)
    to_center = -radial_out
    anti_center = radial_out
    vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dirs = np.vstack([to_center, anti_center, vertical])
    labels = ["toward_center_in_plane", "anti_center_in_plane", "vertical_to_disk"]
    return labels, dirs


def run_random_observer_test(cfg: Config, backend: str, code_cfg: Dict[str, Any], ds, center: np.ndarray, R_faceon: np.ndarray,
                             gas_pos_faceon: Optional[np.ndarray], ne: Optional[np.ndarray], nH: Optional[np.ndarray], temp: Optional[np.ndarray],
                             rho_field: Optional[Tuple[str, str]], temp_field: Optional[Tuple[str, str]], n_observers: int,
                             n_jobs: int, seed: int, outdir: Path):
    rows = []
    if n_observers <= 0:
        return rows
    rng = np.random.default_rng(seed)
    phis = rng.uniform(0.0, 360.0, size=n_observers)
    for obs_id, phi in enumerate(phis):
        labels, dirs = special_observer_dirs(phi)
        local_cfg = Config(**asdict(cfg))
        local_cfg.phi_sun_deg = float(phi)
        local_cfg.n_los = len(labels)
        observer = observer_position_faceon(local_cfg)
        if backend == "particle":
            data = integrate_particle_sightlines_parallel(local_cfg, gas_pos_faceon, ne, nH, temp, observer, dirs, n_jobs=n_jobs, los_chunk=len(labels),
                                                        gas_mass=getattr(cfg, "_gas_mass_full", None), gas_rho=getattr(cfg, "_gas_rho_full", None), gas_hsml=getattr(cfg, "_gas_hsml_full", None))
        else:
            data = integrate_gridray_sightlines_parallel(local_cfg, ds, code_cfg, center, R_faceon, observer, dirs, rho_field, temp_field, n_jobs=n_jobs)
        for i, label in enumerate(labels):
            row = {"observer_id": obs_id, "phi_sun_deg": float(phi), "los_type": label}
            for key, arr in data.items():
                if key not in {"los_id", "l_deg", "b_deg", "dir_x_faceon", "dir_y_faceon", "dir_z_faceon"}:
                    row[key] = float(arr[i])
            rows.append(row)
    if rows:
        csv_path = outdir / "random_observer_special_los_DM_EM.csv"
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        summary = {}
        for los_type in sorted(set(r["los_type"] for r in rows)):
            summary[los_type] = {}
            subset = [r for r in rows if r["los_type"] == los_type]
            for key in ["DM_total_pc_cm3", "EM_ne2_total_pc_cm6", "EM_ne_nH_total_pc_cm6"]:
                vals = np.array([r[key] for r in subset], dtype=np.float64)
                summary[los_type][key] = {
                    "median": float(np.nanmedian(vals)),
                    "p16": float(np.nanpercentile(vals, 16)),
                    "p84": float(np.nanpercentile(vals, 84)),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                }
        with (outdir / "random_observer_special_los_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
    return rows


def make_parser() -> argparse.ArgumentParser:
    p = _base_make_parser()
    p.description = "Parallel AGORA DM/EM sightline, projection, Mollweide, and random-observer pipeline"
    p.add_argument("--n-jobs", type=int, default=1, help="joblib threads for LOS integration. Use 4-32 on a CPU server after testing memory.")
    p.add_argument("--parallel-los-chunk", type=int, default=None, help="LOS chunk size for particle backend parallel jobs.")
    p.add_argument("--make-projections", action="store_true", help="Write face-on/edge-on gas, star, temperature, and velocity projection PNGs.")
    p.add_argument("--projection-box-kpc", type=float, default=60.0)
    p.add_argument("--projection-npix", type=int, default=256)
    p.add_argument("--projection-max-elements", type=int, default=200_000_000)
    p.add_argument("--projection-quiver-step", type=int, default=10)
    p.add_argument("--projection-los-half-thickness-kpc", type=float, default=None,
                   help="Optional half-thickness along projection LOS. None/full column; face-on cuts |z_faceon| and edge-on cuts |y_faceon|.")
    p.add_argument("--projection-kernel-max-particles", type=int, default=500_000,
                   help="Maximum selected gas particles deposited with smoothing-length kernel per projection map; larger inputs are randomly subsampled with mass renormalization.")
    p.add_argument("--projection-quiver-scale", type=float, default=1800.0)
    p.add_argument("--projection-quiver-width", type=float, default=0.0022)
    p.add_argument("--projection-quiver-alpha", type=float, default=0.75)
    p.add_argument("--make-mollweide", action="store_true", help="Write DM and EM Mollweide PNGs from the all-sky LOS samples.")
    p.add_argument("--make-hot-em-diagnostics", action="store_true",
                   help="Write T>hot_Tmin hot EM profile versus Galactic-center angle and a polar sky view.")
    p.add_argument("--random-observers", type=int, default=0, help="Number of random solar azimuths at fixed R_sun for three special LOS tests.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = make_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring unknown arguments:", unknown)
    if args.projection_los_half_thickness_kpc is not None and args.projection_los_half_thickness_kpc <= 0:
        parser.error("--projection-los-half-thickness-kpc must be positive when provided")
    cfg = _base_config_from_args(args)
    cfg.projection_los_half_thickness_kpc = args.projection_los_half_thickness_kpc
    code = normalize_code(cfg.code)
    if code == "AUTO":
        code = infer_code_from_snapshot_path(cfg.snapshot)
        print("Inferred code from snapshot path:", code)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    ds = load_dataset(
        cfg.snapshot, code, cfg.unit_base,
        cfg.tipsy_length_unit_kpc, cfg.tipsy_mass_unit_msun, cfg.tipsy_time_unit_s,
    )
    validate_dataset_units(ds, code, cfg)
    code_cfg = AGORA_CODE_CONFIG.get(code, default_config_for_unknown(code, ds))
    print("Normalized code:", code)
    print("Code family:", code_cfg["family"], "(", code_backend_family(code_cfg), ")")
    print("joblib n_jobs:", args.n_jobs)
    add_common_derived_fields(ds, code, code_cfg)

    arrays = load_geometry_arrays(cfg, ds, code_cfg)
    cfg = apply_dataset_specific_geometry_fallbacks(cfg, code, arrays)
    center = determine_center(cfg, arrays["gas_pos"], arrays["gas_mass"], arrays["gas_rho"], arrays["star_pos"], arrays["star_mass"])
    disk_normal = compute_angular_momentum_normal(
        cfg,
        arrays["gas_pos"], arrays["gas_vel"], arrays["gas_mass"], arrays["gas_temp"],
        arrays["star_pos"], arrays["star_vel"], arrays["star_mass"],
        center,
    )
    bulk_velocity_original = compute_bulk_velocity(
        cfg,
        arrays["gas_pos"], arrays["gas_vel"], arrays["gas_mass"],
        arrays["star_pos"], arrays["star_vel"], arrays["star_mass"],
        center,
    )
    R_faceon = rotation_matrix_from_vectors(disk_normal, np.array([0.0, 0.0, 1.0]))
    bulk_velocity_faceon = R_faceon @ bulk_velocity_original
    observer_faceon = observer_position_faceon(cfg)
    dirs_faceon = fibonacci_sphere(cfg.n_los, cfg.random_rotate_directions, cfg.seed)
    backend = cfg.integration_backend
    if backend == "auto":
        backend = "gridray" if code_cfg["family"] == "grid" else "particle"
    print("Center [kpc, original frame] =", center)
    print("Disk normal [original frame] =", disk_normal)
    print("Bulk velocity [km/s, original frame] =", bulk_velocity_original)
    print("Bulk velocity [km/s, face-on frame] =", bulk_velocity_faceon)
    print("Observer [kpc, face-on frame] =", observer_faceon)
    print("Integration backend:", backend)

    ne_field_used = None
    gas_pos_faceon = ne = nH = temp = None
    gas_mass_full = gas_hsml = None
    rho_field = arrays["rho_field"]
    temp_field = arrays["temp_field"]
    if backend == "particle":
        gas_pos, rho, temp, ne, nH, rho_field, temp_field, ne_field_used, gas_mass_full, gas_hsml, hsml_field = load_full_particle_gas_for_backend(cfg, ds, code_cfg, arrays["gas_ftype"])
        gas_pos_faceon = original_to_faceon(gas_pos, center, R_faceon)
        print(f"Full gas elements for particle backend: {len(gas_pos_faceon)}")
        print("Smoothing length field used:", hsml_field)
        print("Particle interpolation:", cfg.particle_interpolation)
        cfg._gas_mass_full = gas_mass_full
        cfg._gas_rho_full = rho
        cfg._gas_hsml_full = gas_hsml
        data = integrate_particle_sightlines_parallel(cfg, gas_pos_faceon, ne, nH, temp, observer_faceon, dirs_faceon,
                                                      n_jobs=args.n_jobs, los_chunk=args.parallel_los_chunk,
                                                      gas_mass=gas_mass_full, gas_rho=rho, gas_hsml=gas_hsml)
    elif backend == "gridray":
        if rho_field is None:
            rho_field = first_existing_field(ds, code_cfg.get("gas_types", ["gas"]), code_cfg.get("density_names", ["density"]))
        if temp_field is None:
            temp_field = first_existing_field(ds, code_cfg.get("gas_types", ["gas"]), code_cfg.get("temperature_names", ["temperature"]))
        if rho_field is None:
            raise RuntimeError("gridray backend requires a gas density field.")
        data = integrate_gridray_sightlines_parallel(cfg, ds, code_cfg, center, R_faceon, observer_faceon, dirs_faceon,
                                                     rho_field, temp_field, n_jobs=args.n_jobs)
    else:
        raise ValueError(f"Unknown integration backend: {backend}")
    add_observer_angle_columns(data, observer_faceon)

    metadata = {
        "config": asdict(cfg),
        "dataset": str(ds),
        "dataset_units": dataset_unit_metadata(ds),
        "normalized_code": code,
        "code_family": code_cfg["family"],
        "backend_family_label": code_backend_family(code_cfg),
        "integration_backend": backend,
        "n_jobs": args.n_jobs,
        "gas_ftype": arrays["gas_ftype"],
        "star_ftype": arrays["star_ftype"],
        "density_field": str(rho_field),
        "temperature_field": str(temp_field),
        "electron_field_used_particle_backend": str(ne_field_used),
        "smoothing_length_field_used_particle_backend": str(locals().get("hsml_field", None)),
        "particle_interpolation": cfg.particle_interpolation,
        "center_kpc_original_frame": center.tolist(),
        "disk_normal_original_frame": disk_normal.tolist(),
        "velocity_reference_source": cfg.velocity_reference_source,
        "velocity_reference_radius_kpc": cfg.velocity_reference_radius_kpc,
        "bulk_velocity_original_km_s": bulk_velocity_original.tolist(),
        "bulk_velocity_faceon_km_s": bulk_velocity_faceon.tolist(),
        "rotation_matrix_original_to_faceon": R_faceon.tolist(),
        "observer_kpc_faceon_frame": observer_faceon.tolist(),
        "wall_time_min_before_output": (time.time() - t_start) / 60.0,
        "notes": {
            "DM_units": "pc cm^-3",
            "EM_units": "pc cm^-6",
            "faceon_frame": "x-y is disk plane and z is disk angular-momentum axis",
            "particle_backend": "joblib-threaded k-nearest inverse-distance-squared interpolation; not strict SPH kernel",
            "gridray_backend": "joblib-threaded yt arbitrary ray integration through native grid/AMR cells",
        },
    }
    save_outputs(cfg, data, metadata)
    if args.make_mollweide:
        plot_mollweide(data, outdir)
    if args.make_hot_em_diagnostics:
        plot_hot_em_diagnostics(data, outdir, observer_faceon, cfg.hot_Tmin_K)
    if args.make_projections:
        make_projection_products(outdir, cfg, ds, code_cfg, arrays, center, R_faceon,
                                 args.projection_box_kpc, args.projection_npix,
                                 args.projection_max_elements, args.projection_quiver_step,
                                 args.projection_los_half_thickness_kpc,
                                 args.projection_kernel_max_particles,
                                 args.projection_quiver_scale, args.projection_quiver_width,
                                 args.projection_quiver_alpha,
                                 bulk_velocity_original=bulk_velocity_original)
    if args.random_observers > 0:
        run_random_observer_test(cfg, backend, code_cfg, ds, center, R_faceon,
                                 gas_pos_faceon, ne, nH, temp, rho_field, temp_field,
                                 args.random_observers, args.n_jobs, cfg.seed + 100, outdir)
    metadata["total_wall_time_min"] = (time.time() - t_start) / 60.0
    with (outdir / "parallel_pipeline_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Total wall time: {metadata['total_wall_time_min']:.2f} min")


if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    main()
