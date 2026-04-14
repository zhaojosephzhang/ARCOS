#!/usr/bin/env python3
"""
Validate halo or subhalo catalogs and generate diagnostic figures.

Examples
--------
python plot_halo_catalog_validation.py \
    --input /path/to/halo_catalog_physical_020.h5 \
    --output-dir /path/to/halo_catalog_validation_020 \
    --catalog-type halo

python plot_halo_catalog_validation.py \
    --input /path/to/subhalo_catalog_physical_020.h5 \
    --output-dir /path/to/subhalo_catalog_validation_020 \
    --catalog-type subhalo \
    --mass-bin-min 8.5
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
TITLE_SIZE = 15
COMPONENT_COLORS = {
    "gas": "#4C78A8",
    "dm": "#72B7B2",
    "stellar": "#F58518",
    "bh": "#E45756",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate halo or subhalo catalog and generate summary plots."
    )
    parser.add_argument(
        "--input",
        default="/sqfs/work/hp240141/z6b340/results/halo_catalog_physical_020.h5",
        help="Input halo/subhalo catalog HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        default="/sqfs/work/hp240141/z6b340/results/halo_catalog_validation_020",
        help="Directory for figures and summaries.",
    )
    parser.add_argument(
        "--catalog-type",
        choices=["auto", "halo", "subhalo"],
        default="auto",
        help="Catalog type. auto detects from dataset names.",
    )
    parser.add_argument(
        "--mass-scale",
        type=float,
        default=1.0e10,
        help="Scale factor converting catalog masses to Msun.",
    )
    parser.add_argument(
        "--mass-bin-min",
        type=float,
        default=None,
        help="Minimum log10 mass bin edge in Msun. Defaults to 9.0 for halo and 8.5 for subhalo.",
    )
    parser.add_argument(
        "--mass-bin-max",
        type=float,
        default=15.0,
        help="Maximum log10 mass bin edge in Msun.",
    )
    parser.add_argument(
        "--mass-bin-step",
        type=float,
        default=0.5,
        help="Mass bin width in dex.",
    )
    parser.add_argument(
        "--max-points-3d",
        type=int,
        default=200000,
        help="Maximum number of objects plotted in the 3D scatter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20,
        help="Random seed used for 3D down-sampling.",
    )
    return parser.parse_args()


def detect_catalog_type(handle):
    if "GroupPos" in handle and "GroupMass" in handle:
        return "halo"
    if "SubhaloPos" in handle and "SubhaloMass" in handle:
        return "subhalo"
    raise ValueError("Could not infer catalog type from HDF5 datasets.")


def get_mass_bin_min(args, catalog_type):
    if args.mass_bin_min is not None:
        return args.mass_bin_min
    if catalog_type == "subhalo":
        return 8.5
    return 9.0


def build_mass_bins(args, catalog_type):
    mass_bin_min = get_mass_bin_min(args, catalog_type)
    return np.arange(mass_bin_min, args.mass_bin_max + args.mass_bin_step, args.mass_bin_step)


def build_bin_labels(log_edges):
    labels = []
    for left, right in zip(log_edges[:-1], log_edges[1:]):
        labels.append(rf"$10^{{{left:.1f}}}$-$10^{{{right:.1f}}}$")
    return labels


def summarize_vector(name, values):
    finite = np.isfinite(values)
    return {
        "name": name,
        "size": int(values.size),
        "finite": int(finite.sum()),
        "non_positive": int(np.sum(values <= 0)),
        "minimum": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "maximum": float(np.nanmax(values)),
    }


def summarize_positions(name, positions):
    return {
        "name": name,
        "shape": positions.shape,
        "finite_all": bool(np.isfinite(positions).all()),
        "min_xyz": positions.min(axis=0),
        "max_xyz": positions.max(axis=0),
    }


def write_text_summary(path, catalog_name, scalar_summaries, position_summary, extra_lines):
    lines = [
        f"{catalog_name} catalog validation summary",
        "",
    ]
    lines.extend(extra_lines)
    lines.append("")
    for summary in scalar_summaries:
        lines.extend(
            [
                f"[{summary['name']}]",
                f"size: {summary['size']}",
                f"finite: {summary['finite']}",
                f"non_positive: {summary['non_positive']}",
                f"minimum: {summary['minimum']:.6e}",
                f"median: {summary['median']:.6e}",
                f"maximum: {summary['maximum']:.6e}",
                "",
            ]
        )
    lines.extend(
        [
            f"[{position_summary['name']}]",
            f"shape: {position_summary['shape']}",
            f"finite_all: {position_summary['finite_all']}",
            "min_xyz: {:.6e}, {:.6e}, {:.6e}".format(*position_summary["min_xyz"]),
            "max_xyz: {:.6e}, {:.6e}, {:.6e}".format(*position_summary["max_xyz"]),
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bin_csv(path, labels, number_counts, reference_counts=None, median_radius=None, fraction_dict=None):
    columns = ["mass_bin_label", "object_count"]
    if reference_counts is not None:
        columns.append("reference_count")
    if median_radius is not None:
        columns.append("median_radius_mpc")
    if fraction_dict is not None:
        columns.extend(["f_gas", "f_dm", "f_stellar", "f_bh"])

    lines = [",".join(columns) + "\n"]
    for i, label in enumerate(labels):
        row = [label, str(number_counts[i])]
        if reference_counts is not None:
            row.append(str(reference_counts[i]))
        if median_radius is not None:
            row.append("" if np.isnan(median_radius[i]) else f"{median_radius[i]:.8e}")
        if fraction_dict is not None:
            for key in ["gas", "dm", "stellar", "bh"]:
                value = fraction_dict[key][i]
                row.append("" if np.isnan(value) else f"{value:.8e}")
        lines.append(",".join(row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def plot_3d_positions(path, positions, masses_msun, max_points, seed, catalog_name):
    rng = np.random.default_rng(seed)
    num_objects = positions.shape[0]
    positive = np.isfinite(masses_msun) & (masses_msun > 0)
    valid_idx = np.where(positive)[0]
    if valid_idx.size == 0:
        raise ValueError("No positive masses available for the 3D plot.")
    if valid_idx.size > max_points:
        index = rng.choice(valid_idx, size=max_points, replace=False)
    else:
        index = valid_idx

    pos = positions[index]
    mass = masses_msun[index]
    log_mass = np.log10(mass)
    point_size = np.clip((log_mass - np.nanmin(log_mass) + 0.4) * 2.0, 0.6, 18.0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        c=log_mass,
        s=point_size,
        cmap="viridis",
        alpha=0.30,
        linewidths=0.0,
    )
    ax.set_xlabel("x [Mpc]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("y [Mpc]", fontsize=AXIS_LABEL_SIZE)
    ax.set_zlabel("z [Mpc]", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(f"{catalog_name} 3D Position Distribution", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="z", which="major", labelsize=TICK_LABEL_SIZE)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.78)
    cbar.set_label(r"$\log_{10}(M/M_\odot)$", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return int(index.size)


def plot_mass_histogram(path, log_edges, labels, total_mass_msun, catalog_name):
    valid = np.isfinite(total_mass_msun) & (total_mass_msun > 0)
    counts, _ = np.histogram(np.log10(total_mass_msun[valid]), bins=log_edges)
    centers = 0.5 * (log_edges[:-1] + log_edges[1:])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(centers, counts, width=0.42, color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.set_yscale("log")
    ax.set_ylabel(f"{catalog_name} count", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel(r"Mass bin [$M_\odot$]", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(f"{catalog_name} Number Distribution by Mass Bin", fontsize=TITLE_SIZE)
    ax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return counts


def plot_radius_median(path, log_edges, reference_mass_msun, radius_mpc, catalog_name):
    mask = np.isfinite(reference_mass_msun) & np.isfinite(radius_mpc) & (reference_mass_msun > 0) & (radius_mpc > 0)
    log_mass = np.log10(reference_mass_msun[mask])
    radius_valid = radius_mpc[mask]

    medians = np.full(log_edges.size - 1, np.nan)
    counts = np.zeros(log_edges.size - 1, dtype=int)
    centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    for i, (left, right) in enumerate(zip(log_edges[:-1], log_edges[1:])):
        in_bin = (log_mass >= left) & (log_mass < right)
        counts[i] = int(np.sum(in_bin))
        if counts[i] > 0:
            medians[i] = float(np.median(radius_valid[in_bin]))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(centers, medians, marker="o", color="#F58518", linewidth=2.0)
    ax.set_xlabel(r"$\log_{10}(M/M_\odot)$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"Median Radius [Mpc]", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(f"Median Radius vs {catalog_name} Mass", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return counts, medians


def compute_component_fractions(log_edges, total_mass_msun, component_masses):
    valid = np.isfinite(total_mass_msun) & (total_mass_msun > 0)
    for values in component_masses.values():
        valid &= np.isfinite(values)

    log_mass = np.log10(total_mass_msun[valid])
    total_valid = total_mass_msun[valid]
    component_valid = {key: values[valid] for key, values in component_masses.items()}

    fractions = {key: np.full(log_edges.size - 1, np.nan) for key in component_masses}
    centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    for i, (left, right) in enumerate(zip(log_edges[:-1], log_edges[1:])):
        in_bin = (log_mass >= left) & (log_mass < right)
        if not np.any(in_bin):
            continue
        denom = np.sum(total_valid[in_bin])
        if denom <= 0:
            continue
        for key in component_masses:
            fractions[key][i] = np.sum(component_valid[key][in_bin]) / denom
    return centers, fractions


def plot_component_fractions(path, log_edges, total_mass_msun, component_masses, catalog_name):
    centers, fractions = compute_component_fractions(log_edges, total_mass_msun, component_masses)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for key, label in [
        ("gas", r"$f_{\mathrm{gas}}$"),
        ("dm", r"$f_{\mathrm{dm}}$"),
        ("stellar", r"$f_{\mathrm{stellar}}$"),
        ("bh", r"$f_{\mathrm{BH}}$"),
    ]:
        ax.plot(
            centers,
            fractions[key],
            marker="o",
            linewidth=2.0,
            color=COMPONENT_COLORS[key],
            label=label,
        )
    ax.set_xlabel(r"$\log_{10}(M/M_\odot)$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mass fraction", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(f"{catalog_name} Mass Fractions by Mass Bin", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=TICK_LABEL_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return fractions


def load_catalog(handle, catalog_type, mass_scale):
    if catalog_type == "halo":
        return {
            "catalog_name": "Halo",
            "positions": np.asarray(handle["GroupPos"][...], dtype=np.float64),
            "total_mass": np.asarray(handle["GroupMass"][...], dtype=np.float64) * mass_scale,
            "reference_mass": np.asarray(handle["M200"][...], dtype=np.float64) * mass_scale,
            "radius": np.asarray(handle["R200"][...], dtype=np.float64),
            "components": {
                "gas": np.asarray(handle["GroupGasMass"][...], dtype=np.float64) * mass_scale,
                "dm": np.asarray(handle["GroupDMMass"][...], dtype=np.float64) * mass_scale,
                "stellar": np.asarray(handle["GroupStellarMass"][...], dtype=np.float64) * mass_scale,
                "bh": np.asarray(handle["GroupBHMass"][...], dtype=np.float64) * mass_scale,
            },
        }
    return {
        "catalog_name": "Subhalo",
        "positions": np.asarray(handle["SubhaloPos"][...], dtype=np.float64),
        "total_mass": np.asarray(handle["SubhaloMass"][...], dtype=np.float64) * mass_scale,
        "reference_mass": np.asarray(handle["SubhaloMass"][...], dtype=np.float64) * mass_scale,
        "radius": None,
        "components": {
            "gas": np.asarray(handle["SubhaloGasMass"][...], dtype=np.float64) * mass_scale,
            "dm": np.asarray(handle["SubhaloDMMass"][...], dtype=np.float64) * mass_scale,
            "stellar": np.asarray(handle["SubhaloStellarMass"][...], dtype=np.float64) * mass_scale,
            "bh": np.asarray(handle["SubhaloBHMass"][...], dtype=np.float64) * mass_scale,
        },
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as handle:
        catalog_type = detect_catalog_type(handle) if args.catalog_type == "auto" else args.catalog_type
        catalog = load_catalog(handle, catalog_type, args.mass_scale)

    log_edges = build_mass_bins(args, catalog_type)
    labels = build_bin_labels(log_edges)
    catalog_name = catalog["catalog_name"]

    scalar_summaries = [
        summarize_vector(f"{catalog_name}Mass_Msun", catalog["total_mass"]),
        summarize_vector(f"{catalog_name}GasMass_Msun", catalog["components"]["gas"]),
        summarize_vector(f"{catalog_name}DMMass_Msun", catalog["components"]["dm"]),
        summarize_vector(f"{catalog_name}StellarMass_Msun", catalog["components"]["stellar"]),
        summarize_vector(f"{catalog_name}BHMass_Msun", catalog["components"]["bh"]),
    ]
    if catalog["radius"] is not None:
        scalar_summaries.append(summarize_vector("R200_Mpc", catalog["radius"]))
        scalar_summaries.append(summarize_vector("M200_Msun", catalog["reference_mass"]))
    position_summary = summarize_positions(f"{catalog_name}Pos_Mpc", catalog["positions"])

    sampled_points = plot_3d_positions(
        output_dir / f"{catalog_type}_positions_3d.png",
        positions=catalog["positions"],
        masses_msun=catalog["total_mass"],
        max_points=args.max_points_3d,
        seed=args.seed,
        catalog_name=catalog_name,
    )
    counts_total = plot_mass_histogram(
        output_dir / f"{catalog_type}_mass_bin_counts.png",
        log_edges=log_edges,
        labels=labels,
        total_mass_msun=catalog["total_mass"],
        catalog_name=catalog_name,
    )
    fractions = plot_component_fractions(
        output_dir / f"{catalog_type}_mass_fraction_vs_mass.png",
        log_edges=log_edges,
        total_mass_msun=catalog["total_mass"],
        component_masses=catalog["components"],
        catalog_name=catalog_name,
    )

    extra_lines = [
        f"catalog_type: {catalog_type}",
        f"total_objects: {catalog['total_mass'].size}",
        f"sampled_points_for_3d_plot: {sampled_points}",
        f"mass_bin_min_log10_msun: {log_edges[0]:.2f}",
        f"mass_bin_max_log10_msun: {log_edges[-1]:.2f}",
    ]

    radius_counts = None
    radius_medians = None
    if catalog["radius"] is not None:
        radius_counts, radius_medians = plot_radius_median(
            output_dir / "r200_median_vs_m200.png",
            log_edges=log_edges,
            reference_mass_msun=catalog["reference_mass"],
            radius_mpc=catalog["radius"],
            catalog_name=catalog_name,
        )
        valid_pairs = int(
            np.sum(
                (catalog["reference_mass"] > 0)
                & (catalog["radius"] > 0)
                & np.isfinite(catalog["reference_mass"])
                & np.isfinite(catalog["radius"])
            )
        )
        extra_lines.append(f"valid_mass_radius_pairs: {valid_pairs}")

    write_text_summary(
        output_dir / f"{catalog_type}_catalog_validation_summary.txt",
        catalog_name=catalog_name,
        scalar_summaries=scalar_summaries,
        position_summary=position_summary,
        extra_lines=extra_lines,
    )
    write_bin_csv(
        output_dir / f"{catalog_type}_mass_bin_summary.csv",
        labels=labels,
        number_counts=counts_total,
        reference_counts=radius_counts,
        median_radius=radius_medians,
        fraction_dict=fractions,
    )


if __name__ == "__main__":
    main()
