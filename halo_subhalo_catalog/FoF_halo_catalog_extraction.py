import os
import h5py
import numpy as np

PARTICLE_TYPE_INDEX = {
    "Gas": 0,
    "DM": 1,
    "Stars": 4,
    "BH": 5,
}


def get_header_info(snapshot_base):
    """
    从 snapshot header 读取 a 和 h
    snapshot_base 例子:
    /path/to/snapdir_020/snapshot_020
    """
    try:
        fname = f"{snapshot_base}.hdf5"
        f = h5py.File(fname, "r")
    except OSError:
        fname = f"{snapshot_base}.0.hdf5"
        f = h5py.File(fname, "r")

    with f:
        a = f["Header"].attrs["Time"]
        try:
            h = f["Header"].attrs["HubbleParam"]
        except KeyError:
            h = f["Parameters"].attrs["HubbleParam"]

    return a, h


def get_num_fof_files(group_base):
    """
    自动识别 fof catalog 是单文件还是多文件
    group_base 例子:
    /path/to/groups_020/fof_subhalo_tab_020
    """
    try:
        fname = f"{group_base}.hdf5"
        with h5py.File(fname, "r") as f:
            return 1
    except OSError:
        fname = f"{group_base}.0.hdf5"
        with h5py.File(fname, "r") as f:
            try:
                return int(f["Header"].attrs["NumFilesPerSnapshot"])
            except KeyError:
                return int(f["Header"].attrs["NumFiles"])


def apply_scaling(arr, ds, key, a, h):
    """
    优先使用 dataset 自带的 a_scaling / h_scaling
    如果没有，就使用你原代码中的 fallback
    """
    fallback_a_scaling = {
        "halomass": 0,
        "halopos": 1,
        "haloRV": 1,
        "haloMV": 0,
        "subhalomass": 0,
        "subhalopos": 1,
    }

    fallback_h_scaling = {
        "halomass": -1,
        "halopos": -1,
        "haloRV": -1,
        "haloMV": -1,
        "subhalomass": -1,
        "subhalopos": -1,
    }

    a_exp = ds.attrs.get("a_scaling", fallback_a_scaling[key])
    h_exp = ds.attrs.get("h_scaling", fallback_h_scaling[key])

    arr = np.asarray(arr, dtype=np.float64)
    arr *= a ** float(a_exp)
    arr *= h ** float(h_exp)
    return arr


def write_catalog(path, datasets, a, h, mass_unit="1e10 M_sun", length_unit="Mpc"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as hf:
        hf.attrs["Time"] = a
        hf.attrs["HubbleParam"] = h
        hf.attrs["MassUnit"] = mass_unit
        hf.attrs["LengthUnit"] = length_unit

        for name, values in datasets.items():
            ds = hf.create_dataset(name, data=values["data"])
            for attr_name, attr_value in values.get("attrs", {}).items():
                ds.attrs[attr_name] = attr_value


def extract_mass_components(ds, key, a, h):
    mass_type = apply_scaling(ds[:], ds, key, a, h)
    return {
        "GasMass": mass_type[:, PARTICLE_TYPE_INDEX["Gas"]],
        "DMMass": mass_type[:, PARTICLE_TYPE_INDEX["DM"]],
        "StellarMass": mass_type[:, PARTICLE_TYPE_INDEX["Stars"]],
        "BHMass": mass_type[:, PARTICLE_TYPE_INDEX["BH"]],
    }


def extract_halo_catalog(snapshot_base, group_base, save_path=None):
    """
    提取 halo catalog 并恢复到物理单位

    返回:
    {
        "a": a,
        "h": h,
        "GroupMass": ...,
        "M200": ...,
        "R200": ...,
        "GroupPos": ...
    }
    """
    a, h = get_header_info(snapshot_base)
    nfile = get_num_fof_files(group_base)

    group_mass_all = []
    m200_all = []
    r200_all = []
    group_pos_all = []
    group_gas_mass_all = []
    group_dm_mass_all = []
    group_stellar_mass_all = []
    group_bh_mass_all = []

    for i in range(nfile):
        if nfile == 1:
            fname = f"{group_base}.hdf5"
        else:
            fname = f"{group_base}.{i}.hdf5"

        with h5py.File(fname, "r") as f:
            # GroupMass
            ds = f["Group/GroupMass"]
            arr = apply_scaling(ds[:], ds, "halomass", a, h)
            group_mass_all.append(arr)

            # M200 = Group_M_Crit200
            ds = f["Group/Group_M_Crit200"]
            arr = apply_scaling(ds[:], ds, "haloMV", a, h)
            m200_all.append(arr)

            # R200 = Group_R_Crit200
            ds = f["Group/Group_R_Crit200"]
            arr = apply_scaling(ds[:], ds, "haloRV", a, h)
            r200_all.append(arr)

            # GroupPos（可选，但通常建议保留）
            ds = f["Group/GroupPos"]
            arr = apply_scaling(ds[:], ds, "halopos", a, h)
            group_pos_all.append(arr)

            ds = f["Group/GroupMassType"]
            mass_components = extract_mass_components(ds, "halomass", a, h)
            group_gas_mass_all.append(mass_components["GasMass"])
            group_dm_mass_all.append(mass_components["DMMass"])
            group_stellar_mass_all.append(mass_components["StellarMass"])
            group_bh_mass_all.append(mass_components["BHMass"])

    result = {
        "a": a,
        "h": h,
        "GroupMass": np.concatenate(group_mass_all),
        "M200": np.concatenate(m200_all),
        "R200": np.concatenate(r200_all),
        "GroupPos": np.concatenate(group_pos_all, axis=0),
        "GroupGasMass": np.concatenate(group_gas_mass_all),
        "GroupDMMass": np.concatenate(group_dm_mass_all),
        "GroupStellarMass": np.concatenate(group_stellar_mass_all),
        "GroupBHMass": np.concatenate(group_bh_mass_all),
    }

    if save_path is not None:
        write_catalog(
            save_path,
            datasets={
                "GroupMass": {
                    "data": result["GroupMass"],
                    "attrs": {"unit": "1e10 M_sun"},
                },
                "M200": {
                    "data": result["M200"],
                    "attrs": {"unit": "1e10 M_sun"},
                },
                "R200": {
                    "data": result["R200"],
                    "attrs": {"unit": "Mpc"},
                },
                "GroupPos": {
                    "data": result["GroupPos"],
                    "attrs": {"unit": "Mpc"},
                },
                "GroupGasMass": {
                    "data": result["GroupGasMass"],
                    "attrs": {"unit": "1e10 M_sun", "description": "Extracted from GroupMassType[:, 0]"},
                },
                "GroupDMMass": {
                    "data": result["GroupDMMass"],
                    "attrs": {"unit": "1e10 M_sun", "description": "Extracted from GroupMassType[:, 1]"},
                },
                "GroupStellarMass": {
                    "data": result["GroupStellarMass"],
                    "attrs": {"unit": "1e10 M_sun", "description": "Extracted from GroupMassType[:, 4]"},
                },
                "GroupBHMass": {
                    "data": result["GroupBHMass"],
                    "attrs": {"unit": "1e10 M_sun", "description": "Extracted from GroupMassType[:, 5]"},
                },
            },
            a=a,
            h=h,
        )

    return result


def extract_subhalo_catalog(snapshot_base, group_base, save_path=None):
    """
    提取 subhalo catalog 并恢复到物理单位。

    默认使用 SubhaloMassType[:, 4] 作为 stellar mass。
    Gadget 常用粒子类型顺序为:
    0=gas, 1=DM, 2/3=保留, 4=stars, 5=BH
    """
    a, h = get_header_info(snapshot_base)
    nfile = get_num_fof_files(group_base)

    subhalo_pos_all = []
    subhalo_mass_all = []
    subhalo_groupnr_all = []
    subhalo_gas_mass_all = []
    subhalo_dm_mass_all = []
    subhalo_stellar_mass_all = []
    subhalo_bh_mass_all = []

    for i in range(nfile):
        if nfile == 1:
            fname = f"{group_base}.hdf5"
        else:
            fname = f"{group_base}.{i}.hdf5"

        with h5py.File(fname, "r") as f:
            ds = f["Subhalo/SubhaloPos"]
            arr = apply_scaling(ds[:], ds, "subhalopos", a, h)
            subhalo_pos_all.append(arr)

            ds = f["Subhalo/SubhaloMass"]
            arr = apply_scaling(ds[:], ds, "subhalomass", a, h)
            subhalo_mass_all.append(arr)

            ds = f["Subhalo/SubhaloMassType"]
            mass_components = extract_mass_components(ds, "subhalomass", a, h)
            subhalo_gas_mass_all.append(mass_components["GasMass"])
            subhalo_dm_mass_all.append(mass_components["DMMass"])
            subhalo_stellar_mass_all.append(mass_components["StellarMass"])
            subhalo_bh_mass_all.append(mass_components["BHMass"])

            ds = f["Subhalo/SubhaloGroupNr"]
            subhalo_groupnr_all.append(np.asarray(ds[:], dtype=np.int64))

    result = {
        "a": a,
        "h": h,
        "SubhaloPos": np.concatenate(subhalo_pos_all, axis=0),
        "SubhaloMass": np.concatenate(subhalo_mass_all),
        "SubhaloGroupNr": np.concatenate(subhalo_groupnr_all),
        "SubhaloGasMass": np.concatenate(subhalo_gas_mass_all),
        "SubhaloDMMass": np.concatenate(subhalo_dm_mass_all),
        "SubhaloStellarMass": np.concatenate(subhalo_stellar_mass_all),
        "SubhaloBHMass": np.concatenate(subhalo_bh_mass_all),
    }

    if save_path is not None:
        write_catalog(
            save_path,
            datasets={
                "SubhaloPos": {
                    "data": result["SubhaloPos"],
                    "attrs": {"unit": "Mpc"},
                },
                "SubhaloMass": {
                    "data": result["SubhaloMass"],
                    "attrs": {"unit": "1e10 M_sun"},
                },
                "SubhaloGasMass": {
                    "data": result["SubhaloGasMass"],
                    "attrs": {
                        "unit": "1e10 M_sun",
                        "description": "Extracted from SubhaloMassType[:, 0]",
                    },
                },
                "SubhaloDMMass": {
                    "data": result["SubhaloDMMass"],
                    "attrs": {
                        "unit": "1e10 M_sun",
                        "description": "Extracted from SubhaloMassType[:, 1]",
                    },
                },
                "SubhaloStellarMass": {
                    "data": result["SubhaloStellarMass"],
                    "attrs": {
                        "unit": "1e10 M_sun",
                        "description": "Extracted from SubhaloMassType[:, 4]",
                    },
                },
                "SubhaloBHMass": {
                    "data": result["SubhaloBHMass"],
                    "attrs": {
                        "unit": "1e10 M_sun",
                        "description": "Extracted from SubhaloMassType[:, 5]",
                    },
                },
                "SubhaloGroupNr": {
                    "data": result["SubhaloGroupNr"],
                    "attrs": {"description": "Parent FoF group index"},
                },
            },
            a=a,
            h=h,
        )

    return result


if __name__ == "__main__":
    snapshot_base = "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v2/output/snapdir_020/snapshot_020"
    group_base = "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v2/output/groups_020/fof_subhalo_tab_020"

    halo_out = extract_halo_catalog(
        snapshot_base,
        group_base,
        save_path="/sqfs/work/hp240141/z6b340/results/halo_catalog_physical_020.h5"
    )
    subhalo_out = extract_subhalo_catalog(
        snapshot_base,
        group_base,
        save_path="/sqfs/work/hp240141/z6b340/results/subhalo_catalog_physical_020.h5"
    )

    print(f"a = {halo_out['a']}")
    print(f"h = {halo_out['h']}")
    print("Units: mass = 1e10 M_sun, length = Mpc")
    print(f"Number of halos = {len(halo_out['M200'])}")
    print(
        f"First halo: M200 = {halo_out['M200'][0]:.6e} [1e10 M_sun], "
        f"R200 = {halo_out['R200'][0]:.6e} [Mpc]"
    )
    print(
        f"First halo components: Mgas = {halo_out['GroupGasMass'][0]:.6e}, "
        f"Mdm = {halo_out['GroupDMMass'][0]:.6e}, "
        f"Mstar = {halo_out['GroupStellarMass'][0]:.6e}, "
        f"Mbh = {halo_out['GroupBHMass'][0]:.6e} [1e10 M_sun]"
    )
    print(f"Number of subhalos = {len(subhalo_out['SubhaloMass'])}")
    print(
        f"First subhalo: Msub = {subhalo_out['SubhaloMass'][0]:.6e} [1e10 M_sun], "
        f"Mstar = {subhalo_out['SubhaloStellarMass'][0]:.6e} [1e10 M_sun]"
    )
    print(
        f"First subhalo components: Mgas = {subhalo_out['SubhaloGasMass'][0]:.6e}, "
        f"Mdm = {subhalo_out['SubhaloDMMass'][0]:.6e}, "
        f"Mstar = {subhalo_out['SubhaloStellarMass'][0]:.6e}, "
        f"Mbh = {subhalo_out['SubhaloBHMass'][0]:.6e} [1e10 M_sun]"
    )
