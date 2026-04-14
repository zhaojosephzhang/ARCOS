from cProfile import label
import numpy as np
import h5py
import matplotlib.pyplot as plt
import astropy
from astropy.io import ascii
import astropy.units as u
from matplotlib import units
from numpy import histogram2d
import matplotlib.colors as colors
import bisect
import math
from math import sin, cos, tan, atan
import time
import warnings
import sys
import os
from numpy import array, sqrt, sin, cos, tan
import re
import gc
from scipy.spatial import cKDTree
import numpy as np
from mpi4py import MPI
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import psutil
##################################
me = 9.1*10**-28
mH = 1.67*10**-24
solartog = 2.0*10**33
MpcTocm = 3.08567758*10**24
MsunTog = 1.988*10**33



def M6(r,r0, h, D):
    import numpy as np
    from numpy import pi, sqrt, exp
    ##########D = dimension#######
    ##########h = smoothing length########
    d = np.sqrt((r-r0)**2)
    q = d/h
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5+15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W =  h**(-D)*sigma[D-1]*w
    return W



def W(qz,qxy,D):
    import numpy as np
    from numpy import pi, sqrt, exp
    ##########D = dimension#######
    ##########h = smoothing length########
    q = np.sqrt(qxy**2 + qz**2)
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5+15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W =  sigma[D-1]*w
    return W

    

def Gauss(r, r0, h, D):
    from numpy import pi, sqrt, exp 
    d = sqrt((r-r0)**2)
    V = [2.0, 2*pi*d, 4*pi*d**2]
    sigma = [1/sqrt(pi), 1/pi, 1.0/(pi*sqrt(pi))]
    W = h**(-D)*sigma[D-1]*exp(-(r-r0)**2/h**2)
    W = W*V[D-1]
    return W



def F(qxy):
    from scipy import integrate
    v ,err = integrate.quad(W,  -np.sqrt(1-qxy**2), np.sqrt(1-qxy**2), args = (qxy,3))
    return v
qxy =np.linspace(0, 1, 1000)
Fxy = np.array(list(map(F, qxy)))

def F_xy(q):
    qxy =np.linspace(0, 1, 1000)
    q_xy = bisect.bisect(qxy, q) - 1 
    W = Fxy[q_xy]
    return W

def f_y(c):
    W = np.log(1+c) - c/(1+c)
    return W

def Modified_NFW(r, r_200, M_200):
    f_b = 0.75
    h = 0.67
    rho_c = 9.2*1e-30*(1/(1.988*10**33))*(3.085*10**24)**3
    alpha = 2
    y0 = 2
    c = 4.67*(M_200/10**14/h**-1)**(-0.11)
    y = c*(r/r_200)
    rho_0 = (200*rho_c/3)*c**3/f_y(c)
    rho = f_b*rho_0/(y**(1-alpha)*(y0+y)**(2+alpha))
    return rho

def NFW(r, r_200, M_200):
    f_b = 0.75
    h = 0.67
    rho_c = 9.2*1e-30*(1/(1.988*10**33))*(3.085*10**24)**3
    c = 4.67*(M_200/10**14/h**-1)**(-0.11)
    y = c*(r/r_200)
    rho_0 = (200*rho_c/3)*c**3/f_y(c)
    rho = f_b*rho_0/(y*(1+y**2))
    return rho

def create_grid_indices(data, grid_size, boxsize):
    num_grids = int(np.prod(grid_size))
    grid_indices = {particle_type: {} for particle_type in ["sphpos", "dmpos", "starpos"]}
    
    for particle_type in ["sphpos", "dmpos", "starpos"]:
        positions = data[particle_type]*h
        for i in range(num_grids):
            grid_indices[particle_type][i] = []

        for idx, pos in enumerate(positions):
            if idx % 10000000 == 0:
                print(f"Processing particle type: {particle_type}, Index: {idx}")
                sys.stdout.flush()
            grid_x, grid_y, grid_z = calculate_grid_coordinates(pos, grid_size, boxsize)
            grid_index = calculate_grid_index(grid_x, grid_y, grid_z, grid_size)
            grid_indices[particle_type][grid_index].append(idx)
    
    return grid_indices

def get_nearby_grid_indices(halo_center, grid_size, box_size):
    grid_x, grid_y, grid_z = calculate_grid_coordinates(halo_center, grid_size, box_size)
    search_grid_indices = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                x = (grid_x + dx) % grid_size[0]  # 处理周期性边界条件
                y = (grid_y + dy) % grid_size[1]
                z = (grid_z + dz) % grid_size[2]
                grid_index = calculate_grid_index(x, y, z, grid_size)
                search_grid_indices.append(grid_index)
    
    return search_grid_indices

def calculate_grid_coordinates(position, grid_size, box_size):
    # 根据数据点的位置计算其所在网格的坐标
    x, y, z = position
    grid_x = int((x % box_size) / (box_size / grid_size[0]))
    grid_y = int((y % box_size) / (box_size / grid_size[1]))
    grid_z = int((z % box_size) / (box_size / grid_size[2]))
    return grid_x, grid_y, grid_z

def calculate_grid_index(grid_x, grid_y, grid_z, grid_size):
    # 根据网格坐标计算网格索引
    return int(grid_x + grid_size[0] * (grid_y + grid_size[1] * grid_z))
def find_particles_within_RV(pos, halo_center, RV, Lbox):
    # Apply periodic boundary conditions to halo center
    halo_center = np.mod(halo_center, Lbox)

    # Build KDTree with periodic boundary conditions
    tree = cKDTree(pos, boxsize=Lbox)

    # Find all particles within RV of halo center
    idx = tree.query_ball_point(halo_center, RV)

    # Calculate distances of all particles within RV
    dist = np.sqrt(np.sum((pos[idx] - halo_center)**2, axis=1))

    # Check for particles that cross periodic boundaries
    crossings = pos[idx] - halo_center
    crossings = np.mod(crossings + Lbox/2, Lbox) - Lbox/2
    crossing_dist = np.sqrt(np.sum((crossings)**2, axis=1))
    mask = crossing_dist < dist
    idx = np.array(idx)
    idx[mask] = idx[mask]
    dist[mask] = crossing_dist[mask]

    return idx, dist
def load_field_in_chunks(f, key, data_target, a, h, a_pow=0, h_pow=0, chunked=True, unit_factor=1.0):
    dset = f[key]
    total_particles = dset.shape[0]
    batch_size = 10000000 if chunked else total_particles
    is_vector = (len(dset.shape) == 2)  # 自动判断是否为 vector
    for start in range(0, total_particles, batch_size):
        end = min(start + batch_size, total_particles)
        chunk = dset[start:end] * unit_factor
        if a_pow != 0:
            chunk *= a ** a_pow
        if h_pow != 0:
            chunk *= h ** h_pow
        if is_vector:
            chunk = np.atleast_2d(chunk)
            if data_target.size == 0:
                data_target = chunk
            else:
                if data_target.shape[1] != chunk.shape[1]:
                    print(f"[DEBUG] Shape mismatch for key '{key}': data_target.shape = {data_target.shape}, chunk.shape = {chunk.shape}")
                    raise ValueError(f"Shape mismatch: cannot vstack arrays of shape {data_target.shape} and {chunk.shape}")
                data_target = np.vstack((data_target, chunk))
        else:
            chunk = np.ravel(chunk)
            data_target = np.append(data_target, chunk)
        del chunk
        gc.collect()
    return data_target

def load_halo_data(f, alias_dict, a, h, data_i, check_haloRV=False):
    for key in alias_dict.keys():
        try:
            print("loading {}.\n\tmultipling by a^{} h^{} to get physical value".format(
                key, f[alias_dict[key]].attrs["a_scaling"], f[alias_dict[key]].attrs["h_scaling"]))
            sys.stdout.flush()
            a_pow = int(f[alias_dict[key]].attrs.get("a_scaling", a_scaling[key]))
            h_pow = int(f[alias_dict[key]].attrs.get("h_scaling", h_scaling[key]))
        except:
            print("loading {}.\n\tmultipling by a^{} h^{} to get physical value".format(
                key, a_scaling[key], h_scaling[key]))
            sys.stdout.flush()
            a_pow = int(a_scaling[key])
            h_pow = int(h_scaling[key])
        array = np.array(f[alias_dict[key]])
        array = array * (a ** a_pow) * (h ** h_pow)
        if array.ndim == 1:
            data_i[key] = np.append(data_i[key], array)
        else:
            data_i[key] = np.vstack((data_i[key], array))
        if check_haloRV and key == "haloRV":
            print(f"array: {array}")
            sys.stdout.flush()
        del array
        gc.collect()
    return data_i

def initialize_data(snapfile, alias_snap, alias_frac, alias_T, alias_mass, alias_fof, alias_subfind):
    def is_vector_key(key):
        # 判断是否为向量变量，位置坐标或类似字段
        return key.endswith("pos")

    data = [{} for _ in range(len(snapfile))]

    for i in range(len(snapfile)):
        for key in alias_snap.keys():
            if is_vector_key(key):
                data[i][key] = np.empty((0, 3))
            else:
                data[i][key] = np.empty((0,))
        for key in alias_frac.keys():
            data[i][key] = np.empty((0,))  # 金属丰度等为标量
        for key in alias_T.keys():
            data[i][key] = np.empty((0,))  # 温度通常为标量
        for key in alias_mass.keys():
            data[i][key] = np.empty((0,))  # 质量是标量
        for key in alias_fof.keys():
            if key.endswith("pos"):
                data[i][key] = np.empty((0, 3)) if key == "halopos" else np.empty((0,))
            else:
                data[i][key] = np.empty((0,))
        for key in alias_subfind.keys():
            data[i][key] = np.empty((0, 3)) if key.endswith("pos") else np.empty((0,))
    return data

def read_snapshot_blockwise(snapfile, alias_snap, alias_frac, alias_T, alias_mass, alias_fof, alias_subfind, a_scaling, h_scaling):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = initialize_data(snapfile, alias_snap, alias_frac, alias_T, alias_mass, alias_fof, alias_subfind)
    Boxsize = []

    for i in range(len(snapfile)):
        print("\nstart reading data of {}...".format(snapfile[i][0]))
        try:
            fname = "{}.hdf5".format(snapfile[i][0])
            f = h5py.File(fname, "r")
        except:
            fname = "{}.0.hdf5".format(snapfile[i][0])
            f = h5py.File(fname, "r")
        a = f["Header"].attrs["Time"]
        try:
            h = f["Header"].attrs["HubbleParam"]
            Omega_0 = f["Header"].attrs["Omega0"]
            OmegaLambda = f["Header"].attrs["OmegaLambda"]
        except KeyError:
            h = f["Parameters"].attrs["HubbleParam"]
            Omega_0 = f["Parameters"].attrs["Omega0"]
            OmegaLambda = f["Parameters"].attrs["OmegaLambda"]
        print("This is cosmological run. Scaling code values to physical value.")
        print("a = {}, h = {}".format(a, h))
        MassTable = f["Header"].attrs["MassTable"]
        NumPart_Total = f["Header"].attrs["NumPart_Total"]
        Boxsize.append(f["Header"].attrs["BoxSize"])
        try:
            SPHdensitytocgs = f['Units'].attrs["UnitDensity_in_cgs"]
            SPHmasstocgs = f['Units'].attrs["UnitMass_in_g"]
        except KeyError:
            SPHdensitytocgs = f['PartType0']["Density"].attrs["to_cgs"]
            SPHmasstocgs = f['PartType0']['Masses'].attrs["to_cgs"]
        nfile = f["Header"].attrs["NumFilesPerSnapshot"]
        f.close()

        for j in range(nfile):
            if j % size != rank:
                continue
            fname = "{}.hdf5".format(snapfile[i][0]) if nfile == 1 else "{}.{}.hdf5".format(snapfile[i][0], j)
            print(f"[Rank {rank}] Reading snapshot file: {fname}")
            sys.stdout.flush()
            f = h5py.File(fname, "r")
            NumPart_ThisFile = f["Header"].attrs["NumPart_ThisFile"]

            for key in alias_snap.keys():
                a_pow = int(a_scaling.get(key, 0))
                h_pow = int(h_scaling.get(key, 0))
                is_vector = True if key.endswith("pos") else False
                data[i][key] = load_field_in_chunks(f, alias_snap[key], data[i][key], a, h, a_pow, h_pow)
                if i == 0 and key == "haloRV":
                    print(f"array: {data[i][key]}")
                    sys.stdout.flush()

            for key in alias_frac.keys():
                a_pow = int(a_scaling.get(key, 0))
                h_pow = int(h_scaling.get(key, 0))
                data[i][key] = load_field_in_chunks(f, alias_frac[key], data[i][key], a, h, a_pow, h_pow)

            for key in alias_T.keys():
                a_pow = int(a_scaling.get(key, 0))
                h_pow = int(h_scaling.get(key, 0))
                data[i][key] = load_field_in_chunks(f, alias_T[key], data[i][key], a, h, a_pow, h_pow)

            for key in alias_mass.keys():
                a_pow = int(a_scaling.get(key, 0))
                h_pow = int(h_scaling.get(key, 0))
                if MassTable[alias_mass[key]] == 0:
                    path = "PartType{}/Masses".format(alias_mass[key])
                    data[i][key] = load_field_in_chunks(f, path, data[i][key], a, h, a_pow, h_pow)
                else:
                    count = NumPart_ThisFile[alias_mass[key]]
                    fixed_mass = MassTable[alias_mass[key]]
                    array = np.full(count, fixed_mass)
                    data[i][key] = np.append(data[i][key], array)
                    del array
            f.close()

        try:
            fname = "{}.hdf5".format(snapfile[i][1])
            f = h5py.File(fname, "r")
        except:
            fname = "{}.0.hdf5".format(snapfile[i][1])
            f = h5py.File(fname, "r")
        try:
            nfile = f["Header"].attrs["NumFilesPerSnapshot"]
        except:
            nfile = f["Header"].attrs["NumFiles"]
        f.close()

        for j in range(nfile):
            if j % size != rank:
                continue
            fname = "{}.hdf5".format(snapfile[i][1]) if nfile == 1 else "{}.{}.hdf5".format(snapfile[i][1], j)
            f = h5py.File(fname, "r")
            data[i] = load_halo_data(f, alias_fof, a, h, data[i], check_haloRV=(i == 0))
            data[i] = load_halo_data(f, alias_subfind, a, h, data[i])
            f.close()

    return data, Boxsize, h, a


######################L500N1024################# 
# sim name, snapshot file, FoF file.
snapfile = [
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/snapdir_018/snapshot_018", "/sqfs/work/hp230089/z6b340/Data//L100N512_noAGN/groups_018/fof_subhalo_tab_018"]
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/snapdir_015/snapshot_015", "/sqfs/work/hp230089/z6b340/Data//L100N512_noAGN/groups_015/fof_subhalo_tab_015"]
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_fiducial/snapdir_020/snapshot_020", "/sqfs/work/hp230089/z6b340/Data/L100N512_fiducial/groups_020/fof_subhalo_tab_020"]
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/snapdir_020/snapshot_020", "/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/groups_020/fof_subhalo_tab_020"]
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_fiducial/snapdir_018/snapshot_018", "/sqfs/work/hp230089/z6b340/Data/L100N512_fiducial/groups_018/fof_subhalo_tab_018"]
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_fiducial/snapdir_015/snapshot_015", "/sqfs/work/hp230089/z6b340/Data/L100N512_fiducial/groups_015/fof_subhalo_tab_015"]
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/snapdir_015/snapshot_015", "/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/groups_015/fof_subhalo_tab_015"],
    #["/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/snapdir_018/snapshot_018", "/sqfs/work/hp230089/z6b340/Data/L100N512_noAGN/groups_018/fof_subhalo_tab_018"]
    #["/sqfs/work/hp240141/z6b340/Data/L50N512_fiducial_v1/snapdir_020/snapshot_020", "/sqfs/work/hp240141/z6b340/Data/L50N512_fiducial_v1/groups_020/fof_subhalo_tab_020"],
    #["/sqfs/work/hp240141/z6b340/Data/L50N512_fiducial_v2/snapdir_020/snapshot_020", "/sqfs/work/hp240141/z6b340/Data/L50N512_fiducial_v2/groups_020/fof_subhalo_tab_020"]
    ["/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v1/output/snapdir_019/snapshot_019", "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v1/output/groups_019/fof_subhalo_tab_019"],
    ["/sqfs/work/hp240141/z6b340/Data/L100N1024_NoBH/output/snapdir_019/snapshot_019", "/sqfs/work/hp240141/z6b340/Data/L100N1024_NoBH/output/groups_019/fof_subhalo_tab_019"]
    ]
# ============================
# 1. 获取 snapshot ID
# ============================
if len(sys.argv) != 2:
    raise ValueError("Usage: python script.py <snapshot_number>")

snap_id = int(sys.argv[1])               # 例如 19
snap_str = f"{snap_id:03d}"              # '019'
snap_id = int(sys.argv[1])        # 输入 19
snap_str = f"{snap_id:03d}"       # '019'

# ============================
# 2. 生成 Fiducial 和 NoBH 路径
# ============================

base_Fid = "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v1/output"
base_NoBH = "/sqfs/work/hp240141/z6b340/Data/L100N1024_NoBH/output"

snapfile = [
    [
        f"{base_Fid}/snapdir_{snap_str}/snapshot_{snap_str}",
        f"{base_Fid}/groups_{snap_str}/fof_subhalo_tab_{snap_str}"
    ],
    [
        f"{base_NoBH}/snapdir_{snap_str}/snapshot_{snap_str}",
        f"{base_NoBH}/groups_{snap_str}/fof_subhalo_tab_{snap_str}"
    ]
]

# ============================
# 3. 自动生成输出目录
# ============================
output_dir = f"/sqfs/work/hp240141/z6b340/results/Halo_data_2D/snap_{snap_id}"
os.makedirs(output_dir, exist_ok=True)

print(">>> snapfile list:")
for item in snapfile:
    print(item)

print(">>> Output directory:", output_dir)
sys.stdout.flush()

# set alias name
alias_snap = {
    "rho": "PartType0/Density",
    #"u": "PartType0/InternalEnergy",
    "T": "PartType0/Temperature",
    "sphpos": "PartType0/Coordinates",
    "dmpos": "PartType1/Coordinates",
    "starpos": "PartType4/Coordinates",
    "smoothlen": "PartType0/SmoothingLength",
    "SFR": "PartType0/StarFormationRate",
    "SFT": "PartType4/StellarFormationTime"
}

alias_frac = {    
    "n_HI": "PartType0/HI",
    "n_HII": "PartType0/HII",
    "n_H2I": "PartType0/H2I",
    "n_e": "PartType0/ElectronAbundance",
    #"n_HeI": "PartType0/HeI"
    #"n_HeII": "PartType0/HeII",
    #"n_HeIII": "PartType0/HeIII",
    #"n_DI": "PartType0/DI",
    #"n_DII": "PartType0/DII",
    #"n_H2I": "PartType0/H2I",
    #"n_H2II": "PartType0/H2II"
    "Z_sph": "PartType0/Metallicity",
    "Z_star": "PartType4/Metallicity",
    "CE_O": "PartType0/CELibOxygen",
    "CE_Fe": "PartType0/CELibIron"
 }

alias_T = {
    # "T": "PartType0/Temperature"
}  

alias_mass = {
    "sphmass": 0,
    "dmmass": 1,
    "starmass": 4
}
alias_fof = {
    "halomass": "Group/GroupMass",
    "halopos": "Group/GroupPos",
    "halonum": "Group/GroupNsubs",
    "halo0": "Group/GroupFirstSub",
    "halolen": "Group/GroupLen",
    "haloRV": "Group/Group_R_Crit200",
    "haloMV": "Group/Group_M_Crit200",
    "haloRV_mean": "Group/Group_R_Mean200",
    "haloMV_mean": "Group/Group_M_Mean200"
}

alias_subfind = {
    "subhalopos": "Subhalo/SubhaloPos",
    "subhalomass": "Subhalo/SubhaloMass",
    #"subhaloGrNr": "Subhalo/SubhaloGrNr",
    #"subhaloparent": "Subhalo/SubhaloParent",
    "subhalolen" : "Subhalo/SubhaloLen"
    
}

a_scaling = {
    "rho": "-3",
    "u": "0",
    "sphpos": "1",
    "dmpos": "1",
    "starpos": "1",
    "bhpos": "1",
    "smoothlen": "1",
    "SFR": "0",
    "SFT": "0",
    "T": "0",
    "sphmass": "0",
    "dmmass": "0",
    "starmass": "0",
    "bhmass": "0",
    "bhMdot": "0",
    "Z_sph": "0",
    "Z_star": "0",
    "CE_O": "0",
    "CE_Fe": "0",
    "halopos": "1",
    "halomass": "0",
    "halolen": "0",
    "halonum": "0",
    "halo0": "0",
    "subhalopos": "1",
    "subhalomass": "0",
    "subhalBHMmax": "0",
    "subhalolen" : "0",
    "haloRV" : "1",
    "haloMV": "0",
    "halobhM": "0",
    "haloRV_mean": "1",
    "haloMV_mean": "0",
    "GroupID": "0",
    "Groupmasstype": "0",
    "sphID": "0",
    "dmID": "0",
    "starID": "0"

    
}

h_scaling = {
    "rho": "2",
    "u": "0",
    "sphpos": "-1",
    "dmpos": "-1",
    "starpos": "-1",
    "bhpos": "-1",
    "smoothlen": "-1",
    "SFR": "0",
    "SFT": "0",
    "T": "0",
    "sphmass": "-1",
    "dmmass": "-1",
    "starmass": "-1",
    "bhmass": "-1",
    "bhMdot": "0",
    "Z_sph": "0",
    "Z_star": "0",
    "CE_O": "-1",
    "CE_Fe": "-1",
    "halopos": "-1",
    "halolen": "0",
    "halonum": "0",
    "halo0": "0",
    "subhalopos": "-1",
    "subhalomass": "-1",
    "subhalBHMmax": "-1",
    "subhalolen" : "0",
    "haloRV" : "-1",
    "halomass": "-1",
    "haloMV": "-1",
    "halobhM": "-1",
    "haloRV_mean": "-1",
    "haloMV_mean": "-1",
    "Groupmasstype": "-1",
    "GroupID": "0",
    "sphID": "0",
    "dmID": "0",
    "starID": "0"
    
}

import numpy as np
import sys

def safe_gather_array(arr, comm, root=0, tag_base=100):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    max_bytes = 2**31 - 1  # Max number of bytes per single MPI transmission (limited by MPI implementation)
    arr_flat = arr.ravel()  # Flatten the array to 1D for transmission
    dtype = arr.dtype  # Data type of the array (e.g., float32, float64)
    total_bytes = arr_flat.nbytes  # Total number of bytes of the array
    chunk_size = max_bytes // dtype.itemsize  # Maximum number of elements that can be sent per chunk

    print(f"[Rank {rank}] Starting safe_gather_array with shape={arr.shape}, dtype={dtype}, total_bytes={total_bytes}")
    sys.stdout.flush()

    if rank == root:
        result = []  # Will hold all gathered arrays from each rank
        for r in range(size):
            if r == root:
                # Root rank appends its own array directly without communication
                print(f"[Rank {rank}] Appending local array directly")
                result.append(arr)
            else:
                # Receive the shape of the incoming array from rank r
                print(f"[Rank {rank}] Waiting for shape from Rank {r}...")
                sys.stdout.flush()
                shape = comm.recv(source=r, tag=tag_base)
                print(f"[Rank {rank}] Received shape {shape} from Rank {r}")
                sys.stdout.flush()

                # Allocate a flat buffer with the appropriate number of elements
                # We use np.prod(shape) to calculate total number of elements to receive
                recv_buffer = np.empty(np.prod(shape), dtype=dtype)

                # Receive the data in multiple chunks if needed (to avoid MPI overflow)
                n_chunks = int(np.ceil(recv_buffer.size / chunk_size))
                for i in range(n_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, recv_buffer.size)
                    print(f"[Rank {rank}] Receiving chunk {i+1}/{n_chunks} from Rank {r} [{start}:{end}]")
                    sys.stdout.flush()
                    comm.Recv(recv_buffer[start:end], source=r, tag=tag_base + 1 + i)

                # Reshape the flat buffer back to the original shape
                result.append(recv_buffer.reshape(shape))
        try:
            # Concatenate arrays from all ranks along axis 0
            print(f"[Rank {rank}] Concatenating {len(result)} arrays...")
            sys.stdout.flush()
            return np.concatenate(result, axis=0)
        except Exception as e:
            print(f"[Rank {rank}] Concatenation failed: {e}")
            return None

    else:
        # Non-root ranks send their array shape first
        print(f"[Rank {rank}] Sending shape {arr.shape} to Rank {root}")
        sys.stdout.flush()
        comm.send(arr.shape, dest=root, tag=tag_base)

        # Send the array in chunks if needed
        n_chunks = int(np.ceil(arr_flat.size / chunk_size))
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, arr_flat.size)
            print(f"[Rank {rank}] Sending chunk {i+1}/{n_chunks} [{start}:{end}] to Rank {root}")
            sys.stdout.flush()
            comm.Send(arr_flat[start:end], dest=root, tag=tag_base + 1 + i)

        return None



# radial bin of log(r[Mpc])
Xbins = np.linspace(-2, 1, num=20)
Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
# ----------------------------------------#
# initialization

data_i, Boxsize, h, a = read_snapshot_blockwise(snapfile, alias_snap, alias_frac, alias_T, alias_mass, alias_fof, alias_subfind, a_scaling, h_scaling)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

comm.Barrier()
merged_data = [{} for _ in range(len(snapfile))]

for i in range(len(snapfile)):
    for key in data_i[i]:
        shape = data_i[i][key].shape
        size_MB = data_i[i][key].nbytes / 1024**2
        print(f"[Rank {rank}] Key: {key:<15} | Shape: {shape} | Size: {size_MB:.2f} MB")
        sys.stdout.flush()
print(f"[Rank {rank}] Finished printing all key info for snapfiles.")
sys.stdout.flush()
comm.Barrier()


all_keys = sorted(list(data_i[0].keys()))
key_to_id = {k: i for i, k in enumerate(all_keys)}

#for i in range(len(snapfile)):
#    for key in data_i[i]:
#        print(f"[Rank {rank}] Will gather key={key} for snap {i} with shape={data_i[i][key].shape}")
#        sys.stdout.flush()

#        tag_base = 1000 + i * 1000 + key_to_id[key]
#        gathered = safe_gather_array(data_i[i][key], comm, root=0, tag_base=tag_base)

#        if rank == 0:
#            merged_data[i][key] = gathered
#            print(f"[Rank {rank}] Memory used after storing key={key}: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
#            del gathered  # Explicitly delete to free memory
#            gc.collect()  # Force garbage collection
#    print(f"[Rank {rank}] key={key} already gathered")
#comm.Barrier()
AGN_info = ["f1", "n"]
data_filename_prefix = "gathered_snap"
#output_dir = "/sqfs/work/hp240141/z6b340/results/Halo_data_2D/snap_19"

if rank == 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(len(snapfile)):
    for key in data_i[i]:
        print(f"[Rank {rank}] Will gather key={key} for snap {i} with shape={data_i[i][key].shape}")
        sys.stdout.flush()

        tag_base = 1000 + i * 1000 + key_to_id[key]
        gathered = safe_gather_array(data_i[i][key], comm, root=0, tag_base=tag_base)

        if rank == 0:
            # Immediately write gathered data to disk to avoid RAM accumulation
            filename = os.path.join(output_dir, f"{data_filename_prefix}_{AGN_info[i]}_{key}.h5")
            with h5py.File(filename, "w") as hf:
                hf.create_dataset(key, data=gathered)
            print(f"[Rank {rank}] Saved {key} for snap {i} to {filename}")
            sys.stdout.flush()
            
            # Free memory
            del gathered
            gc.collect()
comm.Barrier()

#computing radial profiles
# log radius

if rank == 0:
    #output_dir = "/sqfs/work/hp240141/z6b340/results/gathered_results"
    #result_dir = "results"
    pattern = r'L(\d+)N(\d+)(_?\w+)?'

    Halo_filename = []
    Data_filename = []
    HaloMass = []
    HaloRV = []
    HaloMV = []
    HaloRV_mean = []
    HaloMV_mean = []
    HaloMass0 = []
    HaloRV0 = []
    mfunc_tinker08 = []
    mfunc_press74 = []

    for i in range(len(snapfile)):
        data_i = {}
        for key_file in os.listdir(output_dir):
            if key_file.startswith(f"gathered_snap_{AGN_info[i]}_") and key_file.endswith(".h5"):
                key = key_file.split(f"gathered_snap_{AGN_info[i]}_")[1].replace(".h5", "")
                with h5py.File(os.path.join(output_dir, key_file), 'r') as f:
                    data_i[key] = f[key][:]

        # Halo properties
        HaloMass.append(data_i['halomass'])
        HaloMV.append(data_i['haloMV'])
        HaloRV.append(data_i['haloRV'])
        HaloMV_mean.append(data_i['haloMV_mean'])
        HaloRV_mean.append(data_i['haloRV_mean'])
        HaloMass0.append(data_i['halomass'][0]*1e10)
        HaloRV0.append(data_i['haloRV'][0])

        cosmology.setCosmology('WMAP9')
        mfunc_tinker08.append(mass_function.massFunction(data_i['halomass'], 0.0, mdef='vir', model='tinker08'))
        mfunc_press74.append(mass_function.massFunction(data_i['halomass'], 0.0, mdef='fof', model='press74'))

        # HDF5 output filename preparation
        split_path = snapfile[i][0].split("Data")
        suffixes = re.search(pattern, snapfile[i][0])[0]
        match = re.match(pattern, suffixes)
        l_number = match.group(1)
        additional = match.group(3) if match.group(3) else ""
        file_num = snapfile[i][0].split("snapshot_")[-1]
        new_path = output_dir
        if not os.path.exists(new_path): os.makedirs(new_path)

        Halo_filename.append(os.path.join(new_path, f"halo_{l_number}_{file_num}{additional}_with_stellar.h5"))
        Data_filename.append(os.path.join(new_path, f"data_{l_number}_{file_num}{additional}_with_stellar.h5"))

        # Save halo info
        with h5py.File(Halo_filename[i], "w") as hf:
            hf.attrs['HubbleParam'] = h
            hf.attrs['Time'] = a
            hf.create_dataset("Halomass", data=HaloMass[i])
            hf.create_dataset("HaloRV", data=HaloRV[i])
            hf.create_dataset("HaloMV", data=HaloMV[i])
            hf.create_dataset("HaloRV_mean", data=HaloRV_mean[i])
            hf.create_dataset("HaloMV_mean", data=HaloMV_mean[i])
            hf.create_dataset("mfunc_tinker08", data=mfunc_tinker08[i])
            hf.create_dataset("mfunc_press74", data=mfunc_press74[i])

        # Save each key individually to avoid memory overflow
        with h5py.File(Data_filename[i], "w") as hf:
            hf.attrs['HubbleParam'] = h
            hf.attrs['Time'] = a
            group = hf.create_group("data")
            for key, value in data_i.items():
                if isinstance(value, np.ndarray):
                    group.create_dataset(key, data=value)

        # Cleanup
        del data_i
        gc.collect()
        print(f"[Rank 0] Finished storing snapshot {i}")
        sys.stdout.flush()
