from cProfile import label
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import joblib
from joblib import Parallel, delayed
import multiprocessing
from scipy.integrate import quad
import astropy
from astropy.io import ascii
import astropy.units as u
from matplotlib import units
from numpy import histogram2d
import matplotlib.colors as colors
import bisect
import math
from math import atan
import time
import warnings
import sys
import os
from numpy import array, sqrt, sin, cos, tan
from mpl_toolkits.mplot3d import Axes3D
import random
import copy
import re
from astropy.cosmology import Planck15 as cosmo
from scipy.optimize import root_scalar
H0 = cosmo.H0
from astropy.constants import G, c, m_e, m_p
import concurrent
from numba import njit
#from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor  
from mpi4py import MPI
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import argparse
import glob
import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# 初始化 MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # 当前进程 ID
size = comm.Get_size()   # 总进程数
num_cores_per_rank = 60 # 每个 MPI 进程使用 60 个核心
if rank == 0:
    print(f"MPI Parallel Execution: {size} MPI processes, each using {num_cores_per_rank} CPU cores.")
    sys.stdout.flush()
# 设置 joblib 的并行数
font = {
    'family': 'serif',
    'size': 16
}

# 允许的星系名称列表
Mass_Ranges_Choices = ["10-11", "11-12", "12-13", "13-14", "14-15"]
# 创建解析器
parser = argparse.ArgumentParser(description="Process input parameters.")
parser.add_argument("mass_range", type=str, choices=Mass_Ranges_Choices, help="Name of the galaxy (Dwarf, AGORA, GalCluster)")
parser.add_argument("halo_num", type=int, help="The number of analysis halos")
parser.add_argument("--snap-num", type=int, default=20, help="Snapshot number used for reading and writing data. Default: 20")
parser.add_argument("--radial-bin-mode", type=str, choices=["inner", "uniform"], default="inner",
                    help="Radial binning mode for impact parameter sampling. Default: inner")
parser.add_argument("--agn-info", type=str, choices=["f", "n", "both"], default="both",
                    help="Run mode: f (Fiducial only), n (NoBH only), or both (default).")
parser.add_argument("--h5-write-mode", type=str, choices=["append", "overwrite"], default="overwrite",
                    help="HDF5 output mode: append keeps existing halos and adds missing ones; overwrite recreates file.")
parser.add_argument("--mpi-io-mode", type=str, choices=["mpio", "merge"], default="mpio",
                    help="MPI HDF5 mode: mpio writes one shared file; merge writes per-rank files and merges on rank0.")

##################################
# 解析参数
args = parser.parse_args()
mass_range_label = args.mass_range
halo_num_analysis =  args.halo_num
snap_num = args.snap_num
radial_bin_mode = args.radial_bin_mode
agn_info_mode = args.agn_info
h5_write_mode = args.h5_write_mode
mpi_io_mode = args.mpi_io_mode
snap_label = f"snap_{snap_num}"
snap_tag = f"{snap_num:03d}"
#print(H_mass_frac, He_mass_frac)
#f_IGM =  0.65
SPHmasstocgs =  1.989e+43
Mpctokpc = 1e3
warnings.simplefilter('always', RuntimeWarning)
me = 9.1*10**-28
mH = 1.67*10**-24
mp = 1.67*10**-24
solartog = 2.0*10**33
# 宇宙学参数
Omega_m = 0.31  # 物质密度参数
Omega_Lambda = 0.69  # 暗能量密度参数
H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
c = 299792.458  # 光速，单位为 km/s
MpcTocm = 3.08567758*10**24
directory_path = f"/sqfs/work/hp240141/z6b340/results/Halo_data_2D/{snap_label}/"
storage_path_100f = f"/sqfs/work/hp240141/z6b340/results/Halo_data_2D/{snap_label}/DM_map_100f_low_with_stellar_1D"
save_path_100f = f"./DM_map_100f_figure_joblib_max_with_stellar_1D/{snap_label}"
storage_path_100n = f"/sqfs/work/hp240141/z6b340/results/Halo_data_2D/{snap_label}/DM_map_100n_low_with_stellar_1D"
save_path_100n = f"./DM_map_100n_figure_joblib_max_with_stellar_1D/{snap_label}"
# 检查目录是否存在，如果不存在则创建目录
h = 0.67742
if rank == 0:
    if not os.path.exists(storage_path_100f):
        os.makedirs(storage_path_100f)
        print(f"Directory {storage_path_100f} created.")
    else:
        print(f"Directory {storage_path_100f} already exists.")

    if not os.path.exists(save_path_100f):
        os.makedirs(save_path_100f)
        print(f"Directory {save_path_100f} created.")
    else:
        print(f"Directory {save_path_100f} already exists.")


    if not os.path.exists(storage_path_100n):
        os.makedirs(storage_path_100n)
        print(f"Directory {storage_path_100n} created.")
    else:
        print(f"Directory {storage_path_100n} already exists.")

    if not os.path.exists(save_path_100n):
        os.makedirs(save_path_100n)
        print(f"Directory {save_path_100n} created.")
    else:
        print(f"Directory {save_path_100n} already exists.")
comm.Barrier()

def format_mass_range(mass_range_key):
    # 分割mass_range_key到两部分
    parts = mass_range_key.split('-')
    
    # 将每部分转换为科学计数法并取对数，然后构造LaTeX格式的字符串
    formatted_parts = [r'10^{' + f"{round(math.log10(float(part)),1)}" + '}' for part in parts]
    
    # 将转换后的部分再次组合
    return '-'.join(formatted_parts)

def _prepare_positive_array(arr):
    arr = np.array(arr, dtype=np.float64, copy=True)
    arr[arr <= 0] = np.nan
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr

def _mask_nonpositive(arr):
    out = np.array(arr, dtype=np.float64, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[out <= 0] = np.nan
    return out

def _has_positive_values(*arrays):
    for arr in arrays:
        a = np.asarray(arr)
        if np.any(np.isfinite(a) & (a > 0)):
            return True
    return False

def log_mean_without_zeros(arr):
    arr = _prepare_positive_array(arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_arr = np.log(arr)
    valid_cols = np.any(~np.isnan(log_arr), axis=0)
    log_mean = np.full(log_arr.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        log_mean[valid_cols] = np.nanmean(log_arr[:, valid_cols], axis=0)
    return np.exp(log_mean)

def linear_mean_without_zeros(arr):
    arr = _prepare_positive_array(arr)
    valid_cols = np.any(~np.isnan(arr), axis=0)
    linear_mean = np.full(arr.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        linear_mean[valid_cols] = np.nanmean(arr[:, valid_cols], axis=0)
    return linear_mean

def median_without_zeros(arr):
    arr = _prepare_positive_array(arr)
    valid_cols = np.any(~np.isnan(arr), axis=0)
    median = np.full(arr.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        median[valid_cols] = np.nanmedian(arr[:, valid_cols], axis=0)
    return median

def calculate_error_bands(data):
    """
    计算上下两条曲线作为误差范围。

    参数：
    data (numpy.ndarray): 数据数组。

    返回：
    (numpy.ndarray, numpy.ndarray): 返回上限曲线和下限曲线。
    """
    # 在对数空间中计算 percentiles
    data = _prepare_positive_array(data)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_data = np.log(data)

    # 计算上下限
    percentiles = [16, 84]  # 16th and 84th percentiles
    valid_cols = np.any(~np.isnan(log_data), axis=0)
    lower_bound_curve = np.full(log_data.shape[1], np.nan, dtype=np.float64)
    upper_bound_curve = np.full(log_data.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        percentile_values = np.nanpercentile(log_data[:, valid_cols], percentiles, axis=0)
        lower_bound_curve[valid_cols] = percentile_values[0]
        upper_bound_curve[valid_cols] = percentile_values[1]

    return np.exp(lower_bound_curve), np.exp(upper_bound_curve)
# 假设共动距离为5330 Mpc
#@njit
def M6(r, r0, h, D):
    ##unit in cm
    d = np.sqrt((r-r0)**2)
    q = d/h
    V = [2.0, 2*np.pi*d, 4*np.pi*d**2]
    sigma = np.array([1.0/120*3, 7.0/(478.0*np.pi)*3**2, 1.0/(120*np.pi)*3**3])
    if q >= 0 and q < 1.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5 + 15*(3)**5*(1.0/3-q)**5
    elif q >= 1.0/3 and q < 2.0/3:
        w = (3.0)**5*(1.0 - q)**5 - 6*(3.0)**5*(2.0/3-q)**5
    elif q >= 2.0/3 and q < 1.0:
        w = (3.0)**5*(1.0 - q)**5
    else:
        w = 0
    W = h**(-D)*sigma[D-1]*w
    return W

#@njit(cache=True, fastmath=True)
def M6_3D(d_cm, h_cm):
    # q = d/h
    q = d_cm / h_cm

    # sigma3 = 1/(120*pi) * 3^3 = 27/(120*pi)
    sigma3 = 27.0 / (120.0 * np.pi)

    if 0.0 <= q < 1.0/3.0:
        w = (3.0**5) * (1.0 - q)**5 - 6.0*(3.0**5) * (2.0/3.0 - q)**5 + 15.0*(3.0**5) * (1.0/3.0 - q)**5
    elif 1.0/3.0 <= q < 2.0/3.0:
        w = (3.0**5) * (1.0 - q)**5 - 6.0*(3.0**5) * (2.0/3.0 - q)**5
    elif 2.0/3.0 <= q < 1.0:
        w = (3.0**5) * (1.0 - q)**5
    else:
        w = 0.0

    # W = h^{-3} * sigma3 * w
    return (sigma3 * w) / (h_cm*h_cm*h_cm)

# 计算共动距离和尺度因子的函数
def comoving_distance(z):
    integrand = lambda zp: 1.0 / np.sqrt(Omega_m * (1.0+zp)**3 + Omega_Lambda)
    integral, _ = quad(in3tegrand, 0, z)
    return c/H_0 * integral

def scale_factor(z):
    return 1.0 / (1.0 + z)

def log_mean_without_zeros(arr):
    arr = _prepare_positive_array(arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_arr = np.log(arr)
    valid_cols = np.any(~np.isnan(log_arr), axis=0)
    log_mean = np.full(log_arr.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        log_mean[valid_cols] = np.nanmean(log_arr[:, valid_cols], axis=0)
    return np.exp(log_mean)

def linear_mean_without_zeros(arr):
    arr = _prepare_positive_array(arr)
    valid_cols = np.any(~np.isnan(arr), axis=0)
    linear_mean = np.full(arr.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        linear_mean[valid_cols] = np.nanmean(arr[:, valid_cols], axis=0)
    return linear_mean

def median_without_zeros(arr):
    arr = _prepare_positive_array(arr)
    valid_cols = np.any(~np.isnan(arr), axis=0)
    median = np.full(arr.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        median[valid_cols] = np.nanmedian(arr[:, valid_cols], axis=0)
    return median

def calculate_error_bands(data):
    """
    计算上下两条曲线作为误差范围。

    参数：
    data (numpy.ndarray): 数据数组。

    返回：
    (numpy.ndarray, numpy.ndarray): 返回上限曲线和下限曲线。
    """
    # 在对数空间中计算 percentiles
    data = _prepare_positive_array(data)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_data = np.log(data)

    # 计算上下限
    percentiles = [0.15, 99.85] # 0.15th and 99.85th percentiles
    valid_cols = np.any(~np.isnan(log_data), axis=0)
    lower_bound_curve = np.full(log_data.shape[1], np.nan, dtype=np.float64)
    upper_bound_curve = np.full(log_data.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        percentile_values = np.nanpercentile(log_data[:, valid_cols], percentiles, axis=0)
        lower_bound_curve[valid_cols] = percentile_values[0]
        upper_bound_curve[valid_cols] = percentile_values[1]

    return np.exp(lower_bound_curve), np.exp(upper_bound_curve)


def _allocate_segment_counts(num_radii, weights):
    """Allocate an exact number of radial bins across multiple segments."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    counts = np.floor(weights * num_radii).astype(int)
    counts = np.maximum(counts, 2)

    while counts.sum() > num_radii:
        idx = np.argmax(counts)
        if counts[idx] > 2:
            counts[idx] -= 1
        else:
            break

    if counts.sum() < num_radii:
        fractional = weights * num_radii - np.floor(weights * num_radii)
        order = np.argsort(-fractional)
        for i in range(num_radii - counts.sum()):
            counts[order[i % len(order)]] += 1

    while counts.sum() < num_radii:
        counts[np.argmin(counts)] += 1

    return counts


def build_radial_bin_centers(HaloRV, C_R_max, num_radii, radial_bin_mode="uniform"):
    """Build impact-parameter bin centers in Mpc."""
    r_max = C_R_max * HaloRV
    if radial_bin_mode != "inner":
        return np.linspace(0.0, r_max, num_radii)

    # Keep radial sampling independent from sightline overlap at small radii:
    # r_min_inner = min(0.05 * R200, 30 kpc)
    r_min_inner = min(0.05 * HaloRV, 30.0 / 1e3)  # Mpc
    r_min_inner = min(r_min_inner, r_max)

    # Piecewise-linear inner grid with reduced central over-weighting.
    breakpoints = [r_min_inner]
    for edge_norm in (0.05, 0.2):
        edge = edge_norm * HaloRV
        if r_min_inner < edge < r_max:
            breakpoints.append(edge)
    breakpoints.append(r_max)

    edges = np.array(sorted(set(breakpoints)), dtype=np.float64)
    num_segments = len(edges) - 1
    if num_segments == 1:
        return np.linspace(r_min_inner, r_max, num_radii)

    # Fewer bins at very small radii, more bins at larger radii.
    base_weights = np.array([0.12, 0.28, 0.60], dtype=np.float64)[:num_segments]
    counts = _allocate_segment_counts(num_radii, base_weights)

    segments = []
    for i in range(num_segments):
        endpoint = (i == num_segments - 1)
        segment = np.linspace(edges[i], edges[i + 1], counts[i], endpoint=endpoint)
        segments.append(segment)

    radii = np.concatenate(segments)
    if radii.size != num_radii:
        raise ValueError(f"Expected {num_radii} radial bins, got {radii.size}.")
    if np.any(np.diff(radii) <= 0):
        raise ValueError("Radial bin centers must be strictly increasing without duplicates.")

    return radii

# 已知共动距离，求红移和尺度因子的函数
def find_redshift_and_scale_factor(dC):
    # 定义被求解的方程
    equation = lambda z: comoving_distance(z) - dC

    # 使用root_scalar函数求解方程
    sol = root_scalar(equation, bracket=[0, 10])

    # 检查解是否收敛，如果不收敛则返回None
    if not sol.converged:
        return None, None

    # 返回红移和尺度因子
    z = sol.root
    a = scale_factor(z)
    return z, a
def find_particles_within_RV_sph(pos, halo_center, RV, Lbox):
    halo_center = np.mod(halo_center, Lbox)
    pos = np.array(pos)
    pos = np.mod(pos, Lbox)
    halo_center = np.array(halo_center)
    tree = cKDTree(pos, boxsize=Lbox)
    idx = tree.query_ball_point(halo_center, RV)
    dist = np.sqrt(np.sum((pos[idx] - halo_center)**2, axis=1))
    crossings = pos[idx] - halo_center
    crossings = np.mod(crossings + Lbox/2, Lbox) - Lbox/2
    crossing_dist = np.sqrt(np.sum((crossings)**2, axis=1))
    mask = crossing_dist < dist
    idx = np.array(idx)
    idx[mask] = idx[mask]
    dist[mask] = crossing_dist[mask]
    return idx, dist

def find_particles_within_RV_sph_tree(tree, pos, halo_center, RV, Lbox):
    """
    Find particles within a spherical region using a pre-built KDTree.

    Parameters:
    - tree: Pre-built cKDTree for the particle positions.
    - pos: Positions of all particles (N x 3 array).
    - halo_center: Center of the halo (3D coordinate).
    - RV: Radius of the spherical region.
    - Lbox: Size of the simulation box for applying periodic boundary conditions.

    Returns:
    - idx: Indices of particles within the spherical region.
    - dist: Distances of those particles from the halo center.
    """
    # Apply periodic boundary conditions to the halo center
    halo_center = np.mod(halo_center, Lbox)

    # Query the KDTree to find particles within the radius
    idx = tree.query_ball_point(halo_center, RV)

    # Calculate distances from the halo center
    pos = np.array(pos)
    pos = np.mod(pos, Lbox)
    halo_center = np.array(halo_center)
    dist = np.sqrt(np.sum((pos[idx] - halo_center)**2, axis=1))

    # Handle periodic boundary crossings
    crossings = pos[idx] - halo_center
    crossings = np.mod(crossings + Lbox / 2, Lbox) - Lbox / 2
    crossing_dist = np.sqrt(np.sum(crossings**2, axis=1))

    # Update distances for particles crossing periodic boundaries
    mask = crossing_dist < dist
    idx = np.array(idx)
    dist[mask] = crossing_dist[mask]

    return idx, dist


def find_particles_within_cylinder(pos, cylinder_center, cylinder_axis, radius, height, Lbox):
    """
    Find particles within a cylindrical region.
    
    Parameters:
    - pos: Positions of all particles (N x 3 array).
    - cylinder_center: Center of the cylinder (3D coordinate).
    - cylinder_axis: Direction of the cylinder axis (unit vector along the sightline).
    - radius: Radius of the cylindrical cross-section.
    - height: Height of the cylinder (along the axis).
    - Lbox: Size of the simulation box for applying periodic boundary conditions.
    
    Returns:
    - indices of particles within the cylindrical region.
    - distances of those particles from the axis.
    """
    # Apply periodic boundary conditions
    pos = np.mod(pos - cylinder_center + Lbox / 2, Lbox) - Lbox / 2

    # Calculate the projection of each particle onto the cylinder axis
    proj = np.dot(pos, cylinder_axis)

    # Find particles within the height range of the cylinder
    mask_height = np.abs(proj) < height / 2

    # Calculate the perpendicular distance from the cylinder axis
    perp_distances = np.linalg.norm(pos - np.outer(proj, cylinder_axis), axis=1)

    # Find particles within the radius of the cylinder
    mask_radius = perp_distances < radius

    # Combine the height and radius masks
    mask = mask_height & mask_radius

    # Get the indices and distances of particles inside the cylinder
    indices = np.where(mask)[0]
    distances = perp_distances[mask]

    return indices, distances

def calc_ne_LOS_shifted(k, i, LOS_shifted, sphpos, h_smooth_max, Lbox, SPHmasstocgs, f_e, mp, Smoothlen):
    ne_LC_shifted_k_i = 0
    idx_LC, dist_LC = find_particles_within_RV_sph(sphpos, LOS_shifted[k][i], h_smooth_max, Lbox)
    for j in range(len(idx_LC)):
        if idx_LC[j] >= len(sphpos):
            idx_LC[j] = idx_LC[j] - len(sphpos)
        if dist_LC[j] >= Smoothlen[idx_LC[j]]:
            continue
        else:
            #weight = M6(dist_LC[j], 0, Smoothlen[idx_LC[j]], 3)
            weight = M6(dist_LC[j], 0, Smoothlen[idx_LC[j]], 3)
            ne_LC_shifted_k_i += SPHmasstocgs[idx_LC[j]] * f_e[idx_LC[j]] / mp * weight
    return ne_LC_shifted_k_i

def get_halo_indices(halo_group):
    halo_indices = []
    for key in halo_group.keys():
        idx = int(key.replace('halo_', ''))
        halo_indices.append(idx)
    return halo_indices

# Function to load halo data
def load_halo_data(directory_path, snap_tag):
    #halo_100_noAGN = h5py.File(directory_path + "all_ne_DM_2RV_N500_B100_18_n_alongx.h5", 'r')
    haloinfo_100_noAGN = h5py.File(
        directory_path + f"Haloinfo_11RV_N500_B100_{snap_tag}_n_alongx_withstellar_addition.h5", 'r'
    )

    #halo_100_AGN = h5py.File(directory_path + "all_ne_DM_2RV_N500_B100_18_f_alongx.h5", 'r')
    haloinfo_100_AGN = h5py.File(
        directory_path + f"Haloinfo_11RV_N500_B100_{snap_tag}_f_alongx_withstellar_addition.h5", 'r'
    )

    #halo_500_noAGN = h5py.File(directory_path + "all_ne_DM_2RV_N500_B500_alongx.h5", 'r')
    #haloinfo_500_noAGN = h5py.File(directory_path + "all_DM_Haloinfo_2RV_N500_B500_24_n_alongx.h5", 'r')
    
    #data_100_noAGN = h5py.File(directory_path + "data_100_018_noAGN.h5", 'r')
    #data_100_AGN = h5py.File(directory_path + "data_100_018_fiducial.h5", 'r')
    #data_500_noAGN = h5py.File(directory_path + "data_500_024.h5", 'r')
    #Halocenter_100_noAGN = data_100_noAGN['data']['halopos']
    #Halocenter_100_AGN = data_100_AGN['data']['halopos']
    #Halocenter_500_noAGN = data_500_noAGN['data']['halopos']
    
    return  haloinfo_100_noAGN, haloinfo_100_AGN#, haloinfo_500_noAGN


def read_scale_factors_from_haloinfo(haloinfo_hf):
    """Read a and h from Haloinfo HDF5 attributes."""
    if "snapfile_0" in haloinfo_hf:
        snap_group = haloinfo_hf["snapfile_0"]
        if "Time" in snap_group.attrs and "HubbleParam" in snap_group.attrs:
            return float(snap_group.attrs["Time"]), float(snap_group.attrs["HubbleParam"])

    if "Time" in haloinfo_hf.attrs and "HubbleParam" in haloinfo_hf.attrs:
        return float(haloinfo_hf.attrs["Time"]), float(haloinfo_hf.attrs["HubbleParam"])

    raise KeyError("Could not find Time/HubbleParam in Haloinfo file attributes.")



# Function to get top N halos by mass PRIVOUS
def get_top_n_halos(haloinfo_data, n=10):
    top_halos = {}
    for box_key, box_halo_info_data in haloinfo_data.items():
        for mass_range_key, mass_range_data in box_halo_info_data.items():
            halo_masses = []
            for Halo_idx in mass_range_data.keys():
                halomass = mass_range_data[Halo_idx]['Halomass'][()]
                halo_mv = mass_range_data[Halo_idx]['Halo_MV'][()]
                halo_rv = mass_range_data[Halo_idx]['Halo_RV'][()]
                halo_center = mass_range_data[Halo_idx]['halo_center'][()]
                idx = int(Halo_idx.replace('halo_', ''))
                halo_center = halo_center[idx]
                halo_masses.append((Halo_idx, halomass, halo_mv, halo_rv, halo_center))
            # 排序并获取前n个Halo
            halo_masses.sort(key=lambda x: x[1], reverse=True)
            top_halo_keys = halo_masses[:n]
            if mass_range_key not in top_halos:
                top_halos[mass_range_key] = {}
            for key, halomass, halo_mv, halo_rv, halo_center in top_halo_keys:
                top_halos[mass_range_key][key] = {
                    'Halomass': halomass,
                    'Halo_MV': halo_mv,
                    'Halo_RV': halo_rv,
                    'halo_center': halo_center,
                    'sphpos': mass_range_data[key]['sphpos'][()],
                    'sphmass': mass_range_data[key]['sphmass'][()],
                    'smoothlen': mass_range_data[key]['smoothlen'][()],
                    'f_e': mass_range_data[key]['f_e'][()]
                }
    return top_halos

##############取前n个质量的halo
def get_top_n_halos_mass(haloinfo_data, n=1):
    top_halos = {}
    for box_key, box_halo_info_data in haloinfo_data.items():
        for mass_range_key, mass_range_data in box_halo_info_data.items():
            halo_masses = []
            for Halo_idx in mass_range_data.keys():
                halomass = mass_range_data[Halo_idx]['Halomass'][()]
                halo_mv = mass_range_data[Halo_idx]['Halo_MV'][()]
                halo_rv = mass_range_data[Halo_idx]['Halo_RV'][()]
                halo_center = mass_range_data[Halo_idx]['halo_center'][()]
                halo_masses.append((Halo_idx, halomass, halo_mv, halo_rv, halo_center))
            # 排序并获取前n个Halo
            halo_masses.sort(key=lambda x: x[1], reverse=True)
            top_halo_keys = halo_masses[:n]
            if mass_range_key not in top_halos:
                top_halos[mass_range_key] = {}
            for key, halomass, halo_mv, halo_rv, halo_center in top_halo_keys:
                top_halos[mass_range_key][key] = {
                    'Halomass': halomass,
                    'Halo_MV': halo_mv,
                    'Halo_RV': halo_rv,
                    'halo_center': halo_center,
                    'sphpos': mass_range_data[key]['sphpos'][()],
                    'sphmass': mass_range_data[key]['sphmass'][()],
                    'smoothlen': mass_range_data[key]['smoothlen'][()],
                    'f_e': mass_range_data[key]['f_e'][()],
                    'starmass': mass_range_data[key]['starmass'][()],
                    'starpos': mass_range_data[key]['stellarpos'][()],
                    'sphmass': mass_range_data[key]['sphmass'][()],
                    'SFR': mass_range_data[key]['SFR'][()],
                    'Z_sph': mass_range_data[key]['Z_sph'][()],
                    'Z_star': mass_range_data[key]['Z_stellar'][()]
                }
    return top_halos

##############取前n个序号的halo
def get_top_n_halos_index_number(haloinfo_data, n=10):
    top_halos = {}
    for box_key, box_halo_info_data in haloinfo_data.items():
        for mass_range_key, mass_range_data in box_halo_info_data.items():
            halo_masses = []
            for Halo_idx in mass_range_data.keys():
                halomass = mass_range_data[Halo_idx]['Halomass'][()]
                halo_mv = mass_range_data[Halo_idx]['Halo_MV'][()]
                halo_rv = mass_range_data[Halo_idx]['Halo_RV'][()]
                halo_center = mass_range_data[Halo_idx]['halo_center'][()]
                halo_masses.append((Halo_idx, halomass, halo_mv, halo_rv, halo_center))
            # 按序号排序并获取前n个Halo
            halo_masses.sort(key=lambda x: int(x[0].split('_')[1]))
            top_halo_keys = halo_masses[:n]
            if mass_range_key not in top_halos:
                top_halos[mass_range_key] = {}
            for key, halomass, halo_mv, halo_rv, halo_center in top_halo_keys:
                top_halos[mass_range_key][key] = {
                    'Halomass': halomass,
                    'Halo_MV': halo_mv,
                    'Halo_RV': halo_rv,
                    'halo_center': halo_center,
                    'sphpos': mass_range_data[key]['sphpos'][()],
                    'sphmass': mass_range_data[key]['sphmass'][()],
                    'smoothlen': mass_range_data[key]['smoothlen'][()],
                    'f_e': mass_range_data[key]['f_e'][()]
                }
    return top_halos

##############取AGN和noAGN共有的前n个序号的halo
def get_top_n_halos_index_common(haloinfo_agn, haloinfo_noagn, n=10):
    def extract_halo_info(haloinfo):
        halos = {}
        for box_key, box_halo_info_data in haloinfo.items():
            for mass_range_key, mass_range_data in box_halo_info_data.items():
                halo_masses = []
                for Halo_idx in mass_range_data.keys():
                    halomass = mass_range_data[Halo_idx]['Halomass'][()]
                    halo_mv = mass_range_data[Halo_idx]['Halo_MV'][()]
                    halo_rv = mass_range_data[Halo_idx]['Halo_RV'][()]
                    halo_center = mass_range_data[Halo_idx]['halo_center'][()]
                    halo_masses.append((Halo_idx, halomass, halo_mv, halo_rv, halo_center))
                # 按序号排序
                halo_masses.sort(key=lambda x: int(x[0].split('_')[1]))
                if mass_range_key not in halos:
                    halos[mass_range_key] = {}
                for idx, halomass, halo_mv, halo_rv, halo_center in halo_masses:
                    halos[mass_range_key][idx] = {
                        'Halomass': halomass,
                        'Halo_MV': halo_mv,
                        'Halo_RV': halo_rv,
                        'halo_center': halo_center,
                        'sphpos': mass_range_data[idx]['sphpos'][()],
                        'sphmass': mass_range_data[idx]['sphmass'][()],
                        'smoothlen': mass_range_data[idx]['smoothlen'][()],
                        'f_e': mass_range_data[idx]['f_e'][()]
                    }
        return halos

    halos_agn = extract_halo_info(haloinfo_agn)
    halos_noagn = extract_halo_info(haloinfo_noagn)

    common_halos = {}
    for mass_range_key in halos_agn.keys():
        if mass_range_key in halos_noagn:
            common_keys = set(halos_agn[mass_range_key].keys()) & set(halos_noagn[mass_range_key].keys())
            common_keys = list(common_keys)[:n]
            if common_keys:
                common_halos[mass_range_key] = common_keys

    top_halos_agn = {mass_range_key: {key: halos_agn[mass_range_key][key] for key in keys} for mass_range_key, keys in common_halos.items()}
    top_halos_noagn = {mass_range_key: {key: halos_noagn[mass_range_key][key] for key in keys} for mass_range_key, keys in common_halos.items()}

    return top_halos_agn, top_halos_noagn
#top_halos_100_AGN, top_halos_100_noAGN = get_top_n_halos_index_common(haloinfo_100_AGN, haloinfo_100_noAGN)




############寻找最近的Halo index#############
def get_top_n_halos_nearest_common(haloinfo_agn, haloinfo_noagn, n=10):
    def extract_halo_info(haloinfo):
        halos = {}
        for box_key, box_halo_info_data in haloinfo.items():
            for mass_range_key, mass_range_data in box_halo_info_data.items():
                for Halo_idx in mass_range_data.keys():
                    halomass = mass_range_data[Halo_idx]['Halomass'][()]
                    halo_mv = mass_range_data[Halo_idx]['Halo_MV'][()]
                    halo_rv = mass_range_data[Halo_idx]['Halo_RV'][()]
                    halo_center = mass_range_data[Halo_idx]['halo_center'][()]
                    if mass_range_key not in halos:
                        halos[mass_range_key] = {}
                    halos[mass_range_key][Halo_idx] = {
                        'Halomass': halomass,
                        'Halo_MV': halo_mv,
                        'Halo_RV': halo_rv,
                        'halo_center': halo_center,
                        'sphpos': mass_range_data[Halo_idx]['sphpos'][()],
                        'sphmass': mass_range_data[Halo_idx]['sphmass'][()],
                        'smoothlen': mass_range_data[Halo_idx]['smoothlen'][()],
                        'f_e': mass_range_data[Halo_idx]['f_e'][()]
                    }
        return halos

    def find_nearest_pairs(halos_agn, halos_noagn, n):
        nearest_pairs = {}
        for mass_range_key in halos_agn.keys():
            if mass_range_key in halos_noagn:
                agn_centers = [(idx, halos_agn[mass_range_key][idx]['halo_center']) for idx in halos_agn[mass_range_key].keys()]
                noagn_centers = [(idx, halos_noagn[mass_range_key][idx]['halo_center']) for idx in halos_noagn[mass_range_key].keys()]

                # 计算最近的n对
                pairs = []
                for agn_idx, agn_center in agn_centers:
                    distances = [(noagn_idx, np.linalg.norm(agn_center - noagn_center)) for noagn_idx, noagn_center in noagn_centers]
                    distances.sort(key=lambda x: x[1])
                    nearest_noagn_idx, _ = distances[0]
                    pairs.append((agn_idx, nearest_noagn_idx))

                # 选择前n个
                nearest_pairs[mass_range_key] = pairs[:n]
        return nearest_pairs

    halos_agn = extract_halo_info(haloinfo_agn)
    halos_noagn = extract_halo_info(haloinfo_noagn)

    nearest_pairs = find_nearest_pairs(halos_agn, halos_noagn, n)

    top_halos_agn = {mass_range_key: {agn_idx: halos_agn[mass_range_key][agn_idx] for agn_idx, _ in pairs} for mass_range_key, pairs in nearest_pairs.items()}
    top_halos_noagn = {mass_range_key: {noagn_idx: halos_noagn[mass_range_key][noagn_idx] for _, noagn_idx in pairs} for mass_range_key, pairs in nearest_pairs.items()}

    return top_halos_agn, top_halos_noagn



def calculate_metallicity_along_sightline(sphpos, stellarpos, metallicity_sph, metallicity_stellar, sightline_pos,sightline_radius_Mpc, Lbox):
    # 找到SPH和恒星粒子
    idx_sph, _ = find_particles_within_RV_sph(sphpos, sightline_pos, sightline_radius_Mpc, Lbox)
    idx_stellar, _ = find_particles_within_RV_sph(stellarpos, sightline_pos, sightline_radius_Mpc, Lbox)

    sph_metallicity = metallicity_sph[idx_sph]
    stellar_metallicity = metallicity_stellar[idx_stellar]

    # 计算金属丰度的平均值
    combined_metallicity = np.concatenate([sph_metallicity, stellar_metallicity])
    mean_metallicity_along_sightline = np.mean(combined_metallicity)
    
    return mean_metallicity_along_sightline


def calculate_sightline_directions(num_directions=20):
    """
    Generate direction vectors (in spherical coordinates) for the sightlines.
    """
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    directions = []
    for theta in angles:
        # Generate 20 direction vectors in each of xy, yz, xz planes
        directions.append([np.cos(theta), np.sin(theta), 0])   # xy-plane
        directions.append([np.cos(theta), 0, np.sin(theta)])   # xz-plane
        directions.append([0, np.cos(theta), np.sin(theta)])   # yz-plane
    return directions

def calculate_sightline_directions(num_directions=20):
    """
    Generate direction vectors and their respective plane labels.

    Returns:
        tuple: Two lists - direction vectors and plane labels.
    """
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
    directions = []
    labels = []

    for theta in angles:
        directions.append([np.cos(theta), np.sin(theta), 0])
        labels.append('xy')
        directions.append([np.cos(theta), 0, np.sin(theta)])
        labels.append('xz')
        directions.append([0, np.cos(theta), np.sin(theta)])
        labels.append('yz')
    return directions, labels

def calculate_total_sfr_and_metallicity(
    halo_center, 
    sphpos, 
    sphmass_cgs,
    stellarpos, 
    stellarmass_cgs,
    star_formation_rate,
    metallicity_sph, 
    metallicity_stellar, 
    Lbox,
    HaloRV,
    C_R_max = 2
):
    """
    Calculate total SFR and average metallicity within C_R_max*RV of the halo center.

    Parameters:
        halo_center: Array-like, the center of the halo.
        sphpos: SPH particle positions.
        stellarpos: Stellar particle positions.
        sphmass_cgs: Mass of SPH particles in unit gram.
        stellarmass_cgs: Mass of stellar particles in unit_gram.
        star_formation_rate: SFR of SPH particles.
        metallicity_sph: Metallicity of SPH particles.
        metallicity_stellar: Metallicity of stellar particles.
        Lbox: Size of the simulation box.
        HaloRV: Halo virial radius in Mpc.

    Returns:
        total_SFR: Total star formation rate within 5 RV.
        avg_metallicity_sph: Mass-weighted average metallicity of gas within 5 RV.
        avg_metallicity_stellar: Mass-weighted average metallicity of stars within 5 RV.
    """
    total_SFR = 0.0
    total_sph_mass_in_5RV = 0.0
    total_stellar_mass_in_5RV = 0.0
    weighted_metallicity_sph = 0.0
    weighted_metallicity_stellar = 0.0
    # Define search radius (5 RV in Mpc)
    search_radius = C_R_max * HaloRV

    # Search for SPH particles within 5 RV
    print(f"Type of halo_center: {type(halo_center)}, Value: {halo_center}")
    print(f"Type of Lbox: {type(Lbox)}, Value: {Lbox}")
    sys.stdout.flush()
    idx_sph_5RV, _ = find_particles_within_RV_sph(sphpos, halo_center, search_radius, Lbox)
    for idx in idx_sph_5RV:
        total_SFR += star_formation_rate[idx]
        total_sph_mass_in_5RV += sphmass_cgs[idx]
        weighted_metallicity_sph += sphmass_cgs[idx] * metallicity_sph[idx]

    # Search for stellar particles within 5 RV
    idx_stellar_5RV, _ = find_particles_within_RV_sph(stellarpos, halo_center, search_radius, Lbox)
    print(f"stellar_num_5RV:{len(idx_stellar_5RV)}")
    for idx in idx_stellar_5RV:
        total_stellar_mass_in_5RV += stellarmass_cgs[idx]
        weighted_metallicity_stellar += stellarmass_cgs[idx] * metallicity_stellar[idx]
    
    # Calculate mass-weighted average metallicities
    avg_metallicity_sph = (
        weighted_metallicity_sph / total_sph_mass_in_5RV if total_sph_mass_in_5RV > 0 else 0.0
    )
    avg_metallicity_stellar = (
        weighted_metallicity_stellar / total_stellar_mass_in_5RV if total_stellar_mass_in_5RV > 0 else 0.0
    )

    return total_SFR, avg_metallicity_sph, avg_metallicity_stellar


def calculate_sightline_dm_stellar(
    direction,  
    label,                        
    halo_center, 
    sphpos, 
    stellarpos, 
    sphmass_cgs, 
    stellarmass_cgs, 
    smoothlen,
    f_e, 
    HaloRV, 
    Lbox,
    radii,
    num_bins, 
    sightline_radius_kpc, 
    metallicity_sph, 
    metallicity_stellar,
    C_R_max = 2
):
    """
    Calculate the DM, stellar mass, and metallicity along a single sightline direction, with a given sightline radius for stellar mass.
    """
    mp_cgs = 1.67e-24 
    direction = np.array(direction)
    if label == 'xy':  # xy-plane
        # z-axis as vertical direction
        vertical_axis = np.array([0, 0, 1])
    elif label == "xz":  # xz-plane
        # y-axis as vertical direction
        vertical_axis = np.array([0, 1, 0])
    elif label == "yz":  # yz-plane
        # x-axis as vertical direction
        vertical_axis = np.array([1, 0, 0])
    
    #bin_edges = np.linspace(-C_R_max * HaloRV, C_R_max * HaloRV, num_bins)
    #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers for integration

    ########################Previous adaptive binning approach (commented out)########################
    # Create adaptive bin edges with more bins near the center
    # 让 frac 随着 R_200 变化，同时确保最小分辨率 >= 3.38 ckpc

    #radii = np.linspace(0, C_R_max * HaloRV, num_radii)
    #min_bin_size = 3.38 / 1e3  # 转换为 Mpc
    #frac = max(min_bin_size / HaloRV, 0.1 * (C_R_max / HaloRV))  # 既满足解析需求，又考虑 halo 质量
    #adaptive_bins = np.geomspace(frac * HaloRV, C_R_max * HaloRV, num_bins//2)
    #adaptive_bins = np.concatenate((-adaptive_bins[::-1], adaptive_bins))  # 负区间对称
    # Bin centers
    #bin_centers = (adaptive_bins[:-1] + adaptive_bins[1:]) / 2  # Bin centers for integration
    
    ###################### LOS adaptive bins (keep) ######################

    # ---------- 1) build LOS adaptive_bins (symmetric) ----------
    min_bin_size = 3.38 / 1e3  # Mpc  (3.38 kpc)
    frac = max(min_bin_size / HaloRV, 0.1 * (C_R_max / HaloRV))  # dimensionless
    r_min = frac * HaloRV
    r_max = C_R_max * HaloRV

    # positive edges only (avoid 0)
    adaptive_pos = np.geomspace(r_min, r_max, num_bins // 2)  # length ~ num_bins/2
    adaptive_bins = np.concatenate((-adaptive_pos[::-1], adaptive_pos))  # symmetric, NO zero edge

    bin_centers = 0.5 * (adaptive_bins[:-1] + adaptive_bins[1:])  # LOS bin centers
    dl = np.diff(adaptive_bins) * 1e6  # Mpc -> pc (your comment says pc; 1e6 pc per Mpc)

    # Initialize arrays for radial results
    DM = np.zeros(len(radii))  # Corresponds to radii
    stellar_mass_column = np.zeros(len(radii))  # Corresponds to radii
    metallicity_1d = np.zeros(len(radii))  # Corresponds to radii
    
    # Define the sightline cross-sectional area (500pc radius cylinder)
    sightline_radius_Mpc = sightline_radius_kpc / 1e3  # Convert to Mpc
    sightline_area = np.pi * (sightline_radius_Mpc*MpcTocm) ** 2  # Circular cross-sectional area of the cylinder
    
    # Precompute spatial tree for sphpos
    sphpos_wrapped = np.mod(sphpos, Lbox)
    stellarpos_wrapped = np.mod(stellarpos, Lbox)

    tree_sph = cKDTree(sphpos_wrapped, boxsize=Lbox)
    tree_stellar = cKDTree(stellarpos_wrapped, boxsize=Lbox)
    
    kernel_cutoff_factor = 2.5  # 平滑核的截断因子
    query_radius = min(np.percentile(smoothlen, 90), kernel_cutoff_factor * np.median(smoothlen))
    stellar_num = len(stellarpos)
    radii_iterator = tqdm(enumerate(radii), total=len(radii), desc="Processing radii", disable=True)
    #dl = (bin_edges[1] - bin_edges[0])*1e6 #unit in kpc
    # 计算每个 bin 的宽度（适应非均匀 bin）
    dl = np.diff(adaptive_bins) * 1e6  # 转换单位：Mpc → pc
    for i, radius in radii_iterator:
        sightline_pos_plane = halo_center + direction * radius
        local_dm = 0.0
        local_stellar_mass = 0.0
        local_metallicity = 0.0

        #################Latest modification#####################################
        sightline_pos = sightline_pos_plane[None, :] + vertical_axis[None, :] * bin_centers[:, None]
        sightline_pos = np.mod(sightline_pos, Lbox)  # periodic wrap, once

        # ✅ 一次 KDTree 查询：返回 list-of-lists
        idx_lists = tree_sph.query_ball_point(sightline_pos, query_radius)
        idx_stellar_lists = tree_stellar.query_ball_point(sightline_pos, sightline_radius_Mpc)
        #########################################################################

        for j, center in enumerate(bin_centers):  # 这里添加 j 作为索引
            # 找到 SPH 粒子
            ##################last version#########
            #sightline_pos = sightline_pos_plane + vertical_axis * center
            #idx_LC, dist_LC = find_particles_within_RV_sph_tree(tree_sph, sphpos, sightline_pos, query_radius, Lbox)
            #for m in range(len(idx_LC)):
            #    if dist_LC[m] < smoothlen[idx_LC[m]]:
            #        weight = M6(dist_LC[m] * MpcTocm, 0, smoothlen[idx_LC[m]] * MpcTocm, 3)
            #        ne = sphmass_cgs[idx_LC[m]] * f_e[idx_LC[m]] / mp_cgs * weight
                    #local_dm += ne * dl 
            #        local_dm += ne * dl[j]  
            ###################################

            idx = idx_lists[j]
            if len(idx) == 0:
                continue
            idx = np.asarray(idx, dtype=np.int64)
            dr = sphpos_wrapped[idx] - sightline_pos[j]
            dr = (dr + 0.5 * Lbox) % Lbox - 0.5 * Lbox
            dist = np.sqrt((dr * dr).sum(axis=1))  # Mpc
            # 只保留 dist < smoothlen
            h = smoothlen[idx]
            mask = dist < h
            if np.any(mask):
                idx2 = idx[mask]
                dist2 = dist[mask]
                h2 = h[mask]
                dist2_cm = dist2 * MpcTocm
                h2_cm    = h2    * MpcTocm
                # 这里直接用 numba 累加（每个粒子一遍）
                for kk in range(dist2.size):
                    weight = M6(dist2_cm[kk], 0, h2_cm[kk], 3)
                    ne = sphmass_cgs[idx2[kk]] * f_e[idx2[kk]] / mp_cgs * weight
                    local_dm += ne * dl[j]  # ✅ dl 用 pc（与你 DM 单位一致）
    
            # Find particles in stellar using KDTree
            #if len(idx_stellar) != 0:
            #    print(f"stellar_num : {len(idx_stellar)}")
            #idx_stellar = np.array(idx_stellar)
            # 筛选柱高度范围内的粒子
            idx_stellar = idx_stellar_lists[j]
            if len(idx_stellar) == 0:
                continue
            # 计算粒子到柱中心的轴向距离
            idx_stellar = np.asarray(idx_stellar, dtype=np.int64)
            relative_positions = stellarpos[idx_stellar] - sightline_pos[j]  # 粒子相对位置
            relative_positions = np.mod(relative_positions + Lbox / 2, Lbox) - Lbox / 2  # 考虑周期边界条件
            axis_distances = np.abs(np.dot(relative_positions, direction))  # 投影到柱轴方向
            #bin_height = (bin_edges[1] - bin_edges[0])
            bin_height = dl[j]  #  这里替换 bin_height
            # 筛选在柱高度范围内的粒子
            valid_idx = idx_stellar[axis_distances <= bin_height / 2]  # 仅保留柱内粒子

            # 计算柱密度和金属丰度
            if len(valid_idx) > 0:
                local_stellar_mass += np.sum(stellarmass_cgs[valid_idx]) / sightline_area
                local_metallicity += np.sum(metallicity_stellar[valid_idx])/len(valid_idx)

        DM[i] = local_dm
        stellar_mass_column[i] = local_stellar_mass
        metallicity_1d[i] = local_metallicity / max(1, len(bin_centers))
    
    return DM, stellar_mass_column, metallicity_1d

def calculate_column_density_v2(halo, Lbox, radii, C_R_max=2, num_directions=20, num_bins=100, 
                                process_n_jobs=5, sightline_radius_kpc=30, h = 0.67742, a = 1.0):
    """
    Calculate DM along multiple sightlines in different directions from the halo center,
    and also calculate the stellar mass column density along those sightlines.
    This version also computes star formation rate and metallicity.
    """

    # 初始化 MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # 仅 rank 0 显示进度条
    progress_bar_enabled = (rank == 0)
    HaloRV = np.array(halo['Halo_RV'])
    halo_center = np.array(halo['halo_center'])
    sphpos = np.array(halo['sphpos'])
    stellarpos = np.array(halo['starpos'])
    print(f"max sphpos:{np.max(sphpos)}")
    sys.stdout.flush()
    sphmass_cgs = np.array(halo['sphmass'] * SPHmasstocgs)
    stellarmass_cgs = np.array(halo['starmass'] * SPHmasstocgs)
    smoothlen = np.array(halo['smoothlen'])
    f_e = np.array(halo['f_e'])
    
    # Added properties: star formation rate and metallicity
    star_formation_rate = halo['SFR']
    metallicity_sph = halo['Z_sph']
    metallicity_stellar = halo['Z_star']
    
    # First, calculate total SFR and average metallicity within C_R_max*RV of the halo center
    total_SFR, avg_metallicity_sph, avg_metallicity_stellar = calculate_total_sfr_and_metallicity(
        halo_center, 
        sphpos, 
        sphmass_cgs,
        stellarpos, 
        stellarmass_cgs,
        star_formation_rate, 
        metallicity_sph, 
        metallicity_stellar,
        Lbox/h*a, 
        HaloRV,
        C_R_max,
    )
    
    directions, labels = calculate_sightline_directions(num_directions)
    if rank == 0:
        # 在rank 0上显示进度条
        # We calculate DM, stellar mass, and metallicity for each direction in parallel
        with tqdm_joblib(tqdm(desc="Calculating sightlines", total=len(directions), disable=True)) as progress_bar:
            results = Parallel(n_jobs=process_n_jobs)(delayed(calculate_sightline_dm_stellar)(
                direction, 
                label,
                halo_center, 
                sphpos, 
                stellarpos, 
                sphmass_cgs, 
                stellarmass_cgs, 
                smoothlen, 
                f_e, 
                HaloRV, 
                Lbox/h*a, 
                radii,
                num_bins, 
                sightline_radius_kpc,
                metallicity_sph, 
                metallicity_stellar,
                C_R_max
            ) for direction, label in zip(directions, labels))
    else:
        with tqdm_joblib(tqdm(desc="Calculating sightlines", total=len(directions), disable=True)) as progress_bar:
            results = Parallel(n_jobs=process_n_jobs)(delayed(calculate_sightline_dm_stellar)(
                direction, 
                label,
                halo_center, 
                sphpos, 
                stellarpos, 
                sphmass_cgs, 
                stellarmass_cgs, 
                smoothlen, 
                f_e, 
                HaloRV, 
                Lbox/h*a, 
                radii,
                num_bins, 
                sightline_radius_kpc,
                metallicity_sph, 
                metallicity_stellar,
                C_R_max
            ) for direction, label in zip(directions, labels))

    
    
    return results, total_SFR, avg_metallicity_sph, avg_metallicity_stellar  # Return all results for further processing or plotting




# Now we need to modify the plotting functions to visualize these new results



def save_to_hdf5(file_path, column_density_map, bin_centers, mean_DM, median_DM, lower_bound, upper_bound, halo):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('column_density_map', data=column_density_map)
        f.create_dataset('bin_centers', data=bin_centers)
        f.create_dataset('mean_DM', data=mean_DM)
        f.create_dataset('median_DM', data=median_DM)
        f.create_dataset('lower_bound', data=lower_bound)
        f.create_dataset('upper_bound', data=upper_bound)

        # Set attributes
        f.attrs['halomass'] = halo['Halomass']
        f.attrs['HaloMV'] = halo['Halo_MV']
        f.attrs['HaloRV'] = halo['Halo_RV']


def precreate_halo_layout(hf, all_halos, halo_data, num_radii, num_sightlines):
    """Create all HDF5 metadata collectively before owner ranks fill values."""
    one_d_fields = [
        'bin_centers',
        'lin_mean_DM', 'log_mean_DM', 'median_DM', 'lower_bound_DM', 'upper_bound_DM',
        'linear_mean_stellar_column_density', 'log_mean_stellar_column_density',
        'median_stellar_column_density', 'lower_bound_stellar_column_density', 'upper_bound_stellar_column_density',
        'linear_mean_metallicity', 'log_mean_metallicity', 'median_metallicity',
        'lower_bound_metallicity', 'upper_bound_metallicity',
    ]
    two_d_fields = ['DM_sightlines', 'stellar_mass_sightlines', 'metallicity_sightlines']

    for mass_bin, halo_key, _ in all_halos:
        if mass_bin not in halo_data or halo_key not in halo_data[mass_bin]:
            continue
        halo = halo_data[mass_bin][halo_key]
        halo_index = int(halo_key.replace('halo_', ''))
        grp = hf.require_group(f"halo_{halo_index}")

        # Pre-fill metadata placeholders collectively.
        grp.attrs['Halomass'] = halo['Halomass'] * 1e10
        grp.attrs['HaloMV'] = halo['Halo_MV'] * 1e10
        grp.attrs['HaloRV'] = halo['Halo_RV']
        grp.attrs['Total_SFR'] = np.nan
        grp.attrs['Average_Metallicity_SPH'] = np.nan
        grp.attrs['Average_Metallicity_Stellar'] = np.nan

        for name in one_d_fields:
            if name not in grp:
                grp.create_dataset(name, shape=(num_radii,), dtype=np.float64)
        for name in two_d_fields:
            if name not in grp:
                grp.create_dataset(name, shape=(num_sightlines, num_radii), dtype=np.float64)


def process_halo(halo_key, halo, hf, save_path, Lbox=100, AGN_info="n", process_id=0, process_n_jobs=5, 
                 C_R_max=2,  num_radii=30, num_direction =20, h = 0.67742, a = 1.0,
                 radial_bin_mode="uniform"):
    halomass = halo['Halomass'] * 1e10
    HaloMV = halo['Halo_MV'] * 1e10
    log_halomass = np.log10(halomass)
    HaloRV = halo['Halo_RV']# Mpc
    halo_index = int(halo_key.replace('halo_', ''))
    num_bins = int(np.clip(2*C_R_max * HaloRV / (3.38 / h) * 1e3, 80, 600))
    radii = build_radial_bin_centers(HaloRV, C_R_max, num_radii, radial_bin_mode=radial_bin_mode)
    sightline_radius_kpc = min(0.05 * HaloRV * 1e3, 30.0)
    # Calculate DM, stellar mass column density, metallicity distribution, and total SFR
    results, total_SFR, avg_metallicity_sph, avg_metallicity_stellar = calculate_column_density_v2(
        halo, Lbox, radii, C_R_max, num_directions=num_direction, num_bins=num_bins,
        process_n_jobs=process_n_jobs, sightline_radius_kpc=sightline_radius_kpc, h = h, a = a
    )

    # Unpack results for further processing
    DM_list, stellar_mass_column_list, metallicity_1d_list = zip(*results)

    DM_array = np.array(DM_list)
    stellar_mass_array = np.array(stellar_mass_column_list)
    metallicity_array = np.array(metallicity_1d_list)

    bin_centers = radii / HaloRV  # Normalize by HaloRV using the true radial sampling

    # Calculate mean and error bands
    lin_mean_DM = linear_mean_without_zeros(DM_array)
    log_mean_DM = log_mean_without_zeros(DM_array)
    median_DM = median_without_zeros(DM_array)
    lower_bound_DM, upper_bound_DM = calculate_error_bands(DM_array)

    lin_mean_stellar_mass = linear_mean_without_zeros(stellar_mass_array)
    log_mean_stellar_mass = log_mean_without_zeros(stellar_mass_array)
    median_stellar_mass = median_without_zeros(stellar_mass_array)
    lower_bound_stellar_mass, upper_bound_stellar_mass = calculate_error_bands(stellar_mass_array)

    lin_mean_metallicity = linear_mean_without_zeros(metallicity_array)
    log_mean_metallicity = log_mean_without_zeros(metallicity_array)
    median_metallicity = median_without_zeros(metallicity_array)
    lower_bound_metallicity, upper_bound_metallicity = calculate_error_bands(metallicity_array)

    # Metadata was pre-created collectively. Owner rank only fills values here.
    grp = hf[f"halo_{halo_index}"]
    
    # Store attributes
    grp.attrs['Halomass'] = halomass
    grp.attrs['HaloMV'] = HaloMV
    grp.attrs['HaloRV'] = HaloRV
    grp.attrs['Total_SFR'] = total_SFR
    grp.attrs['Average_Metallicity_SPH'] = avg_metallicity_sph
    grp.attrs['Average_Metallicity_Stellar'] = avg_metallicity_stellar

    # Save DM, stellar mass, and metallicity mean and error bands
    grp['bin_centers'][...] = bin_centers
    grp['lin_mean_DM'][...] = lin_mean_DM
    grp['log_mean_DM'][...] = log_mean_DM
    grp['median_DM'][...] = median_DM
    grp['lower_bound_DM'][...] = lower_bound_DM
    grp['upper_bound_DM'][...] = upper_bound_DM

    grp['linear_mean_stellar_column_density'][...] = lin_mean_stellar_mass
    grp['log_mean_stellar_column_density'][...] = log_mean_stellar_mass
    grp['median_stellar_column_density'][...] = median_stellar_mass
    grp['lower_bound_stellar_column_density'][...] = lower_bound_stellar_mass
    grp['upper_bound_stellar_column_density'][...] = upper_bound_stellar_mass

    grp['linear_mean_metallicity'][...] = lin_mean_metallicity
    grp['log_mean_metallicity'][...] = log_mean_metallicity
    grp['median_metallicity'][...] = median_metallicity
    grp['lower_bound_metallicity'][...] = lower_bound_metallicity
    grp['upper_bound_metallicity'][...] = upper_bound_metallicity

    grp['DM_sightlines'][...] = DM_array
    grp['stellar_mass_sightlines'][...] = stellar_mass_array
    grp['metallicity_sightlines'][...] = metallicity_array

    print(f"Halo {halo_index} processed and saved successfully.")

    # ---- Plot DM, stellar mass, and metallicity curves ----

    # Plot DM curve
    plt.figure()
    dm_plot = _mask_nonpositive(DM_array)
    dm_lin_mean = _mask_nonpositive(lin_mean_DM)
    dm_log_mean = _mask_nonpositive(log_mean_DM)
    dm_median = _mask_nonpositive(median_DM)
    dm_lower = _mask_nonpositive(lower_bound_DM)
    dm_upper = _mask_nonpositive(upper_bound_DM)
    for i in range(len(dm_plot)):
        if i == 0:
            plt.plot(bin_centers, dm_plot[i], color = "k", alpha=0.15)
        else:
            plt.plot(bin_centers, dm_plot[i], color = "k", alpha=0.15)
    plt.fill_between(bin_centers, dm_lower, dm_upper, color='red', alpha=0.3, label='1-$\sigma$ DM Error')
    plt.plot(bin_centers, dm_lin_mean, label='Linear Mean D')
    plt.plot(bin_centers, dm_log_mean, label='Log Mean DM')
    plt.plot(bin_centers, dm_median, label='Median DM')
    plt.axvline(x=1, color='r', linestyle='--', label=f'HaloRV: {HaloRV:.1f} kpc')
    plt.title(f'Halo {halo_index} - DM Distribution')
    plt.xlabel('$b / R_{200}$')
    plt.ylabel('DM [pc cm$^{-3}$]')
    plt.xlim(0, C_R_max)
    plt.legend(loc='best', fontsize='small')
    if _has_positive_values(dm_plot, dm_lin_mean, dm_log_mean, dm_median, dm_lower, dm_upper):
        plt.yscale('log')
    plt.savefig(f'{save_path}/figures/{halo_index}_{Lbox}_{AGN_info}_DM_distribution.png')
    plt.show()
    plt.close()

    # Plot stellar mass column density curve
    plt.figure()
    stellar_plot = _mask_nonpositive(stellar_mass_array)
    stellar_lin_mean = _mask_nonpositive(lin_mean_stellar_mass)
    stellar_log_mean = _mask_nonpositive(log_mean_stellar_mass)
    stellar_median = _mask_nonpositive(median_stellar_mass)
    stellar_lower = _mask_nonpositive(lower_bound_stellar_mass)
    stellar_upper = _mask_nonpositive(upper_bound_stellar_mass)
    for i in range(len(stellar_plot)):
        if i == 0:
            plt.plot(bin_centers, stellar_plot[i], color = "k", alpha=0.15)
        else:
            plt.plot(bin_centers, stellar_plot[i], color = "k", alpha=0.15)
    plt.plot(bin_centers, stellar_lin_mean, label='linear Mean')
    plt.plot(bin_centers, stellar_log_mean, label='Log Mean')
    plt.plot(bin_centers, stellar_median, label='Median')
    plt.fill_between(bin_centers, stellar_lower, stellar_upper, color='red', alpha=0.3, label='1-$\sigma$ Stellar Mass Error')
    plt.axvline(x=1, color='r', linestyle='--', label=f'HaloRV: {HaloRV:.1f} kpc')
    plt.title(f'Halo {halo_index} - Stellar Column Density Distribution')
    plt.xlabel('$b / R_{200}$')
    plt.ylabel('Stellar Mass Column [g/cm²]')
    plt.xlim(0, C_R_max)
    plt.legend(loc='best', fontsize='small')
    if _has_positive_values(stellar_plot, stellar_lin_mean, stellar_log_mean, stellar_median, stellar_lower, stellar_upper):
        plt.yscale('log')
    plt.savefig(f'{save_path}/figures/{halo_index}_{Lbox}_{AGN_info}_stellar_mass_distribution.png')
    plt.show()
    plt.close()

    # Plot metallicity curve
    plt.figure()
    metal_plot = _mask_nonpositive(metallicity_array)
    metal_lin_mean = _mask_nonpositive(lin_mean_metallicity)
    metal_log_mean = _mask_nonpositive(log_mean_metallicity)
    metal_median = _mask_nonpositive(median_metallicity)
    metal_lower = _mask_nonpositive(lower_bound_metallicity)
    metal_upper = _mask_nonpositive(upper_bound_metallicity)
    for i in range(len(metal_plot)):
        if i == 0:
            plt.plot(bin_centers, metal_plot[i], color = "k", alpha=0.15)
        else:
            plt.plot(bin_centers, metal_plot[i], color = "k", alpha=0.15)
    plt.plot(bin_centers, metal_lin_mean, label='linear Mean')
    plt.plot(bin_centers, metal_log_mean, label='Log Mean')
    plt.plot(bin_centers, metal_median, label='Median')
    plt.fill_between(bin_centers, metal_lower, metal_upper, color='red', alpha=0.3, label='1-$\sigma$ Metallicity Error')
    plt.axvline(x=1, color='r', linestyle='--', label=f'HaloRV: {HaloRV:.1f} kpc')
    plt.title(f'Halo {halo_index} - Metallicity Distribution')
    plt.xlabel('$b / R_{200, c}$')
    plt.ylabel('Metallicity')
    plt.xlim(0, C_R_max)
    plt.legend(loc='best', fontsize='small')
    if _has_positive_values(metal_plot, metal_lin_mean, metal_log_mean, metal_median, metal_lower, metal_upper):
        plt.yscale('log')
    plt.savefig(f'{save_path}/figures/{halo_index}_{Lbox}_{AGN_info}_metallicity_distribution.png')
    plt.show()
    plt.close()



############## 基于Halo的MPI并行 ###################
def plot_halo_maps(halo_data, save_path, Lbox=500, AGN_info="n",  mass_range_label = "15", num_cores = 60,
                   C_R_max =  2, num_radii=30, num_direction = 20, h = 0.67742, a = 1.0,
                   radial_bin_mode="inner", h5_write_mode="append", mpi_io_mode="mpio"):
    process_id = 0
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mass_bin_label = mass_range_label
    if mass_bin_label == "14-15":
        mass_bins_to_process = ['1.00e+14-3.16e+15']
    elif mass_bin_label == "13-14":
        mass_bins_to_process = ['1.00e+13-3.16e+13','3.16e+13-1.00e+14']
    elif mass_bin_label == "12-13":
        mass_bins_to_process = ['1.00e+12-3.16e+12','3.16e+12-1.00e+13']
    elif mass_bin_label == "11-12":
       mass_bins_to_process = ['1.00e+11-3.16e+11','3.16e+11-1.00e+12']
    elif mass_bin_label == "10-11":
       mass_bins_to_process = ['1.00e+10-3.16e+10','3.16e+10-1.00e+11']
    else:
        mass_bins_to_process = ['1.00e+10-3.16e+10','3.16e+10-1.00e+11', '1.00e+11-3.16e+11','3.16e+11-1.00e+12', '1.00e+12-3.16e+12','3.16e+12-1.00e+13', '1.00e+13-3.16e+13','3.16e+13-1.00e+14', '1.00e+14-3.16e+15']
    print(f"MPI rank size: {size}")
    # 将所有halo数据收集到一个列表中
    all_halos = []
    for mass_bin, halos in halo_data.items():
        if mass_bin in mass_bins_to_process:
            for halo_key in halos.keys():
                all_halos.append((mass_bin, halo_key, AGN_info))
    print(f"tot_halo_num:{len(all_halos)}")
    #print(f"all_halos:{all_halos}")
    sys.stdout.flush()
    # Keep pre-created 2D dataset shape consistent with actual direction generator.
    directions, labels = calculate_sightline_directions(num_direction)
    num_sightlines = len(directions)
    file_suffix = "_inner" if radial_bin_mode == "inner" else ""
    output_h5 = f'{save_path}/halos_data_{mass_bin_label}_200{file_suffix}.h5'
    file_mode = "a" if h5_write_mode == "append" else "w"

    if mpi_io_mode == "mpio":
        with h5py.File(output_h5, file_mode, driver='mpio', comm=comm) as hf:
            precreate_halo_layout(hf, all_halos, halo_data, num_radii, num_sightlines)
            comm.Barrier()

            for i, (mass_bin, halo_key, AGN_info) in enumerate(all_halos):
                if i % size != rank:
                    continue
                print(f"processing halo {i}")
                if mass_bin in halo_data and halo_key in halo_data[mass_bin]:
                    halo = halo_data[mass_bin][halo_key]
                    tag = "n" if AGN_info == "n" else "f"
                    process_halo(halo_key, halo, hf, save_path, Lbox, tag, i, num_cores, C_R_max, num_radii, num_direction, h = h, a = a,
                                radial_bin_mode=radial_bin_mode)
            comm.Barrier()
    elif mpi_io_mode == "merge":
        owned_halos = [
            (i, mass_bin, halo_key, agn_tag)
            for i, (mass_bin, halo_key, agn_tag) in enumerate(all_halos)
            if i % size == rank
        ]

        tmp_h5 = f"{output_h5}.rank{rank}.tmp.h5"
        with h5py.File(tmp_h5, "w") as hf_rank:
            rank_all_halos = [(mass_bin, halo_key, agn_tag) for _, mass_bin, halo_key, agn_tag in owned_halos]
            precreate_halo_layout(hf_rank, rank_all_halos, halo_data, num_radii, num_sightlines)

            for i, mass_bin, halo_key, agn_tag in owned_halos:
                print(f"processing halo {i}")
                if mass_bin in halo_data and halo_key in halo_data[mass_bin]:
                    halo = halo_data[mass_bin][halo_key]
                    tag = "n" if agn_tag == "n" else "f"
                    process_halo(halo_key, halo, hf_rank, save_path, Lbox, tag, i, num_cores, C_R_max, num_radii, num_direction, h = h, a = a,
                                radial_bin_mode=radial_bin_mode)

        comm.Barrier()

        if rank == 0:
            final_mode = file_mode
            with h5py.File(output_h5, final_mode) as hf_final:
                for r in range(size):
                    rank_file = f"{output_h5}.rank{r}.tmp.h5"
                    if not os.path.exists(rank_file):
                        continue
                    with h5py.File(rank_file, "r") as hf_rank:
                        for key in hf_rank.keys():
                            if key in hf_final:
                                del hf_final[key]
                            hf_rank.copy(key, hf_final)
            for r in range(size):
                rank_file = f"{output_h5}.rank{r}.tmp.h5"
                if os.path.exists(rank_file):
                    os.remove(rank_file)

        comm.Barrier()
    else:
        raise ValueError(f"Unsupported mpi_io_mode: {mpi_io_mode}")

    print(f"All rank has finished its assigned tasks.")
    comm.Barrier()
    if rank == 0:
        with h5py.File(output_h5, 'r') as hf_check:
            n_halos = len([k for k in hf_check.keys() if k.startswith("halo_")])
        print(f"Final halo groups in {os.path.basename(output_h5)}: {n_halos}")
   




##############Halo数据的MPI并行###################
def plot_halo_maps_halos(halo_data_noagn, halo_data_agn, save_path_noagn, save_path_agn, Lbox=500):
    num_cores = multiprocessing.cpu_count()
    #num_cores = 15
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"MPI rank size: {size}")
    mass_bins_to_process = ['3.16e+14-3.16e+15']

    all_halos = []
    for mass_bin, halos in halo_data_noagn.items():
        if mass_bin in mass_bins_to_process:
            for halo_key in halos.keys():
                all_halos.append((mass_bin, halo_key, "noagn"))
    
    for mass_bin, halos in halo_data_agn.items():
        if mass_bin in mass_bins_to_process:
            for halo_key in halos.keys():
                all_halos.append((mass_bin, halo_key, "agn"))
    print(f"tot_halo_num:{len(all_halos)}")
    print(f"all_halos:{all_halos[:10]}")
    for i, (mass_bin, halo_key, agn_info) in enumerate(all_halos):
        if i % size == rank:
            if agn_info == "noagn":
                if mass_bin in halo_data_noagn and halo_key in halo_data_noagn[mass_bin]:
                    halo = halo_data_noagn[mass_bin][halo_key]
                    process_halo(halo_key, halo, save_path_noagn, Lbox, "n", i, num_cores)
                    #except Exception as e:
                    #    print(f"Error processing {halo_key} in mass bin {mass_bin} (noAGN): {e}")
            else:
                if mass_bin in halo_data_agn and halo_key in halo_data_agn[mass_bin]:
                    halo = halo_data_agn[mass_bin][halo_key]   
                    process_halo(halo_key, halo, save_path_agn, Lbox, "f", i, num_cores)
                    #except Exception as e:
                        #print(f"Error processing {halo_key} in mass bin {mass_bin} (AGN): {e} {i}")

haloinfo_100_noAGN, haloinfo_100_AGN = load_halo_data(directory_path, snap_tag)
if agn_info_mode == "n":
    a, h = read_scale_factors_from_haloinfo(haloinfo_100_noAGN)
else:
    a, h = read_scale_factors_from_haloinfo(haloinfo_100_AGN)
#top_halos_500_noAGN = get_top_n_halos(haloinfo_500_noAGN)
#num_cores = multiprocessing.cpu_count() 
#print(f"num_cores: {num_cores}")
#sys.stdout.flush()

#plot_halo_maps(top_halos_500_noAGNs, save_path, 500, "n")


#top_halos_100_AGN = get_top_n_halos_index(haloinfo_100_AGN)
#num_cores = multiprocessing.cpu_count() 
#print(f"num_cores: {num_cores}")
#sys.stdout.flush()

#plot_halo_maps_rest(top_halos_100_AGN, save_path_100f, 100, "f")

#top_halos_100_AGN = get_top_n_halos(haloinfo_100_AGN)
if agn_info_mode in ("f", "both"):
    top_halos_100_AGN = get_top_n_halos_mass(haloinfo_100_AGN, halo_num_analysis)
if agn_info_mode in ("n", "both"):
    top_halos_100_noAGN = get_top_n_halos_mass(haloinfo_100_noAGN, halo_num_analysis)
#top_halos_100_AGN, top_halos_100_noAGN = get_top_n_halos_index_common(haloinfo_100_AGN, haloinfo_100_noAGN)
#top_halos_100_AGN, top_halos_100_noAGN = get_top_n_halos_nearest_common(haloinfo_100_AGN, haloinfo_100_noAGN)
# 打印 CPU 核心数
#num_cores = multiprocessing.cpu_count() 
num_cores = num_cores_per_rank
#num_cores = 70
print(f"num_cores: {num_cores}")
sys.stdout.flush()
# 绘制 Halo 图像
num_radii = 150
num_direction = 20
C_R_max = 10


#plot_halo_maps(top_halos_100_AGN, save_path_100f, 100, "f")

start_time = time.time()
if agn_info_mode in ("f", "both"):
    plot_halo_maps(top_halos_100_AGN, storage_path_100f, 100, "f", mass_range_label, num_cores,
                   C_R_max=C_R_max, num_radii=num_radii, num_direction=num_direction, h=h, a=a,
                   radial_bin_mode=radial_bin_mode, h5_write_mode=h5_write_mode, mpi_io_mode=mpi_io_mode)
if agn_info_mode in ("n", "both"):
    plot_halo_maps(top_halos_100_noAGN, storage_path_100n, 100, "n", mass_range_label, num_cores,
                   C_R_max=C_R_max, num_radii=num_radii, num_direction=num_direction, h=h, a=a,
                   radial_bin_mode=radial_bin_mode, h5_write_mode=h5_write_mode, mpi_io_mode=mpi_io_mode)

end_time = time.time()
run_time = end_time - start_time
print("Task Finished")
print(f"The tot time is {run_time/3600:.2f} hours")
#plot_halo_maps_halos(top_halos_100_noAGN, top_halos_100_AGN, save_path_100n, save_path_100f, 100)
