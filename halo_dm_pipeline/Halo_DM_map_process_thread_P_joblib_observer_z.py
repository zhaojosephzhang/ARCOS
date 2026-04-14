from cProfile import label
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#warnings.simplefilter('always', RuntimeWarning)
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
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from matplotlib import units
from numpy import histogram2d
import matplotlib.colors as colors
import bisect
import math
from math import atan
import time
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
import shutil
from typing import Optional
import hashlib
import ast
font_label = {
    'family': 'serif',
    'size': 20
}

tick_size = 18 

font_legend= {
    'family': 'serif',
    'size': 16
}
print("start")
sys.stdout.flush()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#print(H_mass_frac, He_mass_frac)
#f_IGM =  0.65
SPHmasstocgs =  1.989e+43
Mpctokpc = 1e3

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
default_snap_num = 20
default_feedback_on = 1
# 检查命令行参数
if len(sys.argv) == 1:
    # 无外部输入时使用默认值
    snap_num = default_snap_num 
    feedback_on = default_feedback_on
elif len(sys.argv) == 2:
    # 外部输入两个参数
    try:
        snap_num = int(ast.literal_eval(sys.argv[1]))
        feedback_on = default_feedback_on
        directory_label = 0
    except (ValueError, SyntaxError):
        print("Error: Please provide valid inputs for snap_num.")
elif len(sys.argv) == 3:
    # 外部输入两个参数
    try:
        snap_num = int(ast.literal_eval(sys.argv[1]))
        directory_label = 0
        feedback_on = int(sys.argv[2])
    except (ValueError, SyntaxError):
        print("Error: Please provide valid inputs for snap_num and feedback_on.")
else:
    print("Usage:")
    print("  python3 Halo_DM_map_process.py")
    print("  python3 Halo_DM_map_process.py <snap_num> <feedback_on>")
    print("Example: python3 Halo_local_data_withsellar.py 20 1 ")


if directory_label:
    directory_path = "/sqfs/work/hp240141/z6b340/results/gathered_results"
    storage_path_100f = "/sqfs2/cmc/1/work/hp240141/z6b340/results/CROCODILE_v1/DM_map_100f_observer/DM_map_100f_figure_joblib_observer/"
    save_path_100f = '/sqfs/work/hp240141/z6b340/results/CROCODILE_v1/DM_map_100f_observer/DM_map_100f_figure_joblib_observer'
    storage_path_100n = "/sqfs/work/hp240141/z6b340/results/CROCODILE_v1/DM_map_100n_observer/"
    save_path_100n = '/sqfs2/cmc/1/work/hp240141/z6b340/results/CROCODILE_v1/DM_map_100n_observer/DM_map_100n_figure_joblib_observer/'

else:
    directory_path = f"/sqfs/work/hp240141/z6b340/results/Halo_data_2D/snap_{snap_num}/"
    storage_path_100f = f"/sqfs2/cmc/1/work/hp240141/z6b340/results/Halo_data_2D/snap_{snap_num}/DM_map_100f_observer/"
    save_path_100f = f"/sqfs2/cmc/1/work/hp240141/z6b340/results/Halo_data_2D/snap_{snap_num}/DM_map_100f_observer/DM_map_100f_figure_joblib_observer/"
    storage_path_100n =f"/sqfs2/cmc/1/work/hp240141/z6b340/results/Halo_data_2D/snap_{snap_num}/DM_map_100f_observer/"
    save_path_100n =  f"/sqfs2/cmc/1/work/hp240141/z6b340/results/Halo_data_2D/snap_{snap_num}/DM_map_100n_observer/DM_map_100n_figure_joblib_observer/"
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
    sys.stdout.flush()
comm.Barrier()


def format_mass_range(mass_range_key):
    # 分割mass_range_key到两部分
    parts = mass_range_key.split('-')
    
    # 将每部分转换为科学计数法并取对数，然后构造LaTeX格式的字符串
    formatted_parts = [r'10^{' + f"{round(math.log10(float(part)),1)}" + '}' for part in parts]
    
    # 将转换后的部分再次组合
    return '-'.join(formatted_parts)

def log_mean_without_zeros(arr):
    arr = np.where(arr == 0, np.nan, arr)
    if np.all(np.isnan(arr)):
        return np.full(arr.shape[1:], np.nan)  # 或 return np.zeros(...) 看你想要什么默认值
    log_mean = np.nanmean(np.log(arr), axis=0)
    return np.exp(log_mean)


def linear_mean_without_zeros(arr):
    arr = np.where(arr == 0, np.nan, arr)
    if np.all(np.isnan(arr)):
        return np.full(arr.shape[1:], np.nan)
    return np.nanmean(arr, axis=0)


def median_without_zeros(arr):
    arr = np.where(arr == 0, np.nan, arr)
    if np.all(np.isnan(arr)):
        return np.full(arr.shape[1:], np.nan)
    return np.nanmedian(arr, axis=0)


def calculate_error_bands(data):
    """
    计算上下两条曲线作为误差范围。

    参数：
    data (numpy.ndarray): 数据数组。

    返回：
    (numpy.ndarray, numpy.ndarray): 返回上限曲线和下限曲线。
    """
    # 在对数空间中计算 percentiles
    data[data == 0] = np.nan
    log_data = np.log(data)

    # 计算上下限
    percentiles = [16, 84]  # 16th and 84th percentiles
    percentile_values = np.nanpercentile(log_data, percentiles, axis=0)

    # 上下限曲线
    lower_bound_curve = percentile_values[0]
    upper_bound_curve = percentile_values[1]

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
# 计算共动距离和尺度因子的函数
def comoving_distance(z):
    integrand = lambda zp: 1.0 / np.sqrt(Omega_m * (1.0+zp)**3 + Omega_Lambda)
    integral, _ = quad(integrand, 0, z)
    return c/H_0 * integral

def scale_factor(z):
    return 1.0 / (1.0 + z)

def log_mean_without_zeros(arr):
    # 将数组中的零替换为NaN
    arr[arr == 0] = np.nan
    # 计算对数平均值，忽略NaN值
    log_mean = np.nanmean(np.log(arr), axis=0)
    return np.exp(log_mean)

def linear_mean_without_zeros(arr):
    # 将数组中的零替换为NaN
    arr[arr == 0] = np.nan
    # 计算对数平均值，忽略NaN值
    linear_mean = np.nanmean(arr, axis=0)
    return linear_mean

def median_without_zeros(arr):
    # 将数组中的零替换为NaN
    arr[arr == 0] = np.nan
    # 计算对数平均值，忽略NaN值
    median = np.nanmedian(arr, axis=0)
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
    data[data == 0] = np.nan
    log_data = np.log(data)

    # 计算上下限
    percentiles = [0.15, 99.85] # 0.15th and 99.85th percentiles
    percentile_values = np.nanpercentile(log_data, percentiles, axis=0)

    # 上下限曲线
    lower_bound_curve = percentile_values[0]
    upper_bound_curve = percentile_values[1]

    return np.exp(lower_bound_curve), np.exp(upper_bound_curve)

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
    halo_center = np.array(halo_center)
    tree = cKDTree(pos, boxsize=Lbox)
    idx = tree.query_ball_point(halo_center, RV)
    dist = np.sqrt(np.sum((pos[idx] - halo_center)**2, axis=1))
    crossings = pos[idx] - halo_center
    crossings = np.mod(crossings + Lbox/2, Lbox) - Lbox/2
    crossing_dist = np.sqrt(np.sum((crossings)**2, axis=1))
    mask = crossing_dist < dist
    idx = np.array(idx)
    idx[mask] = idx[mask] + len(pos)
    dist[mask] = crossing_dist[mask]
    return idx, dist

def find_particles_within_RV_sph_tree(tree, halo_center, RV, Lbox):
    halo_center = np.mod(halo_center, Lbox)
    halo_center = np.array(halo_center)
    
    pos = tree.data  # 正确地从tree中取出粒子坐标数据
    
    idx = tree.query_ball_point(halo_center, RV)
    dist = np.sqrt(np.sum((pos[idx] - halo_center)**2, axis=1))

    crossings = pos[idx] - halo_center
    crossings = np.mod(crossings + Lbox / 2, Lbox) - Lbox / 2
    crossing_dist = np.sqrt(np.sum((crossings)**2, axis=1))

    mask = crossing_dist < dist
    idx = np.array(idx)
    idx[mask] = idx[mask] + len(pos)
    dist[mask] = crossing_dist[mask]

    return idx, dist


def calc_ne_LOS_shifted(k, i, LOS_shifted, sphpos, h_smooth_max, Lbox, SPHmasstocgs, f_e, mp, Smoothlen):
    ne_LC_shifted_k_i = 0
    idx_LC, dist_LC = find_particles_within_RV_sph(sphpos, LOS_shifted[k][i], h_smooth_max, Lbox)
    for j in range(len(idx_LC)):
        if idx_LC[j] >= len(sphpos):
            idx_LC[j] = idx_LC[j] - len(sphpos)
        if dist_LC[j] >= Smoothlen[idx_LC[j]]:
            continue
        else:
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
def load_halo_data(directory_path):
    #halo_100_noAGN = h5py.File(directory_path + "all_ne_DM_2RV_N500_B100_18_n_alongx.h5", 'r')
    #haloinfo_100_noAGN_max = h5py.File(directory_path + "all_DM_Haloinfo_2RV_N500_B100_20_n_alongx_max.h5", 'r')
    #haloinfo_100_noAGN = h5py.File(directory_path + "Haloinfo_2RV_N500_B100_020_n_alongx_withstellar_addition.h5", 'r')

    #halo_100_AGN = h5py.File(directory_path + "all_ne_DM_2RV_N500_B100_18_f_alongx.h5", 'r')
    #haloinfo_100_AGN_max = h5py.File(directory_path + "all_DM_Haloinfo_2RV_N500_B100_20_f_alongx_max.h5", 'r')
    haloinfo_100_AGN = h5py.File(directory_path + fr"Haloinfo_2RV_N500_B100_0{snap_num}_f_alongx_withstellar_addition.h5", 'r')

    haloinfo_100_noAGN = h5py.File(directory_path + fr"Haloinfo_2RV_N500_B100_0{snap_num}_n_alongx_withstellar_addition.h5", 'r')

    #halo_500_noAGN = h5py.File(directory_path + "all_ne_DM_2RV_N500_B500_alongx.h5", 'r')
    #haloinfo_500_noAGN = h5py.File(directory_path + "all_DM_Haloinfo_2RV_N500_B500_24_n_alongx.h5", 'r')
    
    #data_100_noAGN = h5py.File(directory_path + "data_100_018_noAGN.h5", 'r')
    #data_100_AGN = h5py.File(directory_path + "data_100_018_fiducial.h5", 'r')
    #data_500_noAGN = h5py.File(directory_path + "data_500_024.h5", 'r')
    #Halocenter_100_noAGN = data_100_noAGN['data']['halopos']
    #Halocenter_100_AGN = data_100_AGN['data']['halopos']
    #Halocenter_500_noAGN = data_500_noAGN['data']['halopos']
    
    return  haloinfo_100_AGN, haloinfo_100_noAGN



##############取前n个质量的halo
def get_top_n_halos_mass(haloinfo_data):
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
            n = len(halo_masses)
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
                    'SFR': mass_range_data[key]['SFR'][()],
                    "f_HI": mass_range_data[key]['f_HI'][()],
                    "f_H2I": mass_range_data[key]['f_H2I'][()],
                    "f_HII": mass_range_data[key]['f_HII'][()],
                    "f_Fe": mass_range_data[key]['f_Fe'][()],
                    "f_O_near": mass_range_data[key]['f_O_near'][()],
                    "Temperature": mass_range_data[key]['Temperature_near'][()],
                    'Z_sph': mass_range_data[key]['Z_sph'][()],
                    'Z_star': mass_range_data[key]['Z_stellar'][()],
                    "SFT_star": mass_range_data[key]['SFT_stellar'][()]
                }
    return top_halos


####################jit acceleration
# 定义加速部分的函数，用于处理线积分密度和累加
def accelerated_density_calculation(z_bins, idx_LC, dist_LC, smoothlen, sphmass, f_e, MpcTocm, dz, Omega_0, Omega_Lambda, H_0, z_center, c):
    ne_LC_shifted = np.zeros(len(z_bins))

    for k, z in enumerate(z_bins):
        for m in range(len(idx_LC)):
            if dist_LC[m] >= smoothlen[idx_LC[m]]:
                continue
            else:
                weight = M6(dist_LC[m] * MpcTocm, 0, smoothlen[idx_LC[m]] * MpcTocm, 3)
                ne_LC_shifted[k] += sphmass[idx_LC[m]] * f_e[idx_LC[m]] / 1.67e-24 * weight

    DM = np.zeros(len(z_bins) - 1)
    DM[0] = c / H_0 * ne_LC_shifted[0] * (1 + z_center[0]) * dz[0] / np.sqrt(Omega_0 * (1 + z_center[0]) ** 3 + Omega_Lambda) * 1e6
    for k in range(1, len(z_bins) - 1):
        DM[k] = DM[k - 1] + c / H_0 * ne_LC_shifted[k] * dz[k] / np.sqrt(Omega_0 * (1 + z_center[k]) ** 3 + Omega_Lambda) * 1e6

    return DM


###########Orignal column density calculation

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
    HaloRV
):
    """
    Calculate total SFR and average metallicity within 5 RV of the halo center.

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
    search_radius = 5.0 * HaloRV

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


#def calculate_column_density(halo, Lbox, grid_size=100, z_grid_size=10, process_id=0, process_n_jobs=5, axis=2):
def calculate_column_density(
    halo, 
    Lbox, 
    cell_size_kpc=20,  # 每个网格单元大小为 10 kpc
    grid_size=200,    # 网格数量（200 x 200）
    z_grid_size=200, 
    process_id=0, 
    process_n_jobs=5, 
    axis=2
):
    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.3099)  # 根据你设定的宇宙学参数
    # halo 属性
    HaloRV = halo['Halo_RV']
    halo_center = halo['halo_center']
    Halomass =  halo['Halomass']
    HaloMV =  halo['Halo_RV']
    sphpos = halo['sphpos']
    sphmass_cgs = halo['sphmass'] * SPHmasstocgs
    smoothlen = halo['smoothlen']
    f_e = halo['f_e']
    f_HI = halo['f_HI']
    f_H2I = halo['f_H2I']
    f_HII = halo['f_HII']
    f_Fe = halo['f_Fe']
    f_O = halo['f_O_near']
    Temperature = halo['Temperature']

    # Added properties: star formation rate and metallicity
    star_formation_rate = halo['SFR']
    metallicity_sph = halo['Z_sph']
    

    ####### stellar information
    stellarpos = np.array(halo['starpos'])
    stellarmass_cgs = np.array(halo['starmass'] * SPHmasstocgs)
    star_formation_time = halo['SFT_star']
    metallicity_stellar = halo['Z_star']
    # 网格单位大小转换为 Mpc
    cell_size_mpc = cell_size_kpc / 1e3  # 10 kpc = 0.01 Mpc
    extent_mpc = cell_size_mpc * grid_size / 2  # ±范围

    # 初始化列密度图
    column_density_map = np.zeros((grid_size, grid_size))
    h_smooth_max = np.max(smoothlen)
    
    min_bin_size = 3.38 / 1e3  # 转换为 Mpc
    frac = max(min_bin_size / HaloRV, 0.1 * (2 / HaloRV))  # 既满足解析需求，又考虑 halo 质量
    adaptive_bins = np.geomspace(frac * HaloRV, 2 * HaloRV, z_grid_size//2)
    adaptive_bins = np.concatenate((-adaptive_bins[::-1], adaptive_bins))  # 负区间对称
    # 根据投影方向设定三个方向的网格坐标
    if axis == 0:  # x 为 line-of-sight
        x_bins = halo_center[1] + np.linspace(-extent_mpc, extent_mpc, grid_size + 1)
        y_bins = halo_center[2] + np.linspace(-extent_mpc, extent_mpc, grid_size + 1)
        z_bins = halo_center[0] + adaptive_bins
        los_centers = (z_bins[:-1] + z_bins[1:]) / 2
        dL_list = z_bins[1:] - z_bins[:-1]

    elif axis == 1:  # y 为 line-of-sight
        x_bins = halo_center[0] + np.linspace(-extent_mpc, extent_mpc, grid_size + 1)
        y_bins = halo_center[2] + np.linspace(-extent_mpc, extent_mpc, grid_size + 1)
        z_bins = halo_center[1] + adaptive_bins
        los_centers = (z_bins[:-1] + z_bins[1:]) / 2
        dL_list = z_bins[1:] - z_bins[:-1]

    else:  # axis == 2, z 为 line-of-sight
        x_bins = halo_center[0] + np.linspace(-extent_mpc, extent_mpc, grid_size + 1)
        y_bins = halo_center[1] + np.linspace(-extent_mpc, extent_mpc, grid_size + 1)
        z_bins = halo_center[2] + adaptive_bins
        los_centers = (z_bins[:-1] + z_bins[1:]) / 2
        dL_list = z_bins[1:] - z_bins[:-1]

    # Define grid
    #bin_size = 2 * HaloRV / grid_size
    cell_area = cell_size_mpc ** 2 
    cell_area_cgs = cell_area * (MpcTocm**2)
    grid_length = cell_size_kpc*grid_size
    ne_map = np.zeros((grid_size, grid_size))
    n_HI_column =  np.zeros((grid_size, grid_size))
    n_H2I_column =  np.zeros((grid_size, grid_size))
    n_HII_column = np.zeros((grid_size, grid_size))
    n_Fe_column = np.zeros((grid_size, grid_size))
    n_O_column = np.zeros((grid_size, grid_size))
    star_mass_map = np.zeros((grid_size, grid_size))
    Z_sph_map = np.zeros((grid_size, grid_size))
    T_map = np.zeros((grid_size, grid_size))
    SFR_map = np.zeros((grid_size, grid_size))
    Z_star_map = np.zeros((grid_size, grid_size))
    SFT_star_map = np.zeros((grid_size, grid_size))
    star_count_map = np.zeros((grid_size, grid_size))
    weight_sum_map = np.zeros((grid_size, grid_size))
    

   # x_centers =  (x_bins[1:] + x_bins[:-1]) / 2
   # y_centers =  (y_bins[1:] + y_bins[:-1]) / 2
    #Bin_edges_shifted = np.linspace(0, 4 * HaloRV, z_grid_size)
    #z_list = np.array([find_redshift_and_scale_factor(dC)[0] for dC in Bin_edges_shifted])
    #z_centers = (z_bins[1:] + z_bins[:-1]) / 2
    #dz_list = z_bins[1:] - z_bins[:-1]
    c = 299792.458  # 光速，单位为 km/s
    H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
    Omega_0 = 0.31  # 物质密度参数
    Omega_Lambda = 0.69  # 暗能量密度参数

    # Precompute spatial tree for sphpos
    tree_sph = cKDTree(sphpos, boxsize=Lbox)
    tree_stellar = cKDTree(stellarpos, boxsize=Lbox)
    kernel_cutoff_factor = 2.5  # 平滑核的截断因子

    sph_query_radius = min(np.percentile(smoothlen, 95), kernel_cutoff_factor * np.median(smoothlen))
    def periodic_mask(coord, center, half_width, Lbox):
        delta = coord - center
        delta = delta - Lbox * np.round(delta / Lbox)  # Shift to [-Lbox/2, Lbox/2]
        return np.abs(delta) < half_width
    def calculate_cell_properties(i, j, x, y):
       ##########Gas Physical Values
        LOS_integrated_ne = 0.0
        LOS_integrated_n_HI = 0.0
        LOS_integrated_n_H2I = 0.0
        LOS_integrated_n_HII = 0.0
        LOS_integrated_n_Fe = 0.0
        LOS_integrated_n_O = 0.0
        Z_sph_sum = 0.0
        T_sum = 0.0
        SFR_sum = 0.0
        weight_sum = 0.0

       #######stellar physical values
        star_mass_sum = 0.0
        Z_star_sum = 0.0
        SFT_sum = 0.0
        star_mass_sum = 0.0
        star_count = 0


        for z, dz in zip(los_centers, dL_list):
            LOS_pos = [x, y, z]
            idx_LC, dist_LC = find_particles_within_RV_sph_tree(tree_sph, LOS_pos, sph_query_radius, Lbox)
            for m in range(len(idx_LC)):
                if idx_LC[m]  >= len(sphpos):
                    idx_LC[m] = idx_LC[m] - len(sphpos)
                if dist_LC[m] >= smoothlen[idx_LC[m]]:
                    continue
                else:
                    weight = M6(dist_LC[m] * MpcTocm, 0, smoothlen[idx_LC[m]] * MpcTocm, 3)
                    mass = sphmass_cgs[idx_LC[m]]
                    ne = mass * f_e[idx_LC[m]] / 1.67e-24 * weight
                    n_HI = mass * f_HI[idx_LC[m]] /1.67e-24 * weight
                    n_H2I = mass * f_H2I[idx_LC[m]] /1.67e-24 * weight
                    n_HII = mass * f_HII[idx_LC[m]] /1.67e-24 * weight
                    n_Fe = mass * f_Fe[idx_LC[m]] /1.67e-24 * weight
                    n_O = mass * f_O[idx_LC[m]] /1.67e-24 * weight
                    Z_sph_sum += metallicity_sph[idx_LC[m]] * mass * weight
                    LOS_integrated_ne += ne*dz*1e6
                    LOS_integrated_n_HI += n_HI*dz*MpcTocm
                    LOS_integrated_n_H2I += n_H2I*dz*MpcTocm
                    LOS_integrated_n_HII += n_HII*dz*MpcTocm
                    LOS_integrated_n_Fe += n_Fe*dz*MpcTocm
                    LOS_integrated_n_O += n_O*dz*MpcTocm
                    T_sum += Temperature[idx_LC[m]] * mass * weight
                    SFR_sum += star_formation_rate[idx_LC[m]] * mass * weight
                    weight_sum += mass * weight
                    
                # stellar projection
            star_query_radius = np.sqrt((cell_size_mpc/2)**2 + dz**2)
            
            idx_star_query = tree_stellar.query_ball_point(LOS_pos, star_query_radius)
            idx_star_query = np.asarray(idx_star_query, dtype=int)  # 确保是int数组
            stellar_pos_query = stellarpos[idx_star_query]
            #stellarmass_query = stellarmass_cgs[idx_star_query]
            #Z_stellar_query = metallicity_stellar[idx_star_query]
            #SFT_query = star_formation_time[idx_star_query]
            # 精确筛选落入柱体的粒子
            x_cond = periodic_mask(stellar_pos_query[:, 0], x, cell_size_mpc/2, Lbox)
            y_cond = periodic_mask(stellar_pos_query[:, 1], y, cell_size_mpc/2, Lbox)
            z_cond = periodic_mask(stellar_pos_query[:, 2], z, dz/2, Lbox)
            # 不提取子数组
            mask = x_cond & y_cond & z_cond  # 作用在 stellarpos[idx_star_query]
            idx_star_mask = idx_star_query[mask] # 这是全局索引

            # 然后直接用全局索引在原始数组中取值
            star_mass_sum += np.sum(stellarmass_cgs[idx_star_mask])
            Z_star_sum += np.sum(metallicity_stellar[idx_star_mask] * stellarmass_cgs[idx_star_mask])
            #SFT_sum += np.sum(star_formation_time[idx_star_mask]*stellarmass_cgs[idx_star_mask])
            # 转换形成时间（scale factor）为 Myr
            t_form = cosmo.age(1 / star_formation_time[idx_star_mask] - 1).value  # 单位：Gyr
            t_form_myr = t_form * 1e3
            # 再进行质量加权
            SFT_sum += np.sum(t_form_myr * stellarmass_cgs[idx_star_mask])
            star_count += len(idx_star_mask)
            star_mass_sum += np.sum(stellarmass_cgs[idx_star_mask])  # 质量累计




        return (
            i, j,
            LOS_integrated_ne,
            LOS_integrated_n_HI,
            LOS_integrated_n_H2I,
            LOS_integrated_n_HII,
            LOS_integrated_n_Fe,
            LOS_integrated_n_O,
            Z_sph_sum,
            T_sum,
            SFR_sum,
            Z_star_sum,
            SFT_sum,
            star_mass_sum,
            star_count,
            weight_sum,
        )


    total_iterations = grid_size * grid_size
    start_time = time.time()

    #with tqdm(total=total_iterations, desc=f"Process {process_id}") as pbar:
    #    results = Parallel(n_jobs=process_n_jobs, timeout=18000)(delayed(calculate_los_density)(i, j, x, y) for i, x in enumerate(x_bins) for j, y in enumerate(y_bins))
    #    pbar.update()

    if rank == 0 or rank == 29:
        with tqdm(total=total_iterations, desc=f"Rank {rank}-Process {process_id}") as pbar:
            results = []
            for i in range(0, total_iterations, process_n_jobs):
                batch_results = Parallel(n_jobs=process_n_jobs, timeout=360000)(
                    delayed(calculate_cell_properties)(i // grid_size, i % grid_size, x_bins[i // grid_size], y_bins[i % grid_size])
                    for i in range(i, min(i + process_n_jobs, total_iterations))
                )
                results.extend(batch_results)
                pbar.update(len(batch_results))
    else:
        with tqdm(total=total_iterations, desc=f"Rank {rank}-Process {process_id}", disable=True) as pbar:
            results = []
            for i in range(0, total_iterations, process_n_jobs):
                batch_results = Parallel(n_jobs=process_n_jobs, timeout=360000)(
                    delayed(calculate_cell_properties)(i // grid_size, i % grid_size, x_bins[i // grid_size], y_bins[i % grid_size])
                    for i in range(i, min(i + process_n_jobs, total_iterations))
                )
                results.extend(batch_results)
                pbar.update(len(batch_results))

    for i, j, ne, n_HI, n_H2I, nHII, n_Fe, n_O, Z_sph_sum, T_sum, SFR_sum, Z_star_sum, SFT_sum, star_mass_sum, star_count, weight_sum in results:
        ne_map[i, j]           = ne
        n_HI_column[i, j]      = n_HI
        n_H2I_column[i, j]      = n_H2I
        n_HII_column[i, j]     = nHII
        n_Fe_column[i, j]      = n_Fe
        n_O_column[i, j]       = n_O
        weight_sum_map[i, j]   = weight_sum
        # SPH 质量加权平均
        if weight_sum > 0:
            SFR_map[i, j] = SFR_sum / weight_sum
            T_map[i, j] = T_sum / weight_sum
            Z_sph_map[i, j] = Z_sph_sum / weight_sum
        else:
            SFR_map[i, j] = T_map[i, j] = Z_sph_map[i, j] = 0

        # 恒星相关
        if star_mass_sum > 0:
            star_mass_map[i, j] = star_mass_sum
            Z_star_map[i, j] = Z_star_sum / star_mass_sum
            SFT_star_map[i, j] = SFT_sum / star_mass_sum
            star_count_map[i, j] = star_count
        else:
            star_mass_map[i, j] = Z_star_map[i, j] =  SFT_star_map[i, j] = 0
            star_count_map[i, j] = 0

    total_time = time.time() - start_time
    print(f"rank {rank}-Process {process_id} - Total execution time: {total_time:.2f} seconds")

    return (
        x_bins, y_bins,      # 网格边界
        ne_map,              # 总电子数密度投影
        n_HI_column,         # 中性氢柱密度
        n_H2I_column,        # 中性氢气柱密度
        n_HII_column,        # 电离氢柱密度
        n_Fe_column,         # 铁柱密度
        n_O_column,          # 氧柱密度
        SFR_map,             # SPH 投影 SFR（质量加权）
        Z_sph_map,           # SPH 投影金属丰度
        T_map,               # SPH 投影温度
        star_mass_map,       # 每格内恒星总质量
        Z_star_map,          # 恒星金属丰度（质量加权）
        SFT_star_map,        # 恒星形成时间（质量加权）
        star_count_map,      # 恒星数量
        weight_sum_map,      # SPH加权因子（质量加权）
    )

def calculate_1D_from_column_density(x_bins, y_bins, data_map, halo_center, grid_size):
    b_values = []
    data_values = []

    for i, x in enumerate(x_bins[:-1]):
        for j, y in enumerate(y_bins[:-1]):
            b = np.sqrt((x - halo_center[0]) ** 2 + (y - halo_center[1]) ** 2)
            b_values.append(b)
            data_values.append(data_map[i, j])

    b_values = np.array(b_values)
    data_values = np.array(data_values)

    x_range = x_bins[-1] - x_bins[0]
    y_range = y_bins[-1] - y_bins[0]
    b_max = np.sqrt((x_range / 2)**2 + (y_range / 2)**2)

    bin_edges = np.linspace(0, b_max, grid_size // 2)
    binned_data = [[] for _ in range(len(bin_edges) - 1)]

    for b, val in zip(b_values, data_values):
        bin_index = np.digitize(b, bin_edges) - 1
        if 0 <= bin_index < len(binned_data):
            binned_data[bin_index].append(val)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 * 1e3  # kpc
    mean_data = [log_mean_without_zeros(np.array(vals)) if len(vals) > 0 else 0 for vals in binned_data]
    std_data = [np.std(np.array(vals)) if len(vals) > 0 else 0 for vals in binned_data]
    median_data = [median_without_zeros(np.array(vals)) if len(vals) > 0 else 0 for vals in binned_data]
    lower_bound = []
    upper_bound = []

    for vals in binned_data:
        if len(vals) > 0:
            lb, ub = calculate_error_bands(np.array(vals))
            lower_bound.append(lb)
            upper_bound.append(ub)
        else:
            lower_bound.append(0)
            upper_bound.append(0)

    return bin_centers, mean_data, median_data, lower_bound, upper_bound

def save_to_hdf5(file_path, x_bins, y_bins, maps, profiles, halo):
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('x_bins', data=x_bins)
            f.create_dataset('y_bins', data=y_bins)

            for key, val in maps.items():
                dset = f.create_dataset(f'{key}_map', data=val)
                if key == 'DM' or key == 'ne':
                    dset.attrs['unit'] = 'pc cm^-3'
                elif key.startswith('n_'):
                    dset.attrs['unit'] = 'cm^-2'
                elif key == 'SFR':
                    dset.attrs['unit'] = 'M_sun yr^-1 cm^-2'
                elif key == 'Z_sph' or key == 'Z_star':
                    dset.attrs['unit'] = 'Z_sun'
                elif key == 'T':
                    dset.attrs['unit'] = 'K'
                elif key == 'stellar_mass':
                    dset.attrs['unit'] = 'M_sun'
                elif key == 'SFT_star':
                    dset.attrs['unit'] = 'Gyr'
                elif key == 'star_count':
                    dset.attrs['unit'] = 'count'

            for prof_key, prof_vals in profiles.items():
                grp = f.create_group(f'{prof_key}')
                for k, v in prof_vals.items():
                    grp.create_dataset(k, data=np.array(v))

        
            f.attrs['halomass'] = halo['Halomass']
            f.attrs['HaloMV'] = halo['Halo_MV']
            f.attrs['HaloRV'] = halo['Halo_RV']
    except OSError as e:
        print(f"[OSError] Failed to write file {file_path}: {e}")



# ---- 路径与磁盘空间助手（护栏） ----
SMALL_PREFIX = '/sqfs/work/'
BIG_PREFIX   = '/sqfs2/cmc/1/work/'
def _to_bigspace(path: str) -> str:
    if path is None:
        return path
    # 纯替换，幂等，无递归
    return str(path).replace(SMALL_PREFIX, BIG_PREFIX)

def _ensure_bigspace(path: str) -> str:
    p = _to_bigspace(path)
    if not p.startswith(BIG_PREFIX):
        raise RuntimeError(f"FATAL: output path not in BIG space: {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
def _ensure_dir_for(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _has_space(dirpath: str, min_free_gb: float = 2.0) -> bool:
    total, used, free = shutil.disk_usage(dirpath)
    return (free / 1024**3) > min_free_gb

# ---- 新增：分桶 + rank 子目录 ----
def shard_dir(
    base_dir: str,
    halo_index: int,
    rank: Optional[int] = None,
    fanout: int = 16,
    method: str = "hash"  # "mod" 或 "hash"
):
    """
    将 halo 输出分散到少量桶目录（不依赖 halo_index 顺序）。
    - method="mod":  用 halo_index % fanout 均匀分桶（要求 halo_index 数值分布还算均匀）
    - method="hash": 用 md5(halo_index) 的稳定哈希分桶（强烈推荐，跨进程/重启稳定）
    - fanout: 桶的数量（8~32 通常够用；几百个 halo 用 16 很稳）
    目录结构示例： base/s03/rank_0007/
    """
    hid = int(halo_index)

    if method == "mod":
        shard_id = hid % fanout
    else:
        # 稳定哈希：不同进程/重启结果一致；不依赖 PYTHONHASHSEED
        h = hashlib.md5(str(hid).encode("utf-8")).hexdigest()
        shard_id = int(h[:8], 16) % fanout

    bucket_dir = f"s{shard_id:02x}"  # s00, s01, ... s0f
    parts = [base_dir, bucket_dir]
    if rank is not None:
        parts.append(f"rank_{int(rank):04d}")
    out = os.path.join(*parts)
    os.makedirs(out, exist_ok=True)
    return out

def _fsync_fileobj(f):
    try:
        f.flush()
        os.fsync(f.fid.get_vfd_handle())  # h5py >=3.10: 拿到底层 fd；不同版本可退回到 f.id.get_vfd_handle()
    except Exception:
        try:
            f.flush()
        except: 
            pass

def save_to_hdf5(file_path, x_bins, y_bins, maps, profiles, halo):
    file_path = _ensure_bigspace(file_path)
    tmp_path  = _ensure_bigspace(file_path + ".part")
    try:
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('x_bins', data=x_bins)
            f.create_dataset('y_bins', data=y_bins)

            for key, val in maps.items():
                # 可选：压缩+float32能省不少空间
                arr = np.asarray(val, dtype=np.float32)
                dset = f.create_dataset(f'{key}_map', data=arr, compression="gzip", compression_opts=4, shuffle=True, chunks=True)
                if key in ('DM', 'ne'):
                    dset.attrs['unit'] = 'pc cm^-2'
                elif key.startswith('n_'):
                    dset.attrs['unit'] = 'cm^-2'
                elif key == 'SFR':
                    dset.attrs['unit'] = 'M_sun yr^-1 cm^-2'
                elif key in ('Z_sph', 'Z_star'):
                    dset.attrs['unit'] = 'Z_sun'
                elif key == 'T':
                    dset.attrs['unit'] = 'K'
                elif key == 'stellar_mass':
                    dset.attrs['unit'] = 'M_sun'
                elif key == 'SFT_star':
                    dset.attrs['unit'] = 'Gyr'
                elif key == 'star_count':
                    dset.attrs['unit'] = 'count'

            for prof_key, prof_vals in profiles.items():
                grp = f.create_group(f'{prof_key}')
                for k, v in prof_vals.items():
                    grp.create_dataset(k, data=np.array(v))

            #f.attrs['halomass'] = halo['Halomass']
            #f.attrs['HaloMV'] = halo['Halo_MV']
            #f.attrs['HaloRV'] = halo['Halo_RV']
            for k, v in (('halomass','Halomass'), ('HaloMV','Halo_MV'), ('HaloRV','Halo_RV')):
                f.attrs[k] = halo[v]
            _fsync_fileobj(f)

        # 原子落盘：成功才 rename
        os.replace(tmp_path, file_path)

    except OSError as e:
        print(f"[OSError] Failed to write {file_path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as _:
            pass


##############MPI paralell



def process_halo(halo_key, halo, save_path, Lbox=100, AGN_info="n", rank=0, process_n_jobs=5, process_id=0):
    print("==== process_halo 参数 ====")
    print(f"halo_key: {halo_key}")
    print(f"save_path: {save_path}")
    print(f"Lbox: {Lbox}")
    print(f"AGN_info: {AGN_info}")
    print(f"rank: {rank}")
    print(f"process_n_jobs: {process_n_jobs}")
    print(f"process_id: {process_id}")
    h = 0.67742
    halomass = halo['Halomass'] * 1e10
    log_halomass = np.log10(halomass)
    HaloRV = halo['Halo_RV'] * 1e3  # Convert HaloRV to kpc
    (x_bins, y_bins, ne_map,
     n_HI_column, n_H2I_column, n_HII_column, n_Fe_column, n_O_column,
     SFR_map, Z_sph_map, T_map,
     star_mass_map, Z_star_map, SFT_star_map, star_count_map, weight_sum_map) = \
        calculate_column_density(halo, Lbox/h, cell_size_kpc=20, grid_size=200, z_grid_size=200, process_id=process_id, process_n_jobs=process_n_jobs, axis = 2)

    maps = {
        'DM': ne_map,
        'n_HI': n_HI_column,
        'n_H2I': n_H2I_column,
        'n_HII': n_HII_column,
        'n_Fe': n_Fe_column,
        'n_O': n_O_column,
        'SFR': SFR_map,
        'Z_sph': Z_sph_map,
        'T': T_map,
        'stellar_mass': star_mass_map,
        'Z_star': Z_star_map,
        'SFT_star': SFT_star_map,
        'star_count': star_count_map,
        'weight': weight_sum_map
    }

    units = {
        'DM': 'pc cm$^{-2}$',
        'n_HI': 'cm$^{-2}$',
        'n_H2I': 'cm$^{-2}$',
        'n_HII': 'cm$^{-2}$',
        'n_Fe': 'cm$^{-2}$',
        'n_O': 'cm$^{-2}$',
        'SFR': 'M$_\odot$ yr$^{-1}$ cm$^{-2}$',
        'Z_sph': 'Z$_\odot$',
        'T': 'K',
        'stellar_mass': 'g',
        'Z_star': 'Z$_\odot$',
        'SFT_star': 'Myr',
        'star_count': 'count',
        'weight': 'cm$^{-3}'
    }

    labels = {
        'DM': '$\mathrm{DM}$',
        'n_HI': '$n_{\mathrm{HI}}$',
        'n_H2I': '$n_{\mathrm{H_2}}$',
        'n_HII': '$n_{\mathrm{HII}}$',
        'n_Fe': '$n_{\mathrm{Fe}}$',
        'n_O': '$n_{\mathrm{O}}$',
        'SFR': '$\mathrm{SFR}$',
        'Z_sph': '$\mathrm{Z}_{\mathrm{sph}}$',
        'T': '$T$',
        'stellar_mass': '$\mathrm{Stellar\ Mass}$',
        'Z_star': '$\mathrm{Z}_{\mathrm{star}}$',
        'SFT_star': '$\mathrm{Star\ Formation\ Time}$',
        'star_count': '$\mathrm{Star}$',
        'weight': '$\mathrm{Weight}$'
    }




    halo_index = int(halo_key.replace('halo_', ''))
    # Plot and save column density map
    extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]

    profiles = {}

    fig_path = os.path.join(save_path, "figures")
    #fig_path = shard_dir(os.path.join(save_path, "figures"), halo_index, rank)
    #fig_path = shard_dir(os.path.join(save_path, "figures"), halo_index, rank, fanout=16, method="hash")

    for key, data_map in maps.items():
        bin_centers, mean_data, median_data, lower_bound, upper_bound = calculate_1D_from_column_density(
            x_bins, y_bins, data_map, halo['halo_center'], 100)

        profiles[key] = {
            'halo_center': halo['halo_center'],
            'bin_centers': bin_centers,
            'mean': mean_data,
            'median': median_data,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    file_path = f'{save_path}/DM_map_Halo{halo_index}_{Lbox}_{AGN_info}.h5'
    #h5_dir   = shard_dir(os.path.join(save_path, "h5"), halo_index, rank, fanout=16, method="hash")
    #file_path = os.path.join(h5_dir, f"DM_map_Halo{halo_index}_{Lbox}_{AGN_info}.h5")

    save_to_hdf5(file_path, x_bins, y_bins, maps, profiles, halo)
    if rank == 0 or rank == 29:
        # ---- 你的作图循环（已修正） ----
        for key, data_map in maps.items():
            fig = None
            try:
                # 计算 log10，并做有效性筛选
                log_map = np.log10(np.asarray(data_map, dtype=np.float64) + 1e-10)
                valid = np.isfinite(log_map) & (log_map > -10)
                if not np.any(valid):
                    print(f"[Skip] {key} map: all-NaN or invalid values.")
                    continue
                vmin = np.min(log_map[valid])
                vmax = np.max(log_map[valid])

                # 路径护栏 + 磁盘检查
                out_path = _to_bigspace(f'{fig_path}/{halo_index}_{Lbox}_{AGN_info}_{key}_map.png')
                _ensure_dir_for(out_path)
                if not _has_space(os.path.dirname(out_path)):
                    print(f"[Skip] No disk space to save {out_path}")
                    continue

                # 绘图
                fig, ax = plt.subplots()
                im = ax.imshow(log_map, cmap='viridis', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(r'$\log_{10}$ ' + labels[key])
                cbar.ax.tick_params(labelsize=13)

                ax.set_title(rf'Halo {halo_index} (Mass: $10^{{{log_halomass:.1f}}}$ $\mathrm{{M}}_{{\odot}}$, {Lbox} Mpc, {AGN_info})')
                ax.set_xlabel('X [Mpc]')
                ax.set_ylabel('Y [Mpc]')
                ax.tick_params(labelsize=11)

                fig.tight_layout()
                fig.savefig(out_path)
            except Exception as e:
                print(f"[Error] Plotting {key} map failed: {e}")
            finally:
                try:
                    if fig is not None:
                        plt.close(fig)
                except:
                    pass

        for key, prof in profiles.items():
            fig = None
            try:
                # 取出数据并检查有效性
                x = np.asarray(prof['bin_centers'])
                y_mean   = np.asarray(prof['mean'])
                y_median = np.asarray(prof['median'])
                y_lo     = np.asarray(prof['lower_bound'])
                y_hi     = np.asarray(prof['upper_bound'])

                # 若全为 NaN 或长度不匹配则跳过
                if x.size == 0 or not np.any(np.isfinite(y_mean)):
                    print(f"[Skip] {key} profile: empty or all-NaN.")
                    continue

                # 路径护栏 + 磁盘检查
                out_path = _to_bigspace(f'{fig_path}/{halo_index}_{Lbox}_{AGN_info}_{key}_b_relation.png')
                _ensure_dir_for(out_path)
                if not _has_space(os.path.dirname(out_path)):
                    print(f"[Skip] No disk space to save {out_path}")
                    continue

                fig, ax = plt.subplots()
                ax.plot(x, y_mean, '-', label='Log Mean Curve')
                ax.plot(x, y_median, '--', label='Median Curve')

                # 只有在上下界有效时才填充
                if np.any(np.isfinite(y_lo)) and np.any(np.isfinite(y_hi)):
                    ax.fill_between(x, y_lo, y_hi, color='gray', alpha=0.3, label='3-$\\sigma$ Error bands')

                # HaloRV 竖线
                ax.axvline(x=HaloRV, color='r', linestyle='--', label=f'HaloRV: {HaloRV:.1f} kpc')

                ax.set_xlabel('b [kpc]')
                ax.set_ylabel(f'{labels[key]} [{units[key]}]')
                ax.tick_params(labelsize=11)
                ax.legend(loc='best', fontsize='small')

                # 仅当 y 里有正值时使用对数坐标
                if np.any(np.isfinite(y_mean) & (y_mean > 0)):
                    ax.set_yscale("log")

                fig.tight_layout()
                fig.savefig(out_path)
            except Exception as e:
                print(f"[Error] Plotting {key} profile failed: {e}")
            finally:
                try:
                    if fig is not None:
                        plt.close(fig)
                except:
                    pass
    #comm.Barrier()
        


############## 基于Halo的MPI并行 ###################
def plot_halo_maps(halo_data, save_path, Lbox=500, AGN_info="n", target_halo_keys = []):
    num_cores = multiprocessing.cpu_count()
    num_cores = 30
    process_id = 0
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"MPI rank size: {size}")
    mass_ranges = [
        (10**10, 10**10.5),
        (10**10.5, 10**11),
        (10**11, 10**11.5),
        (10**11.5, 10**12),
        (10**12, 10**12.5),   
        (10**12.5, 10**13),   
        (10**13, 10**13.5),   
        (10**13.5, 10**14),
        (10**14, 10**15.5)   
    ]

    mass_bins_to_process = [f"{min_mass:.2e}-{max_mass:.2e}" for min_mass, max_mass in mass_ranges]
    print(f"mass_bin: {mass_bins_to_process}")
    print(f"halo_data_key: {halo_data.keys()}")
    sys.stdout.flush()
    for mass_bin in mass_bins_to_process:
        dir_path = os.path.join(save_path, mass_bin)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")
        fig_path = os.path.join(dir_path, "figures")
        os.makedirs(fig_path, exist_ok=True)
        print(f"Created: {fig_path}")
        sys.stdout.flush()
    # 将所有halo数据收集到一个列表中
    all_halos = []
    if target_halo_keys == []:
        for mass_bin, halos in halo_data.items():
            if mass_bin in mass_bins_to_process:
                save_path_temp = os.path.join(save_path, mass_bin)
                for halo_key in halos.keys():
                    all_halos.append((mass_bin, halo_key, AGN_info, save_path_temp))
    else:
        for mass_bin, halos in halo_data.items():
            if mass_bin in mass_bins_to_process:
                save_path_temp = os.path.join(save_path, mass_bin)
                for halo_key in halos.keys():
                    if halo_key in target_halo_keys:
                        all_halos.append((mass_bin, halo_key, AGN_info, save_path_temp))
    
    print(f"tot_halo_num:{len(all_halos)}")
    print(f"all_halos:{all_halos[1]}")
    sys.stdout.flush
    #with h5py.File(f'{save_path}/halos_data_specific.h5', 'w', driver='mpio', comm=comm) as hf:  #,
    for i, (mass_bin, halo_key, AGN_info, save_path_temp) in enumerate(all_halos):
        if i % size == rank:
        #if i % size == rank:
            if AGN_info == "n":
                if mass_bin in halo_data and halo_key in halo_data[mass_bin]:
                    halo = halo_data[mass_bin][halo_key]
                    #process_halo(halo_key, halo, save_path, Lbox, "n", rank, num_cores // size, process_id=i)
                    process_halo(halo_key, halo, save_path_temp, Lbox, "n", rank, num_cores, process_id=i)
                    
                    #except Exception as e:
                    #    print(f"Error processing {halo_key} in mass bin {mass_bin} (noAGN): {e}")
            else:
                if mass_bin in halo_data and halo_key in halo_data[mass_bin]:
                    halo = halo_data[mass_bin][halo_key]   
                    process_halo(halo_key, halo, save_path_temp, Lbox, "f", rank, num_cores, process_id=i)
                    #except Exception as e:
                        #print(f"Error processing {halo_key} in mass bin {mass_bin} (AGN): {e} {i}")
    print(f"Rank {rank} has finished its assigned tasks.")
##############Halo数据的MPI并行###################
#haloinfo_100_noAGN, haloinfo_100_AGN = load_halo_data(directory_path)
haloinfo_100_AGN, haloinfo_100_noAGN = load_halo_data(directory_path)
#top_halos_500_noAGN = get_top_n_halos(haloinfo_500_noAGN)
#num_cores = multiprocessing.cpu_count() 
#print(f"num_cores: {num_cores}")
#sys.stdout.flush()

#plot_halo_maps(top_halos_500_noAGN, save_path, 500, "n")


#top_halos_100_AGN = get_top_n_halos_index(haloinfo_100_AGN)
#num_cores = multiprocessing.cpu_count() 
#print(f"num_cores: {num_cores}")
#sys.stdout.flush()

#plot_halo_maps_rest(top_halos_100_AGN, save_path_100f, 100, "f")

print(f"start get top massive halos")
top_halos_100_AGN = get_top_n_halos_mass(haloinfo_100_AGN)
top_halos_100_noAGN = get_top_n_halos_mass(haloinfo_100_noAGN)
print(f"Halos extraction finished")
sys.stdout.flush()
#target_halo_keys_noAGN = ['halo_0', 'halo_1', 'halo_886']
#target_halo_keys_noAGN = ['halo_886']
#target_halo_keys_max_noAGN = ['halo_222']
#target_halo_keys_AGN = ['halo_0', 'halo_1', 'halo_1000']
#target_halo_keys_max_AGN = ['halo_223']
#specific_halos_100_noAGN = get_specific_halos_combined(haloinfo_100_noAGN, haloinfo_100_noAGN_max, target_halo_keys_noAGN , target_halo_keys_max_noAGN)
#specific_halos_100_AGN = get_specific_halos_combined(haloinfo_100_noAGN, haloinfo_100_noAGN_max, target_halo_keys_AGN, target_halo_keys_max_AGN)

#noAGN_keys = list(set(target_halo_keys_noAGN + target_halo_keys_max_noAGN))

#AGN_keys = list(set(target_halo_keys_noAGN + target_halo_keys_max_noAGN))

#top_halos_100_AGN, top_halos_100_noAGN = get_top_n_halos_index_common(haloinfo_100_AGN, haloinfo_100_noAGN)
#top_halos_100_AGN, top_halos_100_noAGN = get_top_n_halos_nearest_common(haloinfo_100_AGN, haloinfo_100_noAGN)
# 打印 CPU 核心数
num_cores = multiprocessing.cpu_count() 
#num_cores = 70
print(f"num_cores: {num_cores}")
sys.stdout.flush()
# 绘制 Halo 图像
h = 0.67742
#plot_halo_maps(top_halos_100_AGN, save_path_100f, 100, "f")

# 示例使用
#plot_halo_maps(top_halos_100_AGN, save_path_100f, 100, "f")
print(len(sys.argv))
if len(sys.argv) == 1 or len(sys.argv) == 2 :
    plot_halo_maps(top_halos_100_AGN, save_path_100f, 100, "f")
elif  len(sys.argv) == 3:
    if not feedback_on:
        plot_halo_maps(top_halos_100_noAGN, save_path_100n, 100, "n")
        print("NoBH Case")
    elif feedback_on:
        plot_halo_maps(top_halos_100_AGN, save_path_100f, 100, "f")
        print("Fiducial Case")
    else:
         print("Input Invalid. example: python3 xxx.py 20 0 (or 1)")
sys.stdout.flush()
#plot_halo_maps(specific_halos_100_AGN, save_path_100n, 100, "n", AGN_keys)

#plot_halo_maps_halos(top_halos_100_noAGN, top_halos_100_AGN, save_path_100n, save_path_100f, 100)
