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
import time
from scipy.spatial import cKDTree
import numpy as np
import random
import ast
import numbers
##################################
me = 9.1*10**-28
mH = 1.67*10**-24
solartog = 2.0*10**33
MpcTocm = 3.08567758*10**24
MsunTog = 1.988*10**33
h = 0.67742
Mu_e = 1.167
Mu_H = 1.3

font_label = {
    'family': 'serif',
    'size': 30
}

tick_size = 25

font_legend= {
    'family': 'serif',
    'size': 25
}
font_legend_small = {
    'family': 'serif',
    'size': 20
}

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

def create_grid_indices(data, grid_size, boxsize, a, h):
    num_grids = int(np.prod(grid_size))
    grid_indices = {particle_type: {} for particle_type in ["sphpos", "dmpos", "starpos"]}
    #grid_indices = {particle_type: {} for particle_type in ["sphpos", "starpos"]}
    for particle_type in ["sphpos", "dmpos", "starpos"]:
    #for particle_type in ["sphpos", "starpos"]:
        positions = data[particle_type]*h/a
        if np.max(positions) >  boxsize:
            max_pos = np.max(positions)
            print(f"max(positions): {max_pos}")
            sys.stdout.flush()
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




#def create_grid_indices(data, grid_size, box_size):
#    grid_size = [int(gs) for gs in grid_size]  # Ensure grid_size contains integers
#    grid_indices = {particle_type: [[] for _ in range(np.prod(grid_size))] for particle_type in ["sphpos", "dmpos", "starpos"]}
    
#    for particle_type in ["sphpos", "dmpos", "starpos"]:
#        positions = np.array(data[particle_type]) * 0.6774
#        for idx, pos in enumerate(positions):
#            if idx % 10000 == 0:
#                print(f"Processing particle type: {particle_type}, Index: {idx}")
#                sys.stdout.flush()
#            grid_x, grid_y, grid_z = calculate_grid_coordinates(pos.reshape(1, -1), grid_size, box_size)
#            grid_index = calculate_grid_index(grid_x[0], grid_y[0], grid_z[0], grid_size)
#            grid_indices[particle_type][grid_index].append(idx)
    
#    return grid_indices

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

def find_particles_within_RV_sph(pos, halo_center, RV, Lbox):
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
    idx[mask] = idx[mask] + len(pos)
    dist[mask] = crossing_dist[mask]

    return idx, dist

def calculate_error_bands(data):
    """
    计算上下两条曲线作为误差范围。

    参数：
    data (numpy.ndarray): 数据数组。

    返回：
    (numpy.ndarray, numpy.ndarray): 返回上限曲线和下限曲线。
    """
    # 在对数空间中计算 percentiles
    log_data = np.log(data)

    # 计算上下限
    percentiles = [16, 84]  # 16th and 84th percentiles
    percentile_values = np.percentile(log_data, percentiles, axis=0)

    # 上下限曲线
    lower_bound_curve = percentile_values[0]
    upper_bound_curve = percentile_values[1]

    return np.exp(lower_bound_curve), np.exp(upper_bound_curve)

def format_mass_range(mass_range_key):
    # 分割mass_range_key到两部分
    parts = mass_range_key.split('-')
    
    # 将每部分转换为科学计数法并取对数，然后构造LaTeX格式的字符串
    formatted_parts = [r'10^{' + f"{round(math.log10(float(part)),1)}" + '}' for part in parts]
    
    # 将转换后的部分再次组合
    return '-'.join(formatted_parts)


# 更新后的部分代码
def calculate_bins(binning_type, halo_rv=None, b=0, C_R_max=8.5):
    halo_rv = np.sqrt(halo_rv**2 + b**2)
    if binning_type == "log_linear":
        Xbins_log = np.logspace(-2, 1, num=4)
        interval_sizes = np.diff(Xbins_log)
        Xbins_log_extended = []
        for kk in range(len(Xbins_log) - 1):
            start_value = Xbins_log[kk]
            end_value = Xbins_log[kk + 1]
            interval_size = interval_sizes[kk]
            sub_bins = np.linspace(start_value, start_value + interval_size, num=10)
            Xbins_log_extended.extend(sub_bins[:-1])
        Xbins = np.log10(np.array(Xbins_log_extended[0:20]))
        Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
    elif binning_type == "linear_rv":
        # 确保分 bin 细致到 r/rv 为 0.01，并根据 halo_rv 的比例分 bin
        num_bins = int(np.around(2 * halo_rv / (0.01 * halo_rv), decimals=0))  # 使用 np.around
        Xbins = np.linspace(0.01 * halo_rv, 2 * halo_rv, num= num_bins)  # 从0.01*halo_rv 到 2*halo_rv 分布
        Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
    if binning_type == "linear_rv_ineq":
        # 划分bin的数量：低半径更密集，高半径逐渐减小bin数量
        low_r_bins = np.linspace(0.01 * halo_rv, 0.2 * halo_rv, num=20)  # 低半径处有较多的点
        high_r_bins = np.linspace(0.2 * halo_rv, 2 * halo_rv, num=15)  # 高半径处减少 bin 数量
        Xbins = np.concatenate([low_r_bins, high_r_bins[1:]])  # 将两部分合并
        # 计算 bin 中心点
        Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
    if binning_type == "log_linear_rv":
       # Adjusting the range from 0.01 * halo_rv to 2 * halo_rv
        Xbins_log = np.logspace(-2.2, np.log10(2), num=4) * halo_rv  # Major nodes from 0.01*halo_rv to 2*halo_rv in log scale
        interval_sizes = np.diff(Xbins_log)
        Xbins_log_extended = []
        for kk in range(len(Xbins_log) - 1):
            start_value = Xbins_log[kk]
            end_value = Xbins_log[kk + 1]
            interval_size = interval_sizes[kk]
            sub_bins = np.linspace(start_value, start_value + interval_size, num=10)
            Xbins_log_extended.extend(sub_bins[:-1])
        Xbins = np.array(Xbins_log_extended[0:40])  # 40 points, logarithmically spaced
        Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
    elif binning_type == "log_linear_rv_more":
        # Adjust the range from 0.01 * halo_rv to 2 * halo_rv
        # 低半径区域使用更高密度的log间隔
        Xbins_log_low = np.logspace(-2, -0.7, num=20) * halo_rv  # 从0.01*halo_rv到约0.2*halo_rv
        # 中等半径到高半径使用log到linear的过渡
        Xbins_log_high = np.logspace(-0.7, np.log10(2), num=20) * halo_rv  # 从0.2*halo_rv到5*halo_rv
        # 合并log和linear过渡部分
        Xbins = np.concatenate([Xbins_log_low, Xbins_log_high[1:]])  # 去掉重复点

        # 计算 bin 中心点
        Xpoints = (Xbins + np.roll(Xbins, -1))[:-1] / 2
        
    elif binning_type == "log_linear_rv_more_outer":
        # Adjust the range from 0.01 * halo_rv to 2 * halo_rv
        # 低半径区域使用更高密度的log间隔
        #Xbins_log_low = np.logspace(-1, -0.7, num=100) * halo_rv  # 从0.01*halo_rv到约0.2*halo_rv
        # 中等半径到高半径使用log到linear的过渡
        #Xbins_log_high = np.logspace(-0.7, np.log10(7), num=100) * halo_rv  # 从0.2*halo_rv到5*halo_rv
        # 合并log和linear过渡部分
        #Xbins = np.concatenate([Xbins_log_low, Xbins_log_high[1:]])  # 去掉重复点
        rmin = 0.1 * halo_rv
        rmax_factor = C_R_max if C_R_max < 1 else (C_R_max - 1)
        # Guard against invalid or too narrow ranges
        if rmax_factor <= 0.1:
            rmax_factor = max(C_R_max, 0.11)
        rmax = rmax_factor * halo_rv
        nbin = 100                
        Xbins = np.logspace(np.log10(rmin), np.log10(rmax), nbin + 1)
        Xpoints = np.sqrt(Xbins[:-1] * Xbins[1:])  # 几何中心
        # 计算 bin 中心点
        Xpoints = (Xbins + np.roll(Xbins, -1))[:-1] / 2
    return Xbins, Xpoints


def process_halo_block(selected_indices, data_entry, HaloRV, HaloMV, Lbox, Grid_idx, grid_size, Xbins, a, h, b = 0, end_factor = 8.5):
    mH = 1.67 * 10**-24
    MpcTocm = 3.08567758 * 10**24
    MsunTog = 1.988 * 10**33
    profiles_block = {}
    Data_local_idx = {} 
    halo_coords = {}
    particle_type = ['sphpos', 'dmpos', 'starpos']
    Lbox_corrected = Lbox/h*a
    for j in selected_indices:
        if j < len(data_entry["halopos"]) and HaloRV[j] != 0:  # 确保不超出halopos的长度
            halo_center = data_entry["halopos"][j]
            search_grid_indices = get_nearby_grid_indices(halo_center*h/a, grid_size, Lbox)
            for particle in particle_type:
                Data_local_idx[particle] = [Grid_idx[particle][m] for m in search_grid_indices]
                Data_local_idx[particle] = [item for sublist in Data_local_idx[particle] for item in sublist]
            sphpos_local = data_entry["sphpos"][Data_local_idx["sphpos"]] 
            dmpos_local = data_entry["dmpos"][Data_local_idx["dmpos"]] 
            starpos_local = data_entry["starpos"][Data_local_idx["starpos"]] 
            smoothlen_local = data_entry["smoothlen"][Data_local_idx["sphpos"]]
            f_e = data_entry["n_e"][Data_local_idx["sphpos"]]
            idx_sph, dist_sph = find_particles_within_RV_sph(sphpos_local, halo_center, end_factor * HaloRV[j], Lbox_corrected)
            idx_DM, dist_DM = find_particles_within_RV(dmpos_local, halo_center, end_factor * HaloRV[j], Lbox_corrected)
            idx_star, dist_star = find_particles_within_RV(starpos_local, halo_center, end_factor * HaloRV[j], Lbox_corrected)
            new_sphpos_local = np.zeros_like(sphpos_local)
            new_dmpos_local = np.zeros_like(dmpos_local)
            new_starpos_local = np.zeros_like(starpos_local)
            for q in range(len(idx_sph)):
                if idx_sph[q] < len(sphpos_local):
                    new_sphpos_local[idx_sph[q]] = sphpos_local[idx_sph[q]]
                if idx_sph[q] >= len(sphpos_local):
                    new_sphpos_local[idx_sph[q]-len(sphpos_local)] = np.mod(sphpos_local[idx_sph[q]-len(sphpos_local)] - halo_center + Lbox_corrected/2,  Lbox_corrected) -  Lbox_corrected/2 + halo_center
                    #new_sphpos_local[idx_sph[q]-len(sphpos_local)] = sphpos_local[idx_sph[q]-len(sphpos_local)]
                    idx_sph[q] = idx_sph[q] - len(sphpos_local)
            sphmass_near = data_entry["sphmass"][Data_local_idx["sphpos"]][list(idx_sph)]
            dmmass_near = data_entry["dmmass"][Data_local_idx["dmpos"]][list(idx_DM)]
            starmass_near = data_entry["starmass"][Data_local_idx["starpos"]][list(idx_star)]
            f_e_near = data_entry["n_e"][Data_local_idx["sphpos"]][list(idx_sph)]
            new_sphpos_near = new_sphpos_local[list(idx_sph)]
            new_dmpos_near = new_dmpos_local[list(idx_DM)]
            new_starpos_near = new_starpos_local[list(idx_star)]
            smoothlen_near =  smoothlen_local[list(idx_sph)]
            f_e_near = f_e[list(idx_sph)]
           
            halo_coords[j] = {
                "sphlogr": np.where(dist_sph > 0, np.log10(dist_sph), np.nan),
                "dmlogr": np.where(dist_DM > 0, np.log10(dist_DM), np.nan),
                "starlogr": np.where(dist_star > 0, np.log10(dist_star), np.nan),
                "sphidx": idx_sph,
                "dmidx": idx_DM,
                "staridx": idx_star,
                "sphpos": new_sphpos_near,
                "dmpos": new_dmpos_near,
                "starpos": new_starpos_near,
                "smoothlen": smoothlen_near,
                "sphmass": sphmass_near,
                "dmmass": dmmass_near,
                "starmass": starmass_near,
                "f_e": f_e_near,
                "haloidx": j
            }

            if j%50 == 0:
                # 计算对数空间的密度分布
                print("sphlogr shape:", halo_coords[j]["sphlogr"].shape)
                print("sphmass_near shape:", sphmass_near.shape)
                print("f_e_near shape:", f_e_near.shape)
                print("weights shape:", (1e10 * (sphmass_near * f_e_near).astype(np.float64) / mH * MsunTog).shape)
                print("HaloRV:", HaloRV[j])
            mshell_sph = np.histogram(halo_coords[j]["sphlogr"], bins=Xbins, weights=1e10 * sphmass_near)[0]
            mshell_dm = np.histogram(halo_coords[j]["dmlogr"], bins=Xbins, weights=1e10 * dmmass_near)[0]
            mshell_star = np.histogram(halo_coords[j]["starlogr"], bins=Xbins, weights=1e10 * starmass_near)[0]
            neshell_sph = np.histogram(halo_coords[j]["sphlogr"], bins=Xbins, weights=1e10 * (sphmass_near * f_e_near).astype(np.float64) / mH * MsunTog)[0]
            vshell = 4/3 * np.pi * (np.power(10, Xbins))**3
            vshell = (np.roll(vshell, -1) - vshell)[0:-1]
            
            # 使用新的 log_linear_rv_more 刻度
            Xbins_rv, Xpoints = calculate_bins("log_linear_rv_more_outer", HaloRV[j], b=b, C_R_max=end_factor)
            Xpoints_rv = Xpoints / HaloRV[j]

            # 计算质量分布（sph、dm、star 和 electron density）
            mshell_sph_rv = np.histogram(dist_sph, bins=Xbins_rv, weights=1e10 * sphmass_near)[0]
            mshell_dm_rv = np.histogram(dist_DM, bins=Xbins_rv, weights=1e10 * dmmass_near)[0]
            mshell_star_rv = np.histogram(dist_star, bins=Xbins_rv, weights=1e10 * starmass_near)[0]
            neshell_sph_rv = np.histogram(dist_sph, bins=Xbins_rv, weights=1e10 * (sphmass_near * f_e_near).astype(np.float64) / mH * MsunTog)[0]

            # 计算壳层体积
            vshell_rv = 4/3 * np.pi * (Xbins_rv)**3
            vshell_rv = (np.roll(vshell_rv, -1) - vshell_rv)[:-1]  # 确保去掉最后一个无效体积

            profiles_block[j] = {
                "sphrho": mshell_sph / vshell,
                "dmrho": mshell_dm / vshell,
                "starrho": mshell_star / vshell,
                "ne": neshell_sph / vshell / (MpcTocm)**3,
                "sphrho_rv": mshell_sph_rv / vshell_rv,
                "dmrho_rv": mshell_dm_rv / vshell_rv,
                "starrho_rv": mshell_star_rv / vshell_rv,
                "ne_rv": neshell_sph_rv / vshell_rv / (MpcTocm)**3,
                "Xpoints_rv": Xpoints_rv,
                "NFW": NFW(np.power(10, Xbins[:-1]), HaloRV[j], HaloMV[j]),
                "Modified_NFW": Modified_NFW(np.power(10, Xbins[:-1]), HaloRV[j], HaloMV[j]),
                "Halo_RV": HaloRV[j],
                "Halo_MV": HaloMV[j]
            }
    return profiles_block, halo_coords

def save_plot(density_data_array, x_values, xlabel, ylabel, box_size, formatted_key, plot_type, filename_prefix):
    median_curve = np.median(density_data_array, axis=0)
    linear_mean_curve = np.mean(density_data_array, axis=0)
    log_mean_curve = log_mean_without_zeros(density_data_array)
    std_curve_lower, std_curve_upper = calculate_error_bands(density_data_array)
    plt.plot(x_values, median_curve, label='Median Curve')
    plt.plot(x_values, linear_mean_curve, label='Linear Mean Curve')
    plt.plot(x_values, log_mean_curve, label='Log Mean Curve')
    plt.plot(x_values, std_curve_lower, label='1-sigma Std', color='red', linestyle="dashed")
    plt.plot(x_values, std_curve_upper, label='1-sigma Std', color='red', linestyle="dashed")

    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel(ylabel, fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=13)
    figure_dir = os.path.join(directory_path, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, f"{filename_prefix}_{box_size}_{formatted_key}.png")
    plt.savefig(figure_path, bbox_inches="tight", dpi=400)
    plt.close()

def plot_density_profiles(profiles, Xpoints_tot, density_type, box_size, formatted_key, label_y, plot_type, filename_prefix, C_R_max=8.5):
    plt.figure(figsize=(8, 6))
    plt.title(f'{density_type.capitalize()} Density Profile ({plot_type}) - Mass Range ${formatted_key}$')
    density_data_list = []

    rv_x_values_ref = None
    for j in profiles[mass_range_key].keys():
        profile = profiles[mass_range_key][j][f"{density_type}_rv" if plot_type == "rv" else density_type]
        density_data_list.append(profile)
        if plot_type == "rv":
            if "Xpoints_rv" in profiles[mass_range_key][j]:
                x_values = np.array(profiles[mass_range_key][j]["Xpoints_rv"])
            else:
                _, x_values = calculate_bins("log_linear_rv_more_outer", 1, b=b, C_R_max=C_R_max)
            if rv_x_values_ref is None:
                rv_x_values_ref = x_values
        else:
            x_values = np.power(10, Xpoints_tot[k])
        plt.plot(x_values, profile, alpha=0.1, color='k', linewidth=0.5)
        if plot_type == "r":
            plt.xscale("log")
        plt.yscale("log")

    density_data_array = np.array(density_data_list)
    if plot_type == "rv":
        x_values_summary = rv_x_values_ref if rv_x_values_ref is not None else calculate_bins("log_linear_rv_more_outer", 1, b=b, C_R_max=C_R_max)[1]
    else:
        x_values_summary = np.power(10, Xpoints_tot[k])
    save_plot(density_data_array, x_values_summary, "r / R$_{vir}$" if plot_type == "rv" else "r [Mpc]", label_y, box_size, formatted_key, plot_type, filename_prefix)




def log_mean_without_zeros(arr):
    arr = np.array(arr, dtype=np.float64, copy=True)
    arr[arr <= 0] = np.nan  # 非正值统一置为 NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        log_arr = np.log(arr)
    valid_cols = np.any(~np.isnan(log_arr), axis=0)
    log_mean = np.full(log_arr.shape[1], np.nan, dtype=np.float64)
    if np.any(valid_cols):
        log_mean[valid_cols] = np.nanmean(log_arr[:, valid_cols], axis=0)
    return np.exp(log_mean)

# 提取文件名中的信息生成box_size_labels

def extract_box_size_labels(filename):
    labels = []
    Box_size = []
    Feedback_info = []
    parts = filename.split('_')
    box_size = parts[1]
    Box_size.append(box_size)
    version = parts[4][1] if len(parts)>4 else ""
    feedback_info = (parts[3].split('.')[0] if len(parts) > 3 else "noAGN") + version
    Feedback_info.append(feedback_info)
    label = f"{box_size} Mpc ({feedback_info})"
    labels.append(label)      
    return box_size, feedback_info, label

############CROCODIEL_v2
def extract_box_size_labels(filename):
    labels = []
    Box_size = []
    Time_info = []
    Feedback_info = []

    # 拆分文件名并提取信息
    parts = filename.split('_')
    
    # 提取 Box Size
    box_size = parts[1] if len(parts) > 1 else "Unknown"
    Box_size.append(box_size)
    time_info = parts[2] if len(parts) > 1 else "Unknown"
    Time_info.append(time_info)
    # 提取反馈信息，包括 "fiducial2" 和其他情况
    if len(parts) > 3:
        part3_lower = parts[3].lower()
        if "fiducial" in part3_lower:
            if len(parts) == 4:
                feedback_info = "fiducial"
            elif len(parts) > 4:
                version = parts[4].split('.')[0]
                feedback_info = f"fiducial_{version}"
        elif "noagn" in part3_lower:
            feedback_info = "noAGN"
        else:
            feedback_info = parts[3].split('.')[0]
    else:
        feedback_info = "noAGN"  # 默认值

    Feedback_info.append(feedback_info)

    # 生成标签
    label = f"{box_size} Mpc ({feedback_info})"
    labels.append(label)
    
    return box_size, time_info, feedback_info, label



def hdf5_data_generator(data_filenames, halo_filenames, chunk_size=5000000):
    assert len(data_filenames) == len(halo_filenames), "Data and Halo filenames lists should have the same length."
    data_content = {}
    halo_content = {}
    for i, (data_file, halo_file) in enumerate(zip(data_filenames, halo_filenames)):
        with h5py.File(data_file, "r") as data_hf, h5py.File(halo_file, "r") as halo_hf:
            # 从data文件读取数据
            data_content.setdefault(i, {}) 
            halo_content.setdefault(i, {}) 
            for key in data_hf["data"].keys():
                if isinstance(data_hf["data"][key], h5py.Dataset):
                    dataset = data_hf["data"][key]
                    if dataset.size > chunk_size:
                        data_content[i][key] = [dataset[j:j+chunk_size].astype(np.float32) for j in range(0, dataset.size, chunk_size)]
                    else:
                        data_content[i][key] = dataset[:].astype(np.float32)
                else:
                    data_content[i][key] = data_hf["data"][key].value

            # 从halo文件读取数据
            for key in halo_hf.keys():
                if isinstance(halo_hf[key], h5py.Dataset):
                    dataset = halo_hf[key]
                    if dataset.size > chunk_size:
                        halo_content[i][key] = [dataset[j:j+chunk_size].astype(np.float32) for j in range(0, dataset.size, chunk_size)]
                    else:
                        halo_content[i][key] = dataset[:].astype(np.float32)
                else:
                    halo_content[i][key] = halo_hf[key].value
        yield data_content, halo_content
        
def hdf5_data_generator(data_filenames, halo_filenames, chunk_size=5000000):
    assert len(data_filenames) == len(halo_filenames), "Data and Halo filenames lists should have the same length."
    data_content = {}
    halo_content = {}
    for i, (data_file, halo_file) in enumerate(zip(data_filenames, halo_filenames)):
        with h5py.File(data_file, "r") as data_hf, h5py.File(halo_file, "r") as halo_hf:
            # 从data文件读取数据
            data_content.setdefault(i, {})
            halo_content.setdefault(i, {})
            if "HubbleParam" in data_hf.attrs:
                data_content[i]["__HubbleParam__"] = float(data_hf.attrs["HubbleParam"])
            if "Time" in data_hf.attrs:
                data_content[i]["__Time__"] = float(data_hf.attrs["Time"])
            for key in data_hf["data"].keys():
                if isinstance(data_hf["data"][key], h5py.Dataset):
                    dataset = data_hf["data"][key]
                    if dataset.size > chunk_size:
                        data_content[i][key] = [dataset[j:j+chunk_size].astype(np.float32) for j in range(0, dataset.size, chunk_size)]
                    else:
                        data_content[i][key] = dataset[:].astype(np.float32)
                else:
                    data_content[i][key] = data_hf["data"][key][()]

            # 从halo文件读取数据
            for key in halo_hf.keys():
                if isinstance(halo_hf[key], h5py.Dataset):
                    dataset = halo_hf[key]
                    if dataset.size > chunk_size:
                        halo_content[i][key] = [dataset[j:j+chunk_size].astype(np.float32) for j in range(0, dataset.size, chunk_size)]
                    else:
                        halo_content[i][key] = dataset[:].astype(np.float32)
                else:
                    halo_content[i][key] = halo_hf[key][()]

        # 合并chunked数据
        for key in data_content[i].keys():
            if isinstance(data_content[i][key], list):
                data_content[i][key] = np.concatenate(data_content[i][key])

        for key in halo_content[i].keys():
            if isinstance(halo_content[i][key], list):
                halo_content[i][key] = np.concatenate(halo_content[i][key])

        yield data_content, halo_content

def save_halo_info_and_profiles(Halo_Information, all_profiles, box_size, time_info, feedback_info, directory_path, b=0, C_R_max=8.5):
    # 定义文件路径
    b_str = str(b).replace('.', 'p')
    c_r_max_str = str(C_R_max).replace('.', 'p')
    Halo_info_name = f"haloinfo_{box_size}_{time_info}_{feedback_info}_morebin_ineq_outer_imapct{b_str}_{c_r_max_str}RVnew.h5"
    all_profile_name = f"all_profile_{box_size}_{time_info}_{feedback_info}_morebin_ineq_outer_impact{b_str}_{c_r_max_str}RVnew.h5"
    sys.stdout.flush()
    Halo_info_path = os.path.join(directory_path, Halo_info_name)
    all_profile_path = os.path.join(directory_path, all_profile_name)
    print("########################")
    print(f"all_profile_path: {all_profile_path}")
    print(f"Halo_info_path: {Halo_info_path}")
    print("########################")
    sys.stdout.flush()
    # 删除已存在的文件
    if os.path.exists(Halo_info_path):
        os.remove(Halo_info_path)
    if os.path.exists(all_profile_path):
        os.remove(all_profile_path)
    
    # 保存 Halo_Information
    with h5py.File(Halo_info_path, 'w') as f:
        for sub_key, sub_value in Halo_Information.items():
            if isinstance(sub_value, (list, tuple, np.ndarray)):
                f.create_dataset(str(sub_key), data=sub_value)
            elif isinstance(sub_value, dict):
                sub_group = f.create_group(str(sub_key))
                for sub_sub_key, sub_sub_value in sub_value.items():
                    if isinstance(sub_sub_value, (list, tuple, np.ndarray)):
                        sub_group.create_dataset(str(sub_sub_key), data=sub_sub_value)
                    elif isinstance(sub_sub_value, dict):
                        sub_sub_group = sub_group.create_group(str(sub_sub_key))
                        for sub_sub_sub_key, sub_sub_sub_value in sub_sub_value.items():
                            if isinstance(sub_sub_sub_value, (list, tuple, np.ndarray)):
                                if sub_sub_sub_key in sub_sub_group:
                                    del sub_sub_group[sub_sub_sub_key]
                                sub_sub_group.create_dataset(str(sub_sub_sub_key), data=sub_sub_sub_value)
                            else:
                                sub_sub_group.attrs[str(sub_sub_sub_key)] = sub_sub_sub_value
                    else:
                        sub_group.attrs[str(sub_sub_key)] = sub_sub_value
            else:
                f.attrs[str(sub_key)] = sub_value
                    
    # 保存对应的 all_profiles
    with h5py.File(all_profile_path, 'w') as f:
        for sub_key, sub_value in all_profiles.items():
            if isinstance(sub_value, (list, tuple, np.ndarray)):
                f.create_dataset(str(sub_key), data=sub_value)
            elif isinstance(sub_value, dict):
                sub_group = f.create_group(str(sub_key))
                for sub_sub_key, sub_sub_value in sub_value.items():
                    if isinstance(sub_sub_value, (list, tuple, np.ndarray)):
                        sub_group.create_dataset(str(sub_sub_key), data=sub_sub_value)
                    elif isinstance(sub_sub_value, dict):
                        sub_sub_group = sub_group.create_group(str(sub_sub_key))
                        for sub_sub_sub_key, sub_sub_sub_value in sub_sub_value.items():
                            if isinstance(sub_sub_sub_value, (list, tuple, np.ndarray)):
                                if sub_sub_sub_key in sub_sub_group:
                                    del sub_sub_group[sub_sub_sub_key]
                                sub_sub_group.create_dataset(str(sub_sub_sub_key), data=sub_sub_sub_value)
                            else:
                                sub_sub_group.attrs[str(sub_sub_sub_key)] = sub_sub_sub_value
                    else:
                        sub_group.attrs[str(sub_sub_key)] = sub_sub_value
            else:
                f.attrs[str(sub_key)] = sub_value

    print(f"Data storing part for Haloinfo_{box_size}")
    print(f"Data storing part for profile_{box_size}")
    sys.stdout.flush()

def parse_boxname(name):
    token = str(name).strip().strip('"').strip("'")
    parts = token.split('_')
    snap = "020"
    model = "fiducial"
    version = ""
    if len(parts) >= 1 and parts[0].isdigit():
        snap = parts[0].zfill(3)
    if len(parts) >= 2:
        model_token = parts[1].lower()
        if model_token.startswith("f"):
            model = "fiducial"
        elif model_token.startswith("n"):
            model = "noAGN"
        else:
            model = parts[1]
        if parts[1][-1].isdigit():
            version = parts[1][-1]
    if len(parts) >= 3:
        if parts[2].lower().startswith("v") and parts[2][1:].isdigit():
            version = parts[2][1:]
        elif parts[2].isdigit():
            version = parts[2]
    return snap, model, version


def generate_file_lists(Boxsize, Boxname, data_source):
    data_files = []
    halo_files = []
    if data_source == "halo2d":
        base_dir = "/sqfs/work/hp240141/z6b340/results/Halo_data_2D"
        for size, name in zip(Boxsize, Boxname):
            snap, model, version = parse_boxname(name)
            model_lower = model.lower()
            if model_lower.startswith("fid"):
                model_cap = "Fiducial"
                version_str = f"_v{version}" if version else "_v1"
            elif model_lower.startswith("noagn") or model_lower.startswith("n") or model_lower.startswith("nobh"):
                # halo2d no-AGN naming is fixed to NoBH and does not track code version
                model_cap = "NoBH"
                version_str = ""
            else:
                model_cap = model
                version_str = f"_v{version}" if version else "_v1"
            data_file = os.path.join(base_dir, f"snap_{int(snap)}", f"data_{size}_{snap}_{model_cap}{version_str}_with_stellar.h5")
            halo_file = os.path.join(base_dir, f"snap_{int(snap)}", f"halo_{size}_{snap}_{model_cap}{version_str}_with_stellar.h5")
            data_files.append(data_file)
            halo_files.append(halo_file)
    else:
        base_dir = "/sqfs/work/hp240141/z6b340/results/CROCODILE_v1/"
        for size, name in zip(Boxsize, Boxname):
            snap, model, version = parse_boxname(name)
            suffix = f"{model}_v{version}" if version else model
            data_files.append(os.path.join(base_dir, f"data_{size}_{snap}_{suffix}.h5"))
            halo_files.append(os.path.join(base_dir, f"halo_{size}_{snap}_{suffix}.h5"))
    return data_files, halo_files


default_directory_path = "/sqfs/work/hp240141/z6b340/results/CROCODILE_v1/"

default_a_ = [1]
default_h_ = [0.6774]
default_Boxsize = [50]
default_Boxname = ["20_n"]
default_b = 0
default_C_R_max = 8.5
default_data_source = "crocodile_v1"

def normalize_optional_cosmo_input(value, default_value, target_len):
    if value is None:
        return [default_value] * target_len
    if not isinstance(value, list):
        value = [value]
    if len(value) == 1 and target_len > 1:
        value = value * target_len
    return value

# 检查命令行参数
if len(sys.argv) == 1:
    a_ = default_a_
    h_ = default_h_
    Boxsize = default_Boxsize
    Boxname = default_Boxname
    b = default_b
    C_R_max = default_C_R_max
    data_source = default_data_source
elif len(sys.argv) in [6, 7, 8]:
    try:
        Boxsize = ast.literal_eval(sys.argv[1])
        Boxname = ast.literal_eval(sys.argv[2])
        b = ast.literal_eval(sys.argv[3])
        C_R_max = ast.literal_eval(sys.argv[4])
        data_source = str(sys.argv[5]).strip().strip('"').strip("'").lower()
        a_input = ast.literal_eval(sys.argv[6]) if len(sys.argv) >= 7 else None
        h_input = ast.literal_eval(sys.argv[7]) if len(sys.argv) >= 8 else None
    
        if not isinstance(Boxsize, list):
            Boxsize = [Boxsize]
        if not isinstance(Boxname, list):
            Boxname = [Boxname]
        a_ = normalize_optional_cosmo_input(a_input, default_a_[0], len(Boxsize))
        h_ = normalize_optional_cosmo_input(h_input, default_h_[0], len(Boxsize))
        if not isinstance(b, numbers.Number):
            print("Error: b must be a number.")
            sys.exit(1)
        if not isinstance(C_R_max, numbers.Number):
            print("Error: C_R_max must be a number.")
            sys.exit(1)

    except (ValueError, SyntaxError) as e:
        print(f"Error: Unable to parse inputs. {e}")
else:
    print("Usage:")
    print("  python3 input_a_h_test.py ")
    print("  python3 density_profile_rv_morebins.py <Boxsize> <Boxname> <b> <C_R_max> <data_source> [a_] [h_]")
    print("Example: python3 density_profile_rv_morebins.py 100 '\"20_f\"' 0.5 8.5 halo2d")
    print("Example: python3 density_profile_rv_morebins.py '[100,100]' '[\"20_f\",\"18_f\"]' 0.5 8.5 crocodile_v1 '[1,1]' '[0.6774,0.6774]'")
    sys.exit(1)

if data_source not in ["crocodile_v1", "halo2d"]:
    print("Error: data_source must be one of ['crocodile_v1', 'halo2d'].")
    sys.exit(1)

# 校验长度一致
if len(a_) != len(h_) or len(a_) != len(Boxsize) or len(a_) != len(Boxname):
    print("Error: a_, h_, Boxsize, and Boxname must have the same length.")
    sys.exit(1)

# 生成 file_list


Data_filepaths, Halo_filepaths = generate_file_lists(Boxsize, Boxname, data_source)
data_files = [os.path.basename(file) for file in Data_filepaths]
halo_files = [os.path.basename(file) for file in Halo_filepaths]
directory_path = os.path.dirname(Data_filepaths[0]) + "/"

# 打印结果
print(f"Input a_: {a_}")
print(f"Input h_: {h_}")
print(f"Input Boxsize: {Boxsize}")
print(f"Input Boxname: {Boxname}")
print(f"Input impact parameter b: {b}")
print(f"Input C_R_max (end_factor): {C_R_max}")
print(f"Input data_source: {data_source}")
print("Resolved files:")
print(f"Data_filepaths: {Data_filepaths}")
print(f"Halo_filepaths: {Halo_filepaths}")

# 列出文件夹中的所有文件
#file_list = [
#            "data_25_020_fiducial.h5", "halo_25_020_fiducial.h5",
#            "data_50_020_fiducial.h5", "halo_50_020_fiducial.h5",
#            "data_50_020_noAGN.h5", "halo_50_020_noAGN.h5"
             #"data_100_100.h5", "halo_100_100.h5",
             #"data_500_024.h5", "halo_500_024.h5"
#            ]

# 列出文件夹中的所有文件
#ile_list = [
             #"data_25_020_fiducial.h5", "halo_25_020_fiducial.h5",
             #"data_50_020_fiducial2_with_stellar.h5", "halo_50_020_fiducial2_with_stellar.h5",
             #"data_50_020_fiducial.h5", "halo_50_020_fiducial.h5"
             #"data_50_020_noAGN.h5", "halo_50_020_noAGN.h5"
             #"data_100_020_fiducial.h5", "halo_100_020_fiducial.h5"
             #"data_100_020_noAGN.h5", "halo_100_020_noAGN.h5"
             #"data_100_100.h5", "halo_100_100.h5"
             #"data_500_024.h5", "halo_500_024.h5"
            #]

#Boxsize = [25,50,50,100,500]
#Boxsize = [50, 50]
#Boxsize = [100]
#Boxsize = [50]
#a_ = [0.490035]
#a_ = [0.777113]
#a_ = [0.490035, 0.490035]
#a_ = [1, 1]
#a_ = [1, 1, 1]
#h_ = [0.6774]
#h_ = [0.6774, 0.6774, 0.6774]
#Boxsize = [500]
Data_filenames = [file for file in data_files]
Halo_filenames = [file for file in halo_files]

print(Data_filepaths, Halo_filepaths)
sys.stdout.flush()

# 读取HDF5文件


# 使用迭代器读取数据和halo内容
generator = hdf5_data_generator(Data_filepaths, Halo_filepaths)   


# 定义不同质量范围（原始）
mass_ranges_old = [
    (10**9.5, 10**10.5),   # 质量范围 10^9 M_sun 到 10^10 M_sun
    (10**10.5, 10**11.5),   # 质量范围 10^11 M_sun 到 10^12 M_sun
    (10**11.5, 10**12.5),   # 质量范围 10^12 M_sun 到 10^13 M_sun
    (10**12.5, 10**13.5),   # 质量范围 10^13 M_sun 到 10^14 M_sun
    (10**13.5, 10**14.5),   # 质量范围 10^14 M_sun 到 10^15 M_sun
    (10**14.5, 10**15.5),   # 质量范围 10^15 M_sun 到 10^16 M_sun
]

# 定义不同质量范围 new
mass_ranges_new = [
    (10**9, 10**10),   # 质量范围 10^9 M_sun 到 10^10 M_sun
    (10**10, 10**11),   # 质量范围 10^11 M_sun 到 10^12 M_sun
    (10**11, 10**12),   # 质量范围 10^12 M_sun 到 10^13 M_sun
    (10**12, 10**13),   # 质量范围 10^13 M_sun 到 10^14 M_sun
    (10**13, 10**14),   # 质量范围 10^14 M_sun 到 10^15 M_sun
    (10**14, 10**15),   # 质量范围 10^15 M_sun 到 10^16 M_sun
]

# 运行时同时包含原始区间和新区间（在原有区间之后追加）
mass_ranges = mass_ranges_new
# 存储每个质量范围所选的索引的字典
selected_indices_dict = {}
Halo_Information = {}
mass_range_key_total = {}
all_profiles = {}
Grid_indices = {} 
Grid_size = []
processed_halos = 0

Boxsize_Labels = []
Halo_info_filename = []
Halo_info_path = []
all_profile_path = []

for data_content, halo_content in generator:
    mass_range_key_total = {}
    total_snapfiles = len(data_content)
    print(f"tot_snapfile: {total_snapfiles}")
    sys.stdout.flush()

for i in range(total_snapfiles):
    attr_h = data_content[i].get("__HubbleParam__", None)
    attr_a = data_content[i].get("__Time__", None)
    if attr_h is not None and attr_a is not None:
        h_[i] = float(attr_h)
        a_[i] = float(attr_a)
        print(f"[Auto cosmology] snapfile {i}: use HubbleParam={h_[i]}, Time={a_[i]} from data attributes.")
    else:
        h_[i] = float(default_h_[0])
        a_[i] = float(default_a_[0])
        print(f"[Default cosmology] snapfile {i}: HubbleParam/Time missing, fallback to h={h_[i]}, a={a_[i]}.")
    sys.stdout.flush()

for i in range(total_snapfiles):
    Grid_size.append([Boxsize[i]/5, Boxsize[i]/5, Boxsize[i]/5])
    Grid_indices[i] = create_grid_indices(data_content[i], Grid_size[i], Boxsize[i], a_[i], h_[i])
print(f"len_grid_indices: {len(Grid_indices)}")

sys.stdout.flush()
for i in range(total_snapfiles):
    selected_indices_dict.setdefault(i, {})
    Halo_Information.setdefault(i, {})
    total_halos_temp = len(data_content[i]["halopos"])
    print(f"Total halos in snapfile {i+1}: {total_halos_temp}")
    total_halos = sum([len(data_content[i]["halopos"]) for i in range(total_snapfiles)])
    print(f"total_halos: {total_halos}")
    sys.stdout.flush()
    print(f"data_content_len: {len(data_content)}")
    mass_range_key_total.setdefault(i, {}) 
    all_profiles.setdefault(i, {}) 
    for k, mass_range in enumerate(mass_ranges):
        min_mass, max_mass = mass_range
        selected_indices_range = []

        for j in range(total_halos_temp):
            halo_mass = halo_content[i]["Halomass"][j]*1e10

            if min_mass <= halo_mass <= max_mass:
                selected_indices_range.append(j)

        # 随机选择200个索引
        if len(selected_indices_range) > 500:
            selected_indices_range = random.sample(selected_indices_range, 500)

        # 存储选取的索引到字典中
        mass_range_key = f"{min_mass:.2e}-{max_mass:.2e}"
        mass_range_key_total[i][k] = mass_range_key
        selected_indices_dict[i][mass_range_key] = selected_indices_range
        print(f"Selected {len(selected_indices_range)} halos in mass range {min_mass:.2e} - {max_mass:.2e}")

    # 遍历选取的索引并处理相应的halo
    halo_tot_usingtocal = sum([len(selected_indices_dict[i][x]) for q, x in mass_range_key_total[i].items()])
    start_time = time.time()
    Halo_cal_num = 0
    Xbins_tot = []
    Xpoints_tot = []
    Xbins_log = np.logspace(-2, 1, num=4)
    # 计算每一份的大小
    interval_sizes = np.diff(Xbins_log)
    # 创建一个新的数组来存储均等分成5个bin的结果
    Xbins_log_extended = []
    for mass_range_key, selected_indices_range in selected_indices_dict[i].items():
        if len(selected_indices_range) != 0: 
            for kk in range(len(Xbins_log) - 1):
                start_value = Xbins_log[kk]
                end_value = Xbins_log[kk + 1]
                interval_size = interval_sizes[kk]

                # 使用np.logspace来创建均等的log bin
                sub_bins = np.linspace(start_value, start_value + interval_size, num=10)
                Xbins_log_extended.extend(sub_bins[:-1])
            Xbins = np.log10(np.array(Xbins_log_extended[0:20]))
            Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
            Xbins_tot.append(Xbins)
            Xpoints_tot.append(Xpoints)
        else:
            Xbins = []
            Xpoints = []
        block_profiles, halo_info_selected = process_halo_block(
            selected_indices_range,
            data_content[i],
            halo_content[i]["HaloRV"],
            halo_content[i]["HaloMV"],
            Boxsize[i],
            Grid_indices[i],
            Grid_size[i],
            Xbins,
            a_[i],
            h_[i],
            b=b,
            end_factor=C_R_max
        )
        
        for j in selected_indices_range:
            # 处理 halo_content[i]["HaloRV"][j] 和 data_content[i]["halopos"][j]
            # 获取当前处理的 halo 的数据
            halo_rv = halo_content[i]["HaloRV"][j]
            halo_pos = data_content[i]["halopos"][j]
            # 设置 mass_range_key 对应的 all_profiles
            if mass_range_key not in all_profiles[i]:
                all_profiles[i][mass_range_key] = {}
                Halo_Information[i][mass_range_key] = {}
            if halo_content[i]["HaloRV"][j]!=0:
                all_profiles[i][mass_range_key][j] = block_profiles[j]
                Halo_Information[i][mass_range_key][j] = halo_info_selected[j]
        elapsed_time = time.time() - start_time
        if len(selected_indices_range) != 0: 
            average_time_per_halo = elapsed_time / (len(selected_indices_range))
            remaining_time = (halo_tot_usingtocal - len(selected_indices_range)) * average_time_per_halo

            hours, remainder = divmod(remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Processed snapfile {i+1}/{total_snapfiles}, halo {len(selected_indices_range)+1}/{halo_tot_usingtocal}. Estimated remaining time: {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds.")
            print(all_profiles[i][mass_range_key][j]["sphrho"])
            sys.stdout.flush()

    box_size, time_info, feedback_info, box_size_labels = extract_box_size_labels(Data_filenames[i])
    #Boxsize.append(box_size)
    Boxsize_Labels.append(box_size_labels)
    b_str = str(b).replace('.', 'p')
    c_r_max_str = str(C_R_max).replace('.', 'p')
    Halo_info_name = "haloinfo_{}_{}_{}_morebin_ineq_outer_imapct{}_{}RV.h5".format(box_size, time_info, feedback_info, b_str, c_r_max_str)
    all_profile_name = "all_profile_{}_{}_{}_morebin_ineq_outer_impact{}_{}RV_new.h5".format(box_size, time_info, feedback_info, b_str, c_r_max_str)
    output_directory_path = os.path.dirname(Data_filepaths[i]) + "/"
    new_path = os.path.join(output_directory_path, Halo_info_name)
    new_path2 = os.path.join(output_directory_path, all_profile_name)
    Halo_info_path.append(new_path)
    all_profile_path.append(new_path2)
    save_halo_info_and_profiles(Halo_Information[i], all_profiles[i], box_size, time_info, feedback_info, output_directory_path, b=b, C_R_max=C_R_max)
    if i == total_snapfiles -1:
        print("finished the Data storing part")
        sys.stdout.flush()
    try:
        for k, mass_range_key in enumerate(all_profiles[i].keys()):
            formatted_key = format_mass_range(mass_range_key)
            
            # Plot profiles based on full Xbins
            plot_density_profiles(all_profiles[i], Xpoints_tot, "ne", box_size, formatted_key, "n_e [cm$^{-3}$]", "r", "Electron_Density_Box_full", C_R_max=C_R_max)
            plot_density_profiles(all_profiles[i], Xpoints_tot, "ne", box_size, formatted_key, "n_e [cm$^{-3}$]", "rv", "Electron_Density_Box_rv", C_R_max=C_R_max)

            plot_density_profiles(all_profiles[i], Xpoints_tot, "sphrho", box_size, formatted_key, "$\\rho_{\mathrm{gas}}$ [$M_{\odot}$ kpc$^{-3}$]", "r", "Gas_Density_Box_full", C_R_max=C_R_max)
            plot_density_profiles(all_profiles[i], Xpoints_tot, "sphrho", box_size, formatted_key, "$\\rho_{\mathrm{gas}}$ [$M_{\odot}$ kpc$^{-3}$]", "rv", "Gas_Density_Box_rv", C_R_max=C_R_max)

            plot_density_profiles(all_profiles[i], Xpoints_tot, "dmrho", box_size, formatted_key, "$\\rho_{\mathrm{DM}}$ [$M_{\odot}$ kpc$^{-3}$]", "r", "DM_Density_Box_full", C_R_max=C_R_max)
            plot_density_profiles(all_profiles[i], Xpoints_tot, "dmrho", box_size, formatted_key, "$\\rho_{\mathrm{DM}}$ [$M_{\odot}$ kpc$^{-3}$]", "rv", "DM_Density_Box_rv", C_R_max=C_R_max)

            plot_density_profiles(all_profiles[i], Xpoints_tot, "starrho", box_size, formatted_key, "$\\rho_{\mathrm{star}}$ [$M_{\odot}$ kpc$^{-3}$]", "r", "Star_Density_Box_full", C_R_max=C_R_max)
            plot_density_profiles(all_profiles[i], Xpoints_tot, "starrho", box_size, formatted_key, "$\\rho_{\mathrm{star}}$ [$M_{\odot}$ kpc$^{-3}$]", "rv", "Star_Density_Box_rv", C_R_max=C_R_max)
    except: 
        pass    
    print(f"Finished plotting for Box_{box_size}")
    sys.stdout.flush()


# 存储整体的 all_profiles
overall_halo_profiles_path = os.path.join(directory_path, f'haloinfo_overall_{Boxsize[0]}.h5')
if os.path.exists(overall_halo_profiles_path):
    os.remove(overall_halo_profiles_path)
overall_halo_profiles = {str(i): all_profiles[i] for i in range(total_snapfiles)}

# 存储整体的 all_profiles
overall_all_profiles_path = os.path.join(directory_path, f'halodata_overall_{Boxsize[0]}.h5')
if os.path.exists(overall_all_profiles_path):
    os.remove(overall_all_profiles_path )
overall_all_profiles = {str(i): all_profiles[i] for i in range(total_snapfiles)}


with h5py.File(overall_halo_profiles_path, 'w') as f:
    for key, value in overall_halo_profiles.items():
        group = f.create_group(str(key))
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, (list,  tuple, np.ndarray)):
                group.create_dataset(str(sub_key), data=sub_value)
            elif isinstance(sub_value, dict):
                sub_group = group.create_group(str(sub_key))
                for sub_sub_key, sub_sub_value in sub_value.items():
                    if isinstance(sub_sub_value, (list, tuple, np.ndarray)):
                        sub_group.create_dataset(str(sub_sub_key), data=sub_sub_value)
                    elif isinstance(sub_value, dict):
                        sub_sub_group = sub_group.create_group(str(sub_sub_key))
                        for sub_sub_sub_key, sub_sub_sub_value in sub_sub_value.items():
                            if isinstance(sub_sub_sub_value, (list, tuple, np.ndarray)):
                                if sub_sub_sub_key in sub_sub_group:
                                    del  sub_sub_group[sub_sub_sub_key]
                                sub_sub_group.create_dataset(str(sub_sub_sub_key), data=sub_sub_sub_value)
                            else:
                                sub_sub_group.attrs[str(sub_sub_sub_key)] = sub_sub_sub_value
                    else:
                        sub_group.attrs[str(sub_sub_key)] = sub_sub_value
            else:
                group.attrs[str(sub_key)] = sub_value

print("Data storing part for Halo informaiton")
sys.stdout.flush()

with h5py.File(overall_all_profiles_path, 'w') as f:
    for key, value in overall_all_profiles.items():
        group = f.create_group(str(key))
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, (list, tuple, np.ndarray)):
                group.create_dataset(str(sub_key), data=sub_value)
            elif isinstance(sub_value, dict):
                sub_group = group.create_group(str(sub_key))
                for sub_sub_key, sub_sub_value in sub_value.items():
                    if isinstance(sub_sub_value, (list, tuple, np.ndarray)):
                        sub_group.create_dataset(str(sub_sub_key), data=sub_sub_value)
                    elif isinstance(sub_sub_value, dict):
                        sub_sub_group = sub_group.create_group(str(sub_sub_key))
                        for sub_sub_sub_key, sub_sub_sub_value in sub_sub_value.items():
                            if sub_sub_sub_key in sub_sub_group:
                                del sub_sub_group[sub_sub_sub_key]
                            if isinstance(sub_sub_sub_value, (list, tuple, np.ndarray)):
                                sub_sub_group.create_dataset(str(sub_sub_sub_key), data=sub_sub_sub_value)
                            else:
                                sub_sub_group.attrs[str(sub_sub_sub_key)] = sub_sub_sub_value
                    else:
                        sub_group.attrs[str(sub_sub_key)] = sub_sub_value
            else:
                group.attrs[str(sub_key)] = sub_value

print("Data storing part for overall profiles")
sys.stdout.flush()




# 设定颜色和线条样式的列表，确保这些列表足够长，能表示所有数据集
Xbins_log = np.logspace(-2, 1, num=4)
# 计算每一份的大小
interval_sizes = np.diff(Xbins_log)
# 创建一个新的数组来存储均等分成5个bin的结果
Xbins_log_extended = []
Xbins_tot = []
Xpoints_tot = []
for mass_range_key, selected_indices_range in selected_indices_dict[i].items():
# 对于每一份，将其分成5个均等的log bin
    if len(selected_indices_range) != 0: 
        for kk in range(len(Xbins_log) - 1):
            start_value = Xbins_log[kk]
            end_value = Xbins_log[kk + 1]
            interval_size = interval_sizes[kk]

            # 使用np.logspace来创建均等的log bin
            sub_bins = np.linspace(start_value, start_value + interval_size, num=10)
            Xbins_log_extended.extend(sub_bins[:-1])
        Xbins = np.log10(np.array(Xbins_log_extended[0:20]))
        Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
        Xbins_tot.append(Xbins)
        Xpoints_tot.append(Xpoints)
    else:
        Xbins = []
        Xpoints = []


vshell = 4/3 * np.pi * (np.power(10, Xbins_tot[0]))**3
vshell = (np.roll(vshell, -1) - vshell)[0:-1]


colors_old = {
    '3.16e+9-3.16e+10': 'pink',
    '3.16e+10-3.16e+11': 'b',
    '3.16e+11-3.16e+12': 'g',
    '3.16e+12-3.16e+13': 'r',
    '3.16e+13-3.16e+14': 'c',
    '3.16e+14-3.16e+15': 'm',
    '3.16e+15-3.16e+16': 'y'
}

colors_new = {
    '1.00e+9-1.00e+10': 'pink',
    '1.00e+10-1.00e+11': 'b',
    '1.00e+11-1.00e+12': 'g',
    '1.00e+12-1.00e+13': 'r',
    '1.00e+13-1.00e+14': 'c',
    '1.00e+14-1.00e+15': 'm',
    '1.00e+15-1.00e+16': 'y'
}
colors = colors_new
linestyles = ['-', '--', '-.', ':']

def calculate_error_bands(data):
    """
    计算上下两条曲线作为误差范围。

    参数：
    data (numpy.ndarray): 数据数组。

    返回：
    (numpy.ndarray, numpy.ndarray): 返回上限曲线和下限曲线。
    """
    # 在对数空间中计算 percentiles
    data = np.array(data, dtype=np.float64, copy=True)
    data[data <= 0] = np.nan
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
# 设定颜色和线条样式的列表，确保这些列表足够长，能表示所有数据集

def format_mass_range(mass_range_key):
    # 分割mass_range_key到两部分
    parts = mass_range_key.split('-')
    
    # 将每部分转换为科学计数法并取对数，然后构造LaTeX格式的字符串
    formatted_parts = [r'10^{' + f"{round(math.log10(float(part)),1)}" + '}' for part in parts]
    
    # 将转换后的部分再次组合
    return '-'.join(formatted_parts)

def plot_profiles(all_profiles_list, all_profile_path, dtype, ylabel, plot_type="r", with_error=True):
    plt.figure(figsize=(12, 8))
    plt.xscale("log" if plot_type == "r" else "linear")
    plt.yscale("log")
    plt.xlabel("r [Mpc]" if plot_type == "r" else "r / R$_{vir}$", fontsize=17)
    plt.ylabel(ylabel, fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    for i, box_key in enumerate(all_profiles_list.keys()):
        profiles = all_profiles_list[box_key]
        all_profile_name = all_profile_path[i].split("all_profile")[1]
        
        box_size, time_info, feedback_info, box_size_labels = extract_box_size_labels(all_profile_name)
        print(box_size)
        for k, mass_range_key in enumerate(profiles.keys()):
            formatted_key = format_mass_range(mass_range_key)
            density_data_list = []
            for j in profiles[mass_range_key].keys():
                data = profiles[mass_range_key][j]
                if plot_type == "r":
                    if dtype == "ne":
                        sphrho = data["sphrho"]
                        density = Mu_e * sphrho / (mH * Mu_H) * MsunTog / (MpcTocm)**3
                    elif dtype == "rho_gas":
                        density = np.array(data["sphrho"])/(1000)**3  # For gas
                    elif dtype == "rho_dm":
                        density = np.array(data["dmrho"])/(1000)**3  # For dark matter
                    elif dtype == "rho_star":
                        density = np.array(data["starrho"])/(1000)**3  # For stars
                    density_data_list.append(density)
                else:
                    if dtype == "ne":
                        sphrho = data["sphrho_rv"]
                        density = Mu_e * sphrho / (mH * Mu_H) * MsunTog / (MpcTocm)**3
                    elif dtype == "rho_gas":
                        density = np.array(data["sphrho_rv"])/(1000)**3  # For gas
                    elif dtype == "rho_dm":
                        density = np.array(data["dmrho_rv"])/(1000)**3  # For dark matter
                    elif dtype == "rho_star":
                        density = np.array(data["starrho_rv"])/(1000)**3  # For stars
                    density_data_list.append(density)

            density_data_array = np.array(density_data_list)
            log_mean_curve = log_mean_without_zeros(density_data_array)
            if plot_type == "r":
                x_values = np.power(10, Xpoints)
            else:
                first_halo_key = next(iter(profiles[mass_range_key]), None)
                if first_halo_key is not None and "Xpoints_rv" in profiles[mass_range_key][first_halo_key]:
                    x_values = np.array(profiles[mass_range_key][first_halo_key]["Xpoints_rv"])
                else:
                    _, x_values = calculate_bins("log_linear_rv_more_outer", 1, b=b, C_R_max=C_R_max)
            if with_error:
                std_curve_lower, std_curve_upper = calculate_error_bands(density_data_array)
                plt.fill_between(x_values, std_curve_lower, std_curve_upper, color=colors.get(mass_range_key, 'k'), alpha=0.3)
            plt.plot(x_values, log_mean_curve, label=f'{box_size_labels} ${formatted_key}$' + '$M_{\odot}$', color=colors.get(mass_range_key, 'k'), linestyle=linestyles[i % len(linestyles)])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.legend(handles, labels, fontsize=13, loc='center left', bbox_to_anchor=(1, 0.5))
    figure_dir = os.path.join(directory_path, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, f"Combined_{dtype}_{'Error' if with_error else 'NoError'}_Median_Curves_{plot_type}.png")
    plt.savefig(figure_path, bbox_inches="tight", dpi=400)
    plt.show()


# Example usage with error bands
plot_profiles(all_profiles, all_profile_path, "ne", r"$n_e$ [cm$^{-3}$]", plot_type="r", with_error=True)
plot_profiles(all_profiles, all_profile_path, "rho_gas", r"$\rho_{\mathrm{gas}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="r", with_error=True)
plot_profiles(all_profiles, all_profile_path, "rho_dm", r"$\rho_{\mathrm{DM}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="r", with_error=True)
plot_profiles(all_profiles, all_profile_path, "rho_star", r"$\rho_{\mathrm{star}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="r", with_error=True)

plot_profiles(all_profiles, all_profile_path, "ne", r"n_e$ [cm$^{-3}$]", plot_type="rv", with_error=True)
plot_profiles(all_profiles, all_profile_path, "rho_gas", r"$\rho_{\mathrm{gas}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="rv", with_error=True)
plot_profiles(all_profiles, all_profile_path, "rho_dm", r"$\rho_{\mathrm{DM}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="rv", with_error=True)
plot_profiles(all_profiles, all_profile_path, "rho_star", r"$\rho_{\mathrm{star}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="rv", with_error=True)

# Example usage without error bands
plot_profiles(all_profiles, all_profile_path, "ne", r"$n_e$ [cm$^{-3}$]", plot_type="r", with_error=False)
plot_profiles(all_profiles, all_profile_path, "rho_gas", r"$\rho_{\mathrm{gas}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="r", with_error=False)
plot_profiles(all_profiles, all_profile_path, "rho_dm", r"$\rho_{\mathrm{DM}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="r", with_error=False)
plot_profiles(all_profiles, all_profile_path, "rho_star", r"$\rho_{\mathrm{star}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="r", with_error=False)

plot_profiles(all_profiles, all_profile_path, "ne", r"$n_e$ [cm$^{-3}$]", plot_type="rv", with_error=False)
plot_profiles(all_profiles, all_profile_path, "rho_gas", r"$\rho_{\mathrm{gas}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="rv", with_error=False)
plot_profiles(all_profiles, all_profile_path, "rho_dm", r"$\rho_{\mathrm{DM}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="rv", with_error=False)
plot_profiles(all_profiles, all_profile_path, "rho_star", r"$\rho_{\mathrm{star}}$ [$M_{\odot}$ kpc$^{-3}$]", plot_type="rv", with_error=False)


plt.close('all')

print("Finsh the plotting part 2")
sys.stdout.flush()
