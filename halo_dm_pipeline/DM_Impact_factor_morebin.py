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
import scipy
import time
from scipy.spatial import cKDTree
import numpy as np
from scipy.optimize import root_scalar
import numpy as np
from scipy.integrate import quad
from joblib import Parallel, delayed
import multiprocessing
import random
import ast
global JJ
JJ = 0
##################################
from astropy.constants import G, c, m_e, m_p

me = 9.1*10**-28
mH = 1.67*10**-24
solartog = 2.0*10**33
MpcTocm = 3.08567758*10**24
MsunTog = 1.988*10**33
h = 0.67742
mp = m_p.to(u.g).value
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

def create_grid_indices(data, grid_size, boxsize, a, h):
    num_grids = int(np.prod(grid_size))
    #grid_indices = {particle_type: {} for particle_type in ["sphpos", "dmpos", "starpos"]}
    grid_indices = {particle_type: {} for particle_type in ["sphpos", "starpos"]}
    #for particle_type in ["sphpos", "dmpos", "starpos"]:
    for particle_type in ["sphpos", "starpos"]:
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

def handle_periodic_boundary(pos, Lbox):
    """
    处理pos中的周期性边界条件，使得所有坐标都在[0, Lbox]范围内。
    """
    pos = np.mod(pos, Lbox)
    return pos
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

def find_particles_within_RV(pos, halo_center, RV, Lbox):
    pos = handle_periodic_boundary(pos, Lbox)

    # Apply periodic boundary conditions to halo center
    halo_center = np.mod(halo_center, Lbox)

    # Build KDTree with periodic boundary conditions
    tree = cKDTree(pos, boxsize=Lbox)

    # Find all particles within RV of halo center
    idx = tree.query_ball_point(halo_center, RV)

    # Ensure idx is a numpy array
    idx = np.asarray(idx)

    # Debug: print idx and its type

    # Calculate distances of all particles within RV
    if idx.size == 0:
        return np.array([]), np.array([])

    pos_within_rv = pos[idx]
    dist = np.sqrt(np.sum((pos_within_rv - halo_center)**2, axis=1))

    # Check for particles that cross periodic boundaries
    crossings = pos_within_rv - halo_center
    crossings = np.mod(crossings + Lbox/2, Lbox) - Lbox/2
    crossing_dist = np.sqrt(np.sum((crossings)**2, axis=1))
    mask = crossing_dist < dist
    idx[mask] = idx[mask]
    dist[mask] = crossing_dist[mask]

    return idx, dist

def find_particles_within_RV_sph(pos, halo_center, RV, Lbox):
    # Apply periodic boundary conditions to halo center
    halo_center = np.mod(halo_center, Lbox)
    #print(halo_center, RV)
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



def find_particles_within_RV_sph(pos, halo_center, RV, Lbox):
    # 对 pos 进行周期性边界处理
    pos = handle_periodic_boundary(pos, Lbox)

    # Apply periodic boundary conditions to halo center
    halo_center = np.mod(halo_center, Lbox)

    # Build KDTree with periodic boundary conditions
    tree = cKDTree(pos, boxsize=Lbox)

    # Find all particles within RV of halo center
    idx = tree.query_ball_point(halo_center, RV)

    # Ensure idx is a numpy array
    idx = np.asarray(idx)

    # Debug: print idx and its type
    #print(f"idx length: {len(idx)}")
    #if len(idx) > 0:
    #    print(f"First 10 indices: {idx[:10]}")

    # Calculate distances of all particles within RV
    if idx.size == 0:
        return np.array([]), np.array([])

    pos_within_rv = pos[idx]
    dist = np.sqrt(np.sum((pos_within_rv - halo_center)**2, axis=1))

    # Check for particles that cross periodic boundaries
    crossings = pos_within_rv - halo_center
    crossings = np.mod(crossings + Lbox/2, Lbox) - Lbox/2
    crossing_dist = np.sqrt(np.sum((crossings)**2, axis=1))
    mask = crossing_dist < dist
    idx[mask] = idx[mask] + len(pos)
    dist[mask] = crossing_dist[mask]

    return idx, dist



# 计算共动距离和尺度因子的函数
def comoving_distance(z):
        # 宇宙学参数
    Omega_m = 0.31  # 物质密度参数
    Omega_Lambda = 0.69  # 暗能量密度参数
    H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
    c = 299792.458  # 光速，单位为 km/s
    integrand = lambda zp: 1.0 / np.sqrt(Omega_m * (1.0+zp)**3 + Omega_Lambda)
    integral, _ = quad(integrand, 0, z)
    return c/H_0 * integral

def scale_factor(z):
    return 1.0 / (1.0 + z)
    
def find_redshift_and_scale_factor(dC):
    #print(f"dC: {dC}")
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

def calc_ne_LOS_shifted(k, i, LOS_shifted, sphpos, h_smooth_max, Lbox, SPHmasstocgs, f_e, mp, Smoothlen):
    try:
        ne_LC_shifted_k_i = 0
        # 确保传递给 find_particles_within_RV_sph 的参数是 numpy 数组，并且是 float 类型
        LOS_shifted_k_i = np.array(LOS_shifted[k][i], dtype=float)
        sphpos = np.array(sphpos, dtype=float)
        
        idx_LC, dist_LC = find_particles_within_RV_sph(sphpos, LOS_shifted_k_i, h_smooth_max, Lbox)
        sys.stdout.flush()
        for j in range(len(idx_LC)):
            if idx_LC[j] >= len(sphpos):
                idx_LC[j] = idx_LC[j] - len(sphpos)
                sys.stdout.flush()
            if dist_LC[j] >= Smoothlen[idx_LC[j]]:
                continue
            else:
                weight = M6(dist_LC[j] * MpcTocm, 0, Smoothlen[idx_LC[j]] * MpcTocm, 3)
                ne_LC_shifted_k_i += SPHmasstocgs[idx_LC[j]] * f_e[idx_LC[j]] / mp * weight
        return ne_LC_shifted_k_i
    except Exception as e:
        print(f"Error in calc_ne_LOS_shifted: {e}")
        return 0

def calculate_segment_and_bins(b, R, max_bins=400):
    """
    Calculate the segment length and number of bins based on the impact factor b and Halo radius R.
    
    Parameters:
    b : float
        Impact factor.
    R : float
        Radius of the Halo.
    max_bins : int
        Maximum number of bins to be used when b=0.
        
    Returns:
    tuple
        Segment length and number of bins.
    """
    # 计算穿过圆的线段长度
    if b <= R:
        L = 2 * np.sqrt(R**2 - b**2)
        num_bins = max(1, int(round(max_bins * L / (2*R))))
    else:
        L = 0
        num_bins = 1
    return L, num_bins


def calculate_segment_and_bins_fixed_length(L_fixed, max_bins=400):
    """
    Calculate the number of bins based on a fixed LOS length.
    
    Parameters:
    L_fixed : float
        Fixed length of the LOS.
    max_bins : int
        Maximum number of bins to be used.
        
    Returns:
    int
        Number of bins.
    """
    num_bins = max(1, max_bins)
    return L_fixed, num_bins

# Example usage:
#R = 50  # Example Halo radius
#b_values = np.linspace(0,R, 10)

# 更新后的部分代码
def calculate_bins(binning_type, halo_rv=None):
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
        Xbins = np.linspace(0, 2 * halo_rv, 21)
        Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
    elif binning_type == "log_linear_rv":
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
    return Xbins, Xpoints
def calculate_density_DM(halo_local_data, HaloRV, halomass, HaloMV, halo_center, b_values, Lbox, axis = 0):
    #print(f"mp:{mp}")
    Omega_0 = 0.31  # 物质密度参数
    Omega_Lambda = 0.69  # 暗能量密度参数
    H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
    c = 299792.458  # 光速，单位为 km/s
    SPHmasstocgs = 1.989e+43
    SPHdensitytocgs = 6.76991e-31
    results = {}
    b_num = len(b_values)
    LOS_origin = np.zeros([b_num, 3])
    if axis == 0: 
        u2 = np.array([1,0,0])
    if axis == 1: 
        u2 = np.array([0,1,0])
    if axis == 2: 
        u2 = np.array([0,0,1])
    z_list_tot = []
    dz_tot = []
    n_bins_tot = []
    idx = halo_local_data['haloidx']
    h_smooth_local = halo_local_data['smoothlen']
    h_smooth_max = np.max(h_smooth_local)
    sphpos_local = halo_local_data['sphpos']
    sphmass_g_local = halo_local_data['sphmass']*1.989e+43
    f_e_within_local = halo_local_data['f_e']
    Radius = HaloRV * 2
    L_tot = []
    z_list =[]
    bin_centers_pos = []
    LOS_shifted = []
    L_fixed = 2.0  # 固定的 LOS 长度（可以根据需要调整）
    max_bins = 400  # 可以调整这个值以优化结果
    for i, b in enumerate(b_values):
        #L, n_bins = calculate_segment_and_bins(b, HaloRV)
        L, n_bins = calculate_segment_and_bins_fixed_length(HaloRV, max_bins)
        L_tot.append(L)
        n_bins_tot.append(n_bins)
        bin_edges = np.linspace(0, L, n_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_pos.append(np.array([u2*x for x in bin_centers]))
        z_list = np.array([output1 for output1, output2 in map(find_redshift_and_scale_factor, bin_edges)])
        z_current = ((z_list[1:] + z_list[:-1])/2)
        z_list_tot.append(z_current) #redshift
        dz =  z_list[1:] - z_list[:-1]
        dz_tot.append(dz)
        LOS_origin[i, 0] = halo_center[0]
        LOS_origin[i, 1] = halo_center[1] + b
        LOS_origin[i, 2] = halo_center[2]
        LOS_shifted.append(np.array(bin_centers_pos[i] +  LOS_origin[i]))
    LOS_shifted = np.array(LOS_shifted, dtype=object)
    for sublist_2d in LOS_shifted:
        for sublist_1d in sublist_2d:
            if len(sublist_1d) > 0:  # 确保子列表至少有一个元素
                sublist_1d[0] %= Lbox
                sublist_1d[1] %= Lbox
                sublist_1d[2] %= Lbox
                sublist_1d[0] += Lbox * (sublist_1d[0] < 0)
                sublist_1d[1] += Lbox * (sublist_1d[1] < 0)
                sublist_1d[2] += Lbox * (sublist_1d[2] < 0)
    ne_LOS_shifted = np.empty(len(b_values), dtype=object)
    # 使用相应的 n_bins 值填充数组
    for i, n_bins in enumerate(n_bins_tot):
        ne_LOS_shifted[i] = np.zeros(n_bins)
    DM_arr = np.zeros(len(b_values))
    num_cores = multiprocessing.cpu_count()  # 获取CPU核数
    #print(LOS_shifted)
    #print(f"np.max_pos: {np.max(sphpos_local)}")
    #for k in range(b_num):
        #for i in range(n_bins_tot[k]):
            #ne_LOS_shifted[k][i] = calc_ne_LOS_shifted(k,i,LOS_shifted, sphpos_local, h_smooth_max, Lbox, sphmass_g_local, f_e_within_local, mp, h_smooth_local)
            #print(f"LOS_shifted[k][i]: {LOS_shifted[k][i]}")
    #ne_results_0 = calc_ne_LOS_shifted(0,0,LOS_shifted, sphpos_local, h_smooth_max, Lbox, sphmass_g_local, f_e_within_local, mp, h_smooth_local)
        #print(f"ne_results_0:{ne_LOS_shifted[k]}")
    ne_results = Parallel(n_jobs=num_cores)(delayed(calc_ne_LOS_shifted)(k, i, LOS_shifted, sphpos_local, h_smooth_max, Lbox, sphmass_g_local, f_e_within_local, mp, h_smooth_local) for k in range(b_num) for i in range(n_bins_tot[k]))
    for k in range(b_num):
        for i in range(n_bins_tot[k]):
            ne_LOS_shifted[k][i] = ne_results[k*n_bins+i]
    DM_IGM_num = []
    DM_Halo_LOS = np.empty(len(b_values), dtype=object)
    DM_Halo_LOS_tot = np.zeros(len(b_values))
    for i in range(b_num):
        DM_IGM_num.append(len(dz_tot[i]))
        DM_Halo_LOS[i] = np.zeros(len(dz_tot[i])) 
        for j in range(DM_IGM_num[i]-1):
            if len(DM_Halo_LOS)>1: 
                DM_Halo_LOS[i][0] = c/H_0*ne_LOS_shifted[i][0]*(1+z_list_tot[i][0])*dz_tot[i][0]/np.sqrt(Omega_0*(1+z_list_tot[i][0])**3+Omega_Lambda)*1e6
                DM_Halo_LOS[i][j+1] = DM_Halo_LOS[i][j] + c/H_0*ne_LOS_shifted[i][j+1]*(1+z_list_tot[i][j+1])*dz_tot[i][j+1]/np.sqrt(Omega_0*(1+z_list_tot[i][j+1])**3+Omega_Lambda)*1e6
                DM_Halo_LOS_tot[i] = DM_Halo_LOS[i][j] 
                #sys.stdout.flush()
            else: 
                DM_Halo_LOS[i][0] = c/H_0*ne_LOS_shifted[i][0]*(1+z_list_tot[i][0])*dz_tot[i][0]/np.sqrt(Omega_0*(1+z_list[i][0])**3+Omega_Lambda)*1e6
    results = {
        'halo': f'Halo_{idx}',
        'n_e': ne_LOS_shifted,
        'DM': 2*DM_Halo_LOS_tot,
        'b_values': b_values,
        "Halo_RV": HaloRV,
        "Halo_MV": HaloMV,
        "Halomass": halomass
    }
    #print(f"DM_Halo_LOS_tot:{DM_Halo_LOS_tot}")
    return results

# 选择一个文件名作为例子

def process_halo_block(selected_indicies, data_entry, Halomass, HaloMV, HaloRV, Lbox, Grid_idx, grid_size, Xbins, a, h):
    profiles_block = {}
    Data_local_idx = {} 
    halo_coords = {}
    halo_local_data = {}
    ne_DM_results = {}
    #particle_type = ['sphpos', 'dmpos', 'starpos']
    particle_type = ['sphpos', 'starpos']
    # 找到每个Halo附近的粒子
    JJ = 0
    for j in selected_indicies:
        if j < len(data_entry["halopos"]) and HaloRV[j]!=0:  # 确保不超出halopos的长度
            halo_center = data_entry["halopos"][j]
            search_grid_indices = get_nearby_grid_indices(halo_center*h/a, grid_size, Lbox)
            for particle in particle_type:
                Data_local_idx[particle] = [Grid_idx[particle][m] for m in search_grid_indices]
                Data_local_idx[particle]  = [item for sublist in Data_local_idx[particle] for item in sublist]
            sphpos_local = data_entry["sphpos"][Data_local_idx["sphpos"]]
            #dmpos_local = data_entry["dmpos"][Data_local_idx["dmpos"]]*h
            starpos_local = data_entry["starpos"][Data_local_idx["starpos"]]
            smoothlen_local = data_entry["smoothlen"][Data_local_idx["sphpos"]]
            f_e = data_entry["n_e"][Data_local_idx["sphpos"]]
            idx_sph, dist_sph = find_particles_within_RV_sph(sphpos_local, halo_center, 5.0 * HaloRV[j], Lbox/h*a)
            #idx_DM, dist_DM = find_particles_within_RV(dmpos_local, halo_center, 5.0 * HaloRV[j], Lbox)
            idx_star, dist_star = find_particles_within_RV(starpos_local, halo_center, 5.0 * HaloRV[j],Lbox/h*a)
            new_sphpos_local = np.zeros_like(sphpos_local)
            #new_dmpos_local = np.zeros_like(dmpos_local)
            new_starpos_local = np.zeros_like(starpos_local)
            for q in range(len(idx_sph)):
                if idx_sph[q] < len(sphpos_local):
                    new_sphpos_local[idx_sph[q]] = sphpos_local[idx_sph[q]]
                if idx_sph[q] >= len(sphpos_local):
                    #new_sphpos_local[idx_sph[q]-len(sphpos_local)] = np.mod(sphpos_local[idx_sph[q]-len(sphpos_local)] - halo_center + Lbox/2, Lbox) - Lbox/2 + halo_center
                    new_sphpos_local[idx_sph[q]-len(sphpos_local)] = sphpos_local[idx_sph[q]-len(sphpos_local)]
                    idx_sph[q] = idx_sph[q] - len(sphpos_local)
            sphmass_near = data_entry["sphmass"][Data_local_idx["sphpos"]][list(idx_sph)]
            #dmmass_near = data_entry["dmmass"][Data_local_idx["dmpos"]][list(idx_DM)]

            # Debug: 打印 starmass 和 starpos 的类型和内容
            #print(f"Data entry {j}:")
            #print(f"starmass type: {type(data_entry['starmass'])}")
            #print(f"starmass shape: {data_entry['starmass'].shape if hasattr(data_entry['starmass'], 'shape') else 'N/A'}")
            #print(f"Data_local_idx['starpos'] type: {type(Data_local_idx['starpos'])}")
            #print(f"Data_local_idx['starpos'] shape: {Data_local_idx['starpos'].shape if hasattr(Data_local_idx['starpos'], 'shape') else 'N/A'}")
            #print(f"Data_local_idx['starpos'] max index: {np.max(Data_local_idx['starpos'])}")
            #print(f"Data_local_idx['starpos'] min index: {np.min(Data_local_idx['starpos'])}")
            #if len(idx_star) > 0:
            #    print(f"idx_star max index: {np.max(idx_star)}")
            #    print(f"idx_star min index: {np.min(idx_star)}")
            #    print(f"idx_star length: {len(idx_star)}")
            #    print(f"idx_star: {idx_star}")
            #else:
             #   print("idx_star is empty.")

            #sys.stdout.flush()
            # 检查所有索引是否在有效范围内
            starmass_near = data_entry["starmass"][Data_local_idx["starpos"]][list(idx_star)]
            new_sphpos_near = new_sphpos_local[list(idx_sph)]
            #new_dmpos_near = new_dmpos_local[list(idx_DM)]
            new_starpos_near = new_starpos_local[list(idx_star)]
            smoothlen_near =  smoothlen_local[list(idx_sph)]
            f_e_near = f_e[list(idx_sph)]
            # 检查类型
            #print("Type of new_sphpos_near:", type(new_sphpos_near))
            #print("Type of new_starpos_near:", type(new_starpos_near))
            #print("Type of smoothlen_near:", type(smoothlen_near))
            #print("Type of f_e_near:", type(f_e_near))
            #sys.stdout.flush()
            #ne_DM_H_resutls = calculate_density_DM(new_sphpos_local, HaloRV[j], halo_center, b_values, n_bins, axis=1)
            #bin_edges = np.linspace(0, los_length, n_bins+1)
            #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            #d_bin = bin_edges[1:] - bin_edges[:-1]
            Xbins_rv, Xpoints_rv = calculate_bins("log_linear_rv", HaloRV[j])
            #b_values = np.linspace(0,2*HaloRV[j], 20)
            b_values = Xbins_rv
            halo_local_data[j] = {
                "sphpos": new_sphpos_near,
                "smoothlen": smoothlen_near,
                "sphmass": sphmass_near,
                "f_e": f_e_near,
                "haloidx": j,
                "b_values": b_values,
                "Halo_RV": HaloRV[j],
                "Halo_MV": HaloMV[j],
                "Halomass":  Halomass[j],
                "halo_center": halo_center
            }
            ne_DM_results[j] = calculate_density_DM(halo_local_data[j], HaloRV[j], Halomass[j], HaloMV[j], halo_center, b_values, Lbox/h*a)
    
            #print(f"ne_DM_results[j]: {ne_DM_results[j] }")
            JJ += 1
            if JJ%100 == 0:
                print(f" HaloRV:{HaloRV[j]}")
                print(f"Inside function: {process_halo_block.__name__}")
                print(f"We have processed {JJ} Halos, Left {len(HaloRV)-JJ} halos. ")
                #print(f"ne_DM_results[j]:{ne_DM_results[j]}.")
                print(f"###################")
                print(f"ne_DM_results[DM]:{ne_DM_results[j]['DM']}")
                sys.stdout.flush()
    return ne_DM_results,halo_local_data

#for mass_range_key, selected_indices_range in selected_indices_dict[i].items():
    #print(selected_indices_range)
def format_mass_range(mass_range_key):
    # 分割mass_range_key到两部分
    parts = mass_range_key.split('-')
    
    # 将每部分转换为科学计数法并取对数，然后构造LaTeX格式的字符串
    formatted_parts = [r'10^{' + f"{round(math.log10(float(part)),1)}" + '}' for part in parts]
    
    # 将转换后的部分再次组合
    return '-'.join(formatted_parts)

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
    percentiles = [16, 84]  # 16th and 84th percentiles
    percentile_values = np.nanpercentile(log_data, percentiles, axis=0)

    # 上下限曲线
    lower_bound_curve = percentile_values[0]
    upper_bound_curve = percentile_values[1]

    return np.exp(lower_bound_curve), np.exp(upper_bound_curve)

def generate_file_list(Boxsize, Boxname):
    file_list = []
    for size, name in zip(Boxsize, Boxname):
        if "n" in name:
            # 如果是 noAGN
            if name[-1].isdigit():
                suffix = f"noAGN_v{name[-1]}"
            else:
                suffix = "noAGN"
        elif "f" in name:
            # 如果是 fiducial
            if name[-1].isdigit():
                suffix = f"fiducial_v{name[-1]}"
            else:
                suffix = "fiducial"
        else:
            suffix = "unknown"  # 未知命名规则，可用于调试

        # 添加文件路径
        data_file = f"data_{size}_020_{suffix}.h5"
        halo_file = f"halo_{size}_020_{suffix}.h5"
        file_list.append(data_file)
        file_list.append(halo_file)
    
    return file_list
# 设置文件夹路径
#directory_path = "/sqfs/work/hp230089/z6b340/results/"
directory_path = "/sqfs/work/hp240141/z6b340/results/CROCODILE_v1/"
# 列出文件夹中的所有文件

#file_list = ["data_500_024.h5", "halo_500_024.h5"]

default_a_ = [1]
default_h_ = [0.6774]
default_Boxsize = [50]
default_Boxname = ["20_n"]
default_factor_RV = 2
# 检查命令行参数
# 初始化变量，先用默认值
a_ = default_a_
h_ = default_h_
Boxsize = default_Boxsize
Boxname = default_Boxname
factor_RV = default_factor_RV

# 处理输入参数
if len(sys.argv) > 1:
    try:
        if len(sys.argv) > 1:
            a_ = ast.literal_eval(sys.argv[1])
            if not isinstance(a_, list):
                a_ = [a_]

        if len(sys.argv) > 2:
            h_ = ast.literal_eval(sys.argv[2])
            if not isinstance(h_, list):
                h_ = [h_]

        if len(sys.argv) > 3:
            Boxsize = ast.literal_eval(sys.argv[3])
            if not isinstance(Boxsize, list):
                Boxsize = [Boxsize]

        if len(sys.argv) > 4:
            Boxname = ast.literal_eval(sys.argv[4])
            if not isinstance(Boxname, list):
                Boxname = [Boxname]

        if len(sys.argv) > 5:
            factor_RV = ast.literal_eval(sys.argv[5])
        
    except (ValueError, SyntaxError) as e:
        print(f"Error: Unable to parse inputs. {e}")
        sys.exit(1)

# 打印最终的参数值
print(f"a_ = {a_}")
print(f"h_ = {h_}")
print(f"Boxsize = {Boxsize}")
print(f"Boxname = {Boxname}")
print(f"factor_RV = {factor_RV}")
# 校验长度一致
if len(a_) != len(h_) or len(a_) != len(Boxsize):
    print("Error: a_, h_, and Boxsize must have the same length.")

# 生成 file_list


file_list = generate_file_list(Boxsize, Boxname)

# 打印结果
print(f"Input a_: {a_}")
print(f"Input h_: {h_}")
print(f"Input Boxsize: {Boxsize}")
print(f"Input Boxname: {Boxname}")
print("Generated file_list:")
print(f"file_list:{file_list}")


# 列出文件夹中的所有文件
#file_list = [
             #"data_25_020_fiducial.h5", "halo_25_020_fiducial.h5",
             #"data_50_020_fiducial_v1.h5", "halo_50_020_fiducial_v1.h5"
             #"data_50_020_fiducial_v2.h5", "halo_50_020_fiducial_v2.h5"
             #"data_50_020_noAGN.h5", "halo_50_020_noAGN.h5"
             #"data_50_015_fiducial.h5", "halo_50_015_fiducial.h5",
             #"data_50_015_noAGN.h5", "halo_50_015_noAGN.h5"
             #"data_100_100.h5", "halo_100_100.h5",
             #"data_500_024.h5", "halo_500_024.h5"
            #]

#file_list = [
#             "data_25_020_fiducial.h5", "halo_25_020_fiducial.h5",
             #"data_100_018_fiducial.h5", "halo_100_018_fiducial.h5"
             #"data_100_018_noAGN.h5", "halo_100_018_noAGN.h5"
#             "data_50_020_fiducial_with_stellar.h5", "halo_50_020_fiducial_with_stellar.h5"
             #"data_100_015_fiducial.h5", "halo_100_015_fiducial.h5"
             #"data_100_015_noAGN.h5", "halo_100_015_noAGN.h5"
             #"data_100_020_noAGN.h5", "halo_100_020_noAGN.h5"
             #"data_100_020_fiducial.h5", "halo_100_020_fiducial.h5"
             #"data_100_100.h5", "halo_100_100.h5",
             #"data_500_024.h5", "halo_500_024.h5"
#            ]
# 根据文件名过滤得到Data和Halo文件
data_files = [file for file in file_list if "data" in file]
halo_files = [file for file in file_list if "halo" in file]
# 排序文件，确保顺序一致
data_files.sort()
halo_files.sort()

# 为文件创建完整路径
Data_filename = [directory_path + file for file in data_files]
Halo_filename = [directory_path + file for file in halo_files]

# 读取HDF5文件
def hdf5_data_generator(data_filenames, halo_filenames):
    # 确保两个文件名列表的长度相同
    assert len(data_filenames) == len(halo_filenames), "Data and Halo filenames lists should have the same length."
    data_content = {}
    halo_content = {}
    for i, (data_file, halo_file) in enumerate(zip(data_filenames, halo_filenames)):
        with h5py.File(data_file, "r") as data_hf, h5py.File(halo_file, "r") as halo_hf:
            # 从data文件读取数据
            data_content.setdefault(i, {}) 
            for key in data_hf["data"].keys():
                print('key:', key)
                if isinstance(data_hf["data"][key], h5py.Dataset):
                    data_content[i][key] = data_hf["data"][key][:].astype(np.float32)
                else:
                    data_content[i][key] = data_hf["data"][key].value
                print(data_content[i][key][0])

            # 从halo文件读取数据
            halo_content.setdefault(i, {}) 
            for key in halo_hf.keys():
                print('key:', key)
                if isinstance(halo_hf[key], h5py.Dataset):
                    halo_content[i][key] = halo_hf[key][:].astype(np.float32)
                else:
                    halo_content[i][key] = halo_hf[key].value
                print(halo_content[i][key][0])
        
        yield data_content, halo_content

def hdf5_data_generator(data_filenames, halo_filenames, chunk_size=50000000):
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

# 计算snapfiles的数量和halos的总数，这部分移到for循环外部
# 使用迭代器读取数据和halo内容
generator = hdf5_data_generator(Data_filename, Halo_filename)

# 存储每个质量范围所选的索引的字典
selected_indices_dict = {}
Halo_Information = {}
# 定义不同质量范围
mass_ranges = [
    (10**10.5, 10**11.5),   # 质量范围 10^10.5 M_sun 到 10^11.5 M_sun
    (10**11.5, 10**12.5),   # 质量范围 10^11.5 M_sun 到 10^12.5 M_sun
    (10**12.5, 10**13.5),   # 质量范围 10^12.5 M_sun 到 10^13.5 M_sun
    (10**13.5, 10**14.5),   # 质量范围 10^13.5 M_sun 到 10^14.5 M_sun
    (10**14.5, 10**15.5),   # 质量范围 10^14.5 M_sun 到 10^15.5 M_sun
]
all_ne_DM = {}
mass_range_key_total = {}
Grid_indices = {} 
Grid_size = []
processed_halos = 0
block_size = 10
#Boxsize = [100, 500]
#Boxsize = [100, 500]
#Boxsize = [50, 50,50,50]
#Boxsize = [100, 100,100,100]
#Boxname = ["18_f", "18_n", "15_f", "15_n"]
#Boxsize = [100,100]
#Boxname = ["15_f", "15_n"]
#Boxname = ["18_f", "18_n"]
#Boxname = ["20_f", "20_n"]
#Boxsize = [50, 50, 50]
#Boxsize = [50]
#a_ = [0.490035]
#a_ = [0.777113]
#a_ = [0.490035, 0.490035]
#a_ = [1, 1, 1]
#a_ = [1]
#h_ = [0.6774, 0.6774, 0.6774]
#h_ = [0.6774]
#Boxsize = [100]
#Boxname = ["20_f1", "20_f2", "20_n"]
#Boxname = ["20_f1"]
colors = {
    '3.16e+10-3.16e+11': 'b',
    '3.16e+11-3.16e+12': 'g',
    '3.16e+12-3.16e+13': 'r',
    '3.16e+13-3.16e+14': 'c',
    '3.16e+14-3.16e+15': 'm',
    '3.16e+15-3.16e+16': 'y'
}
#Boxsize = [500]
#Boxsize = [100]
Xbins = np.linspace(-2, 1, num=20)
Xpoints = (Xbins + np.roll(Xbins, -1))[0:-1] / 2
for data_content, halo_content in generator:
    # 计算snapfiles的数量和halos的总数
    total_snapfiles = len(data_content)
    print(f"tot_snapfile: {total_snapfiles}")
    sys.stdout.flush()



for i in range(total_snapfiles):
    Grid_size.append([Boxsize[i]/5, Boxsize[i]/5, Boxsize[i]/5])
    Grid_indices[i] = create_grid_indices(data_content[i], Grid_size[i], Boxsize[i], a_[i], h_[i])
print(f"len_grid_indices: {len(Grid_indices)}")
sys.stdout.flush()
for i in range(total_snapfiles):
    print("mass_min:", np.log10(np.min(halo_content[i]["Halomass"]*1e10)))
    print("pos_max:", np.max(data_content[i]["sphpos"]))
    print("pos_max:", np.max(data_content[i]["sphpos"]))
    sys.stdout.flush()
    selected_indices_dict.setdefault(i, {})
    Halo_Information.setdefault(i, {})
    total_halos_temp = len(data_content[i]["halopos"])
    print(f"Total halos in snapfile {i+1}: {total_halos_temp}")
    total_halos = sum([len(data_content[i]["halopos"]) for i in range(total_snapfiles)])
    print(f"total_halos: {total_halos}")
    sys.stdout.flush()
    print(f"data_content_len: {len(data_content)}")
    #Grid_size.append([Boxsize[i]/4, Boxsize[i]/4, Boxsize[i]/4])
    #Grid_indices[i] = create_grid_indices(data_content[i], Grid_size[i], Boxsize[i])
    mass_range_key_total.setdefault(i, {}) 
    all_ne_DM.setdefault(i, {}) 
    for k, mass_range in enumerate(mass_ranges):
        min_mass, max_mass = mass_range
        selected_indices_range = []

        for j in range(total_halos_temp):
            halo_mass = halo_content[i]["Halomass"][j]*1e10      
            if min_mass <= halo_mass <= max_mass:

                selected_indices_range.append(j)
            
        # 随机选择500个索引
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
    for mass_range_key, selected_indices_range in selected_indices_dict[i].items():
        ne_DM_results, halo_local_data_selected = process_halo_block(selected_indices_range, data_content[i], halo_content[i]["Halomass"], halo_content[i]["HaloMV"], halo_content[i]["HaloRV"], Boxsize[i], Grid_indices[i], Grid_size[i], Xbins, a_[i], h_[i])
        for j in selected_indices_range:
            if mass_range_key not in all_ne_DM[i]:
                Halo_Information[i][mass_range_key] = {}
                all_ne_DM[i][mass_range_key] = {}
            if halo_content[i]["HaloRV"][j]!=0:
                all_ne_DM[i][mass_range_key][j] = ne_DM_results[j]
                Halo_Information[i][mass_range_key][j] = halo_local_data_selected[j]
        elapsed_time = time.time() - start_time
        if len(selected_indices_range) != 0: 
            Halo_cal_num += len(selected_indices_range)
            average_time_per_halo = elapsed_time / (Halo_cal_num)
            remaining_time = (halo_tot_usingtocal - Halo_cal_num) * average_time_per_halo

            hours, remainder = divmod(remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Processed snapfile {i+1}/{total_snapfiles}, halo {Halo_cal_num }/{halo_tot_usingtocal}. Estimated remaining time: {hours:.0f} hours {minutes:.0f} minutes {seconds:.2f} seconds.")
            #print(all_ne_DM[i][mass_range_key][j]["DM"])


    for k, mass_range_key in enumerate(all_ne_DM[i].keys()):
        formatted_key = format_mass_range(mass_range_key)
        plt.figure(figsize=(8, 6))
        plt.title(f'DM Impact factor {i + 1} - Mass Range ${formatted_key}$')
        DM_list = []
        b_values_list = []
        for j in Halo_Information[i][mass_range_key].keys():
            DM_impact = all_ne_DM[i][mass_range_key][j]['DM']
            DM_list.append(DM_impact)
            b_values = Halo_Information[i][mass_range_key][j]['b_values']
            b_values_list.append(b_values)
            #sphrho = all_profiles[i][mass_range_key][j]["sphrho"]
            plt.plot(b_values, DM_impact, alpha = 0.1, color = 'k',  linewidth=0.5)
            #plt.xscale("log")
            plt.yscale("log")

        plt.xlabel(r"log$_{10}$ r [Mpc]", fontsize=17)
        
        plt.ylabel(r"log$_{10}$ DM [pc cm$^{-3}$]", fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=13)

        plt.savefig(f"DM_Impact_factor_{Boxsize[i]} {formatted_key}_alongx_{factor_RV}RV.png", bbox_inches="tight", dpi=400)
        plt.close()
        #normalized_r = np.power(10, Xpoints_tot[k]) / halo_content[i]["HaloRV"][j]
        #sphrho = all_profiles[i][mass_range_key][j]["sphrho"]
        #profile_ne = Mu_e*sphrho/ (mH * Mu_H)*MsunTog/(MpcTocm)**3
        #density_data_list.append(profile_ne)
        
        plt.figure(figsize=(8, 6))
        plt.title(f'DM Impact factor {i + 1} - Mass Range ${formatted_key}$')
        DM_list = []
        b_values_list = []
        for j in Halo_Information[i][mass_range_key].keys():
            DM_impact = all_ne_DM[i][mass_range_key][j]['DM']
            DM_list.append(DM_impact)
            b_values = Halo_Information[i][mass_range_key][j]['b_values']/halo_content[i]["HaloRV"][j]
            b_values_list.append(b_values)
            #sphrho = all_profiles[i][mass_range_key][j]["sphrho"]
            plt.plot(b_values, DM_impact, alpha = 0.1, color = 'k',  linewidth=0.5)
            #plt.xscale("log")
            plt.yscale("log")
        
        
        DM_array = np.array(DM_list)
        print(f"DM_array,  mass_range_key:{DM_array} {mass_range_key}")
        density_data_array_logc = DM_list.copy()
        #density_data_array_logc[density_data_array_logc==0] = 1e-40
        # 计算中值曲线、线性平均值曲线、对数平均值曲线和标准差曲线
        median_curve = median_without_zeros(DM_array)
        linear_mean_curve = linear_mean_without_zeros(DM_array)
        #log_mean_curve = np.exp(np.mean(np.log(density_data_array_logc), axis=0))
        log_mean_curve = log_mean_without_zeros(DM_array)
        std_curve_lower, std_curve_upper = calculate_error_bands(DM_array)
        # 绘制曲线
        plt.plot(b_values_list[0][:-1], median_curve[:-1], label='Median Curve')
        plt.plot(b_values_list[0][:-1], linear_mean_curve[:-1], label='Linear Mean Curve')
        plt.plot(b_values_list[0][:-1], log_mean_curve[:-1], label='Log Mean Curve')
        plt.plot(b_values_list[0][:-1], std_curve_lower[:-1], label='1-sigma Std', color='red', linestyle="dashed")
        plt.plot(b_values_list[0][:-1], std_curve_upper[:-1], label='1-sigma Std', color='red', linestyle="dashed")
        plt.axvline(x=1, color='blue', linestyle='--', label = 'Viral radius')
        plt.xlabel("b/r$_{v}$", fontsize=17)
        plt.ylabel(r"log$_{10}$DM [pc cm$^{-3}$]", fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=13)

        plt.savefig(f"DM_Impact_factor_RV_{Boxsize[i]}_{Boxname[i]}_{formatted_key}_alongx_{factor_RV}RV_morebins.png", bbox_inches="tight", dpi=400)

        plt.close()

# 设定颜色和线条样式的列表，确保这些列表足够长，能表示所有数据集
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colors = {
    '3.16e+10-3.16e+11': 'b',
    '3.16e+11-3.16e+12': 'g',
    '3.16e+12-3.16e+13': 'r',
    '3.16e+13-3.16e+14': 'c',
    '3.16e+14-3.16e+15': 'm',
    '3.16e+15-3.16e+16': 'y'
}
linestyles = ['-', '--', '-.', ':']

# 遍历所有数据集
for i, profiles in enumerate(all_ne_DM):
    plt.figure(figsize=(10, 8))
    #plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("b/r$_{v}$", fontsize=17)
    plt.ylabel(r"log$_{10}$DM [pc cm$^{-3}$]", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    mass_intervals_keys = list(selected_indices_dict[i].keys())
    for k, mass_range_key in enumerate(all_ne_DM[i].keys()):
        formatted_key = format_mass_range(mass_range_key)
        DM_list = []
        #b_values = np.linspace(0,2,20)
        Xbins_rv, Xpoints_rv = calculate_bins("log_linear_rv", 1)
        b_values = Xbins_rv
        for j in Halo_Information[i][mass_range_key].keys():
            DM = all_ne_DM[i][mass_range_key][j]["DM"]
            DM_list.append(DM)

        DM_list_array = np.array(DM_list)
        std_curve_lower, std_curve_upper = calculate_error_bands(DM_list_array)
        log_mean_curve = log_mean_without_zeros(DM_list_array)
        # 选择颜色和线条样式
        color = colors[mass_range_key]
        linestyle = linestyles[i % len(linestyles)]
        
        # 绘制中值曲线
        plt.plot(b_values, log_mean_curve, label=f'Data {i + 1} - Mass Range ${formatted_key}$',
                 color=color, linestyle=linestyle)
        plt.fill_between(b_values, std_curve_lower, std_curve_upper, color=color, alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.axvline(x=1, color='blue', linestyle='--', label = 'Viral radius')
    plt.legend(fontsize=13)
    plt.savefig(f"Combined_DM_Median_Curves_{Boxsize[i]}_{Boxname[i]}_{factor_RV}RV_alongx.png", bbox_inches="tight", dpi=400)
    plt.show()
    plt.close()

    all_DM_path = os.path.join(directory_path, f'all_ne_DM_{factor_RV}RV_N500_B{Boxsize[i]}_alongx_morebins.h5')
    all_DM_Haloinfo_path = os.path.join(directory_path, f'all_DM_Haloinfo_{factor_RV}RV_N500_B{Boxsize[i]}_alongx_morebins.h5')

    #for i in range(total_snapfiles):
        #if os.path.exists(f'all_ne_DM_{Boxsize[i]}.h5'):
            #os.remove(f'all_ne_DM_{Boxsize[i]}.h5')
        #with h5py.File(f'all_ne_DM_{Boxsize[i]}.h5', 'w') as h5file:
    if os.path.exists(all_DM_path):
        os.remove(all_DM_path )
    with h5py.File(all_DM_path, 'w') as h5file:
        # 遍历 all_ne_DM 字典
        for snapfile_idx, mass_ranges_dict in all_ne_DM.items():
            # 为每个快照创建一个组
            group = h5file.create_group(f'snapfile_{snapfile_idx}')
            # 遍历每个质量范围
            for mass_range_key, halos_data in mass_ranges_dict.items():
                # 在组内为每个质量范围创建一个子组
                mass_range_group = group.create_group(mass_range_key)
                # 遍历每个 halo 的数据
                for halo_idx, halo_data in halos_data.items():
                    # 为每个 halo 创建一个数据集，这里假设 halo_data 是一个数组或列表
                    # 如果 halo_data 是一个更复杂的结构，你可能需要进一步嵌套组或调整数据保存的方式
                    halo_group = mass_range_group.create_group(f'halo_{halo_idx}')
                    for value_idx, halo_subdata in halo_data.items():
                        # 检查是否为字符串，因为字符串需要特别处理
                        if isinstance(halo_subdata, str):
                            # 创建一个单独的数据集用于字符串
                            #print(value_idx)
                            halo_group.create_dataset(value_idx, data=np.string_(halo_subdata))
                        else:
                            # 检查是否为 NumPy 数组
                            if isinstance(halo_subdata, np.ndarray) and value_idx == 'n_e':
                                lengths = [arr.shape[0] for arr in halo_subdata]  # 存储每个数组的长度
                                halo_group.create_dataset('n_e_lengths', data=lengths)  # 保存长度信息
                                # 堆叠并填充 n_e 数据
                                max_length = max(lengths)
                                stacked_n_e = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan) for arr in halo_subdata])
                                halo_group.create_dataset(f'{value_idx}', data=stacked_n_e)
                            elif isinstance(halo_subdata, np.ndarray) and value_idx != 'n_e':
                                # 存储 DM 数据
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
                            else:
                                halo_group.attrs[f'{value_idx}'] = [halo_subdata]
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
    print(f'all_ne_DM has been saved to {all_DM_path}')
    sys.stdout.flush()

    if os.path.exists(all_DM_Haloinfo_path):
        os.remove(all_DM_Haloinfo_path)
    with h5py.File(all_DM_Haloinfo_path, 'w') as h5file:
        # 遍历 all_ne_DM 字典
        for snapfile_idx, mass_ranges_dict in Halo_Information.items():
            # 为每个快照创建一个组
            group = h5file.create_group(f'snapfile_{snapfile_idx}')
            # 遍历每个质量范围
            for mass_range_key, halos_data in mass_ranges_dict.items():
                # 在组内为每个质量范围创建一个子组
                mass_range_group = group.create_group(mass_range_key)
                # 遍历每个 halo 的数据
                for halo_idx, halo_data in halos_data.items():
                    # 为每个 halo 创建一个数据集，这里假设 halo_data 是一个数组或列表
                    # 如果 halo_data 是一个更复杂的结构，你可能需要进一步嵌套组或调整数据保存的方式
                    halo_group = mass_range_group.create_group(f'halo_{halo_idx}')
                    for value_idx, halo_subdata in halo_data.items():
                        # 检查是否为字符串，因为字符串需要特别处理
                        if isinstance(halo_subdata, str):
                            # 创建一个单独的数据集用于字符串
                            #print(value_idx)
                            halo_group.create_dataset(value_idx, data=np.string_(halo_subdata))
                        else:
                            # 检查是否为 NumPy 数组
                            if isinstance(halo_subdata, np.ndarray) and value_idx == 'n_e':
                                lengths = [arr.shape[0] for arr in halo_subdata]  # 存储每个数组的长度
                                halo_group.create_dataset('n_e_lengths', data=lengths)  # 保存长度信息
                                # 堆叠并填充 n_e 数据
                                max_length = max(lengths)
                                stacked_n_e = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan) for arr in halo_subdata])
                                halo_group.create_dataset(f'{value_idx}', data=stacked_n_e)
                            elif isinstance(halo_subdata, np.ndarray) and value_idx != 'n_e':
                                # 存储 DM 数据
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
                            else:
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
    print(f'all_DM_Haloinfo has been saved to {all_DM_Haloinfo_path}')
    sys.stdout.flush()



for i, profiles in enumerate(all_ne_DM):
    all_DM_path = os.path.join(directory_path, f'all_ne_DM_{factor_RV}RV_N500_B{Boxsize[i]}_{Boxname[i]}_alongx_morebins.h5')
    all_DM_Haloinfo_path = os.path.join(directory_path, f'all_DM_Haloinfo_{factor_RV}RV_N500_B{Boxsize[i]}_{Boxname[i]}_alongx_morebins.h5')

    # 删除现有文件，避免重复
    if os.path.exists(all_DM_path):
        os.remove(all_DM_path)
    if os.path.exists(all_DM_Haloinfo_path):
        os.remove(all_DM_Haloinfo_path)

    # 保存 all_ne_DM 数据
    with h5py.File(all_DM_path, 'w') as h5file:
        # 仅保存当前 snapfile_idx 对应的数据
        snapfile_idx = i  # 假设 i 对应的索引为 snapfile_idx
        if snapfile_idx in all_ne_DM:
            mass_ranges_dict = all_ne_DM[snapfile_idx]
            group = h5file.create_group(f'snapfile_{snapfile_idx}')
            for mass_range_key, halos_data in mass_ranges_dict.items():
                mass_range_group = group.create_group(mass_range_key)
                for halo_idx, halo_data in halos_data.items():
                    halo_group = mass_range_group.create_group(f'halo_{halo_idx}')
                    for value_idx, halo_subdata in halo_data.items():
                        if isinstance(halo_subdata, str):
                            halo_group.create_dataset(value_idx, data=np.string_(halo_subdata))
                        else:
                            if isinstance(halo_subdata, np.ndarray) and value_idx == 'n_e':
                                lengths = [arr.shape[0] for arr in halo_subdata]
                                halo_group.create_dataset('n_e_lengths', data=lengths)
                                max_length = max(lengths)
                                stacked_n_e = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan) for arr in halo_subdata])
                                halo_group.create_dataset(f'{value_idx}', data=stacked_n_e)
                            elif isinstance(halo_subdata, np.ndarray) and value_idx != 'n_e':
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
                            else:
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
    print(f'all_ne_DM has been saved to {all_DM_path}')
    sys.stdout.flush()

    # 保存 Halo_Information 数据
    with h5py.File(all_DM_Haloinfo_path, 'w') as h5file:
        if snapfile_idx in Halo_Information:
            mass_ranges_dict = Halo_Information[snapfile_idx]
            group = h5file.create_group(f'snapfile_{snapfile_idx}')
            for mass_range_key, halos_data in mass_ranges_dict.items():
                mass_range_group = group.create_group(mass_range_key)
                for halo_idx, halo_data in halos_data.items():
                    halo_group = mass_range_group.create_group(f'halo_{halo_idx}')
                    for value_idx, halo_subdata in halo_data.items():
                        if isinstance(halo_subdata, str):
                            halo_group.create_dataset(value_idx, data=np.string_(halo_subdata))
                        else:
                            if isinstance(halo_subdata, np.ndarray) and value_idx == 'n_e':
                                lengths = [arr.shape[0] for arr in halo_subdata]
                                halo_group.create_dataset('n_e_lengths', data=lengths)
                                max_length = max(lengths)
                                stacked_n_e = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan) for arr in halo_subdata])
                                halo_group.create_dataset(f'{value_idx}', data=stacked_n_e)
                            elif isinstance(halo_subdata, np.ndarray) and value_idx != 'n_e':
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
                            else:
                                halo_group.create_dataset(f'{value_idx}', data=halo_subdata)
    print(f'all_DM_Haloinfo has been saved to {all_DM_Haloinfo_path}')
    sys.stdout.flush()