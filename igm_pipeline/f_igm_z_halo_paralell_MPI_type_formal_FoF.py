#######
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import os
import joblib
from joblib import Parallel, delayed, parallel_backend
from scipy.spatial import cKDTree
import time
import sys
from tqdm import tqdm
import re
from tqdm_joblib import tqdm_joblib
import logging
from joblib import Parallel, delayed, Memory
import multiprocessing
from multiprocessing import Lock
from filelock import FileLock
import bisect
import ast
from mpi4py import MPI
tqdm_disabled = True
# 初始化共享资源
file_lock = Lock()
#os.environ['JOBLIB_TEMP_FOLDER'] = '/home/zhaozhang/tmp_joblib'
        
# Step 1: 宇宙常数
Omega_b = 0.04889
h = 0.6774
mp = 1.6726219e-24  # 质子质量 [g]

# Step 2: 恒星形成临界数密度（氢原子/cm^3）
n_H_crit = 0.1  # 可根据模拟中写入的值调整

# Step 3: 临界质量密度（单位 g/cm^3）
rho_crit_starforming = n_H_crit * mp

# Step 4: 宇宙临界密度（单位 g/cm^3）
rho_crit_cosmic = 1.8788e-26 * h**2 * 1e3 / 1e6  # 转为 g/cm^3

# Step 5: 平均重子密度
#rho_b_mean = Omega_b * rho_crit_cosmic

# Step 6: 计算阈值：rho_crit / rho_mean 
#rho_thresh = rho_crit_starforming / rho_b_mean
#log_rho_thresh = np.log10(rho_thresh)
#log_rho_thresh = np.log10(10**3.5)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("halo_processing.log", mode="w")  # 输出到日志文件
    ]
)
logger = logging.getLogger()
logger = logging.getLogger(__name__)  # 获取记录器

tqdm(disable=False, dynamic_ncols=True, ascii=True)
# 字体设置

font_label = {
    'family': 'serif',
    'size': 20
}
tick_size = 18
font_legend = {
    'family': 'serif',
    'size': 16
}

# 基础路径
#snapshots = [
#    "snapdir_015/snapshot_015", "groups_015/fof_subhalo_tab_015",
#    "snapdir_016/snapshot_016", "groups_016/fof_subhalo_tab_016",
#    "snapdir_017/snapshot_017", "groups_017/fof_subhalo_tab_017",
#    "snapdir_018/snapshot_018", "groups_018/fof_subhalo_tab_018",
#    "snapdir_019/snapshot_019", "groups_019/fof_subhalo_tab_019",
#    "snapdir_020/snapshot_020", "groups_020/fof_subhalo_tab_020"
#]



# 基础路径
base_path_noAGN = "/sqfs/work/hp240141/z6b340/Data/L100N1024_NoBH/output/"
base_path_fiducial = "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v1/output/"
storage_path = "/sqfs/work/hp240141/z6b340/results/f_IGM_results"
if not os.path.exists(storage_path):
    # 创建目录
    os.makedirs(storage_path)
kdtree_output_path = os.path.join(storage_path, "kdtree_storage")
if not os.path.exists(kdtree_output_path ):
    # 创建目录
    os.makedirs(kdtree_output_path)
# 函数：检测目录下的 snapdir 和 groups 子目录，并生成 snapshot 列表

snapshots = [
    #"snapdir_000/snapshot_000", "groups_000/fof_subhalo_tab_000",
    "snapdir_001/snapshot_001", "groups_001/fof_subhalo_tab_001",
    #"snapdir_017/snapshot_017", "groups_017/fof_subhalo_tab_017",
    #"snapdir_018/snapshot_018", "groups_018/fof_subhalo_tab_018",
    #"snapdir_019/snapshot_019", "groups_019/fof_subhalo_tab_019",
    #"snapdir_020/snapshot_020", "groups_020/fof_subhalo_tab_020"
]
snapshots_noAGN = snapshots
snapshots_fiducial = snapshots


def get_snapshots(base_path):
    snapshots = []
    # 获取目录中所有以 snapdir_ 和 groups_ 开头的子目录
    snapdirs = [d for d in os.listdir(base_path) if d.startswith("snapdir_")]
    groupdirs = [d for d in os.listdir(base_path) if d.startswith("groups_")]

    # 确保按编号排序，适配零填充格式
    snapdirs = sorted(snapdirs, key=lambda x: int(re.search(r'\d+', x).group()))
    groupdirs = sorted(groupdirs, key=lambda x: int(re.search(r'\d+', x).group()))

    # 确保 snapdir 和 group 子目录对应
    for snapdir, groupdir in zip(snapdirs, groupdirs):
        snap_num = "{:03d}".format(int(re.search(r'\d+', snapdir).group()))  # 补零到 3 位
        if snap_num == "{:03d}".format(int(re.search(r'\d+', groupdir).group())):  # 编号匹配
            snapshots.append("{}/snapshot_{}".format(snapdir, snap_num))
            snapshots.append("{}/fof_subhalo_tab_{}".format(groupdir, snap_num))
        else:
            print("Warning: Mismatch between {} and {}".format(snapdir, groupdir))

    return snapshots

# 获取两个路径下的 snapshots 列表
snapshots_noAGN = get_snapshots(base_path_noAGN)
snapshots_fiducial = get_snapshots(base_path_fiducial)


# 打印结果
print("Snapshots for noAGN:")
print(snapshots_noAGN)
print("\nSnapshots for fiducial:")
print(snapshots_fiducial)
# 定义缩放因子
alias_snap = {
    "rho": "PartType0/Density",
    "T": "PartType0/Temperature",
    "sphpos": "PartType0/Coordinates",
    "dmpos": "PartType1/Coordinates",
    "starpos": "PartType4/Coordinates",
    "bhpos": "PartType5/Coordinates",
    "smoothlen": "PartType0/SmoothingLength"
}
alias_fof = {
    "halomass": "Group/GroupMass",
    "halopos": "Group/GroupPos",
    "halonum": "Group/GroupNsubs",
    "halo0": "Group/GroupFirstSub",
    "halolen": "Group/GroupLen",
    "haloRV" : "Group/Group_R_Crit200",
    "haloMV": "Group/Group_M_Crit200",
    "Groupmasstype": "Group/GroupMassType"
    #"GroupID": "IDs"
}

alias_mass = {
    "sphmass": 0,  # 对应粒子类型 0（气体）的质量
    "dmmass": 1,   # 对应粒子类型 1（暗物质）的质量
    "starmass": 4,  # 对应粒子类型 4（恒星）的质量
    "bhmass": 5    # 对应粒子类型 5（黑洞）的质量
}

a_scaling = {
    "rho": "-3",
    "u": "0",
    "sphpos": "1",
    "dmpos": "1",
    "starpos": "1",
    "bhpos": "1",
    "smoothlen": "1",
    "T": "0",
    "sphmass": "0",
    "dmmass": "0",
    "starmass": "0",
    "bhmass": "0",
    "bhMdot": "0",
    "halopos": "1",
    "halomass": "0",
    "halolen": "0",
    "halonum": "0",
    "halo0": "0",
    "subhalopos": "1",
    "subhalomass": "0",
    "subhalBHMmax": "0",
    #"subhalolen" : "0",
    "haloRV" : "1",
    "haloMV": "0",
    "halobhM": "0",
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
    "T": "0",
    "sphmass": "-1",
    "dmmass": "-1",
    "starmass": "-1",
    "bhmass": "-1",
    "bhMdot": "0",
    "halopos": "-1",
    "halolen": "0",
    "halonum": "0",
    "halo0": "0",
    "subhalopos": "-1",
    "subhalomass": "0",
    "subhalBHMmax": "0",
    #"subhalolen" : "0",
    "haloRV" : "-1",
    "halomass": "-1",
    "haloMV": "-1",
    "halobhM": "-1",
    "Groupmasstype": "-1",
    "GroupID": "0",
    "sphID": "0",
    "dmID": "0",
    "starID": "0"

}


        
# 函数：加载分块文件的数据，应用标度因子

def get_scaling_factors(dataset_name, a_scaling, h_scaling):
    """
    获取指定数据集的 a_scaling 和 h_scaling 值。
    """
    if dataset_name in a_scaling and dataset_name in h_scaling:
        return float(a_scaling[dataset_name]), float(h_scaling[dataset_name])
    else:
        raise ValueError(f"Scaling factors for dataset '{dataset_name}' not found.")

        
def load_single_file(file, dataset_path, a_scaling_factor, h_scaling_factor, a, h):
    """
    加载单个文件的数据并应用标度因子。

    参数：
    - file: 单个文件路径
    - dataset_path: 数据集路径
    - a_scaling_factor: a 标度因子
    - h_scaling_factor: h 标度因子
    - a: 当前快照的尺度因子
    - h: 哈勃常数

    返回：
    - numpy 数组，包含加载后的数据，或 None 如果加载失败
    """
    try:
        with h5py.File(file, "r") as f:
            dataset = f[dataset_path][:]
            total_size = dataset.shape[0]
            dataset = dataset * (a ** a_scaling_factor) * (h ** h_scaling_factor)
            return dataset
    except KeyError:
        print(f"Dataset {dataset_path} not found in file {file}.")
    except OSError:
        print(f"Unable to open file {file}.")
    except Exception as e:
        print(f"Error loading {file}: {e}")
    return None

# 函数：加载分块文件的数据，应用标度因子
def load_snapshot_data(file_pattern, dataset_name, a, h):
    """
    无分块加载粒子数据，并动态应用标度因子。
    
    参数：
    - file_pattern: 文件路径模式，例如 "snapdir/snapshot"
    - dataset_name: 数据集名称，例如 "sphpos" 或 "halopos"
    - a: 当前快照的尺度因子
    - h: 哈勃常数
    
    返回：
    - numpy 数组，包含加载后的数据
    """
    # 获取数据集的路径和标度因子
    if dataset_name in alias_snap:
        dataset_path = alias_snap[dataset_name]
    elif dataset_name in alias_fof:
        dataset_path = alias_fof[dataset_name]
    elif dataset_name in alias_mass:
        dataset_path = f"PartType{alias_mass[dataset_name]}/Masses"
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found in aliases.")
    
    a_scaling_factor, h_scaling_factor = get_scaling_factors(dataset_name, a_scaling, h_scaling)

    # 加载数据文件
    file_list = sorted(glob.glob(f"{file_pattern}.*.hdf5"))
    if not file_list:
        file_list = [f"{file_pattern}.hdf5"]

    print(f"Loading files for pattern: {file_pattern}, dataset: {dataset_path}")

    data = []
    for file in file_list:
        try:
            with h5py.File(file, "r") as f:
                dataset = f[dataset_path][:]
                #print(f"Dataset {dataset_path} size: {dataset.shape}")
                #sys.stdout.flush()
                # 应用标度因子
                dataset = dataset * (a ** a_scaling_factor) * (h ** h_scaling_factor)
                data.append(dataset)
        except KeyError:
            print(f"Dataset {dataset_path} not found in file {file}. Skipping.")
            continue
        except OSError:
            print(f"Unable to open file {file}. Skipping.")
            continue

    if data:
        return np.concatenate(data, axis=0)
    else:
        raise ValueError(f"No valid files found for dataset: {dataset_name}")



def parallel_load_snapshot(file_pattern, dataset_name, a, h, num_jobs=-1):
    """
    并行加载粒子数据，并动态应用标度因子。

    参数：
    - file_pattern: 文件路径模式，例如 "snapdir/snapshot"
    - dataset_name: 数据集名称，例如 "sphpos" 或 "halopos"
    - a: 当前快照的尺度因子
    - h: 哈勃常数
    - num_jobs: 并行线程数量，默认值为 4

    返回：
    - numpy 数组，包含加载后的数据
    """
    # 获取数据集的路径和标度因子
    if dataset_name in alias_snap:
        dataset_path = alias_snap[dataset_name]
    elif dataset_name in alias_fof:
        dataset_path = alias_fof[dataset_name]
    elif dataset_name in alias_mass:
        dataset_path = f"PartType{alias_mass[dataset_name]}/Masses"
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found in aliases.")
    
    a_scaling_factor, h_scaling_factor = get_scaling_factors(dataset_name, a_scaling, h_scaling)

    # 获取文件列表
    file_list = sorted(glob.glob(f"{file_pattern}.*.hdf5"))
    if not file_list:
        file_list = [f"{file_pattern}.hdf5"]

    print(f"Parallel loading files for pattern: {file_pattern}, dataset: {dataset_path}")

    # 并行加载
    with tqdm_joblib(tqdm(desc=f"Processing loading {file_pattern}", total=len(file_list), disable=tqdm_disabled)) as progress_bar:
        results = Parallel(n_jobs=num_jobs)(

            delayed(load_single_file)(file, dataset_path, a_scaling_factor, h_scaling_factor, a, h)
            for file in file_list
        )

    # 过滤掉 None 的结果并合并
    results = [r for r in results if r is not None]
    return np.concatenate(results, axis=0) if results else np.array([], dtype=np.float32)

def calculate_R200_with_particles(halo_center, particle_data, rho_crit_Msun_Mpc3, r_max):
    """
    Calculate R200 and M200 for a given halo using an iterative method with minimal memory usage.
    """
    r_min = max(1e-4, 0.01 * r_max)  # Initial search range (Mpc)
    tolerance = 1e-1  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    def compute_enclosed_mass(radius):
        """
        Calculate the enclosed mass within a given radius.
        Iterates over particle types to reduce memory usage.
        """
        enclosed_mass = 0.0
        for ptype in particle_data:
            positions = particle_data[ptype]["positions"]
            masses = particle_data[ptype]["masses"]

            if positions.size > 0:  # Skip empty datasets
                distances = np.linalg.norm(positions - halo_center, axis=1)
                enclosed_mass += np.sum(masses[distances <= radius])
        return enclosed_mass

    previous_r_mid = None  # For tracking convergence issues

    # Iteratively search for R200
    for _ in range(max_iterations):
        r_mid = (r_min + r_max) / 2.0
        enclosed_mass = compute_enclosed_mass(r_mid)
        avg_density = enclosed_mass / ((4 / 3) * np.pi * r_mid**3)
        rho_200 = 200 * rho_crit_Msun_Mpc3

        # Debugging information
        print(f"Iteration {_}: r_mid={r_mid:.4f}, enclosed_mass={enclosed_mass:.4e}, "
              f"avg_density={avg_density:.4e}, rho_200={rho_200:.4e}")
        
        # Convergence criteria
        if np.abs(avg_density - rho_200) < tolerance:
            return r_mid, enclosed_mass
        elif avg_density > rho_200:
            r_min = r_mid  # Expand the searching area
        else:
            r_max = r_mid  # Shrink the searching area

        # Early exit for extreme cases (handle no progress)
        if r_mid == previous_r_mid:
            print("No progress in R200 convergence, stopping iteration.")
            return None, None
        previous_r_mid = r_mid

    raise ValueError("R200 did not converge within the maximum number of iterations.")

######################Group
def build_kdtrees_by_region_parallel_and_save_group(
    particle_data, box_size, region_size, halo_positions, output_path, chunk_size=1000000, n_jobs=-1, group_size=5
):
    """
    构建 KDTree，并将多个相邻区域分组存储到同一个 HDF5 文件中。

    参数：
    - particle_data: 包含粒子坐标和质量的数据字典
    - box_size: 模拟盒子的大小
    - region_size: 区域大小
    - halo_positions: Halo 的位置
    - output_path: HDF5 文件存储路径
    - chunk_size: 每次处理的粒子数量
    - n_jobs: 并行线程数
    - group_size: 每组包含的区域数量
    """ 
    
    def save_group_to_disk(group_data, group_key, output_path, region_counts):
        """
        将一个分组的数据写入磁盘。
        """
        # 清理之前生成的所有 HDF5 文件
        group_key_str = "_".join(map(str, group_key))  # 使用连字符连接 group_key
        file_path = f"{output_path}/kdtree_group_{group_key_str}.h5"
        with h5py.File(file_path, "a") as f:  # "a" 模式追加写入
            for region_key, data in group_data.items():
                region_key_str = "_".join(map(str, region_key))  # 使用连字符连接 region_key
                group = f.require_group(f"region_{region_key_str}")  # 确保分组存在
                for ptype, positions_list in data["positions"].items():
                    # 如果有对应的粒子类型数据
                    if ptype in data["masses"]:
                        masses_list = data["masses"][ptype]

                        # 将分块的数据合并
                        positions = np.concatenate(positions_list, axis=0)
                        masses = np.concatenate(masses_list, axis=0)

                        # 创建或更新数据集
                        if f"{ptype}/positions" in group:
                            #print(f"Dataset {ptype}/positions already exists in {region_key_str}. Overwriting.")
                            del group[f"{ptype}/positions"]
                        group.create_dataset(f"{ptype}/positions", data=positions, compression="gzip")

                        if f"{ptype}/masses" in group:
                            #print(f"Dataset {ptype}/masses already exists in {region_key_str}. Overwriting.")
                            del group[f"{ptype}/masses"]
                        group.create_dataset(f"{ptype}/masses", data=masses, compression="gzip")
            f.attrs["region_counts"] = region_counts
    

    def process_chunk(start, end, ptype, positions, masses):
        """
        对单个粒子块进行分区处理。
        """
        chunk_positions = np.mod(positions[start:end], box_size)
        chunk_masses = masses[start:end]
        region_ids = (chunk_positions // region_size).astype(int)
        unique_regions = np.unique(region_ids, axis=0)

        local_kdtrees = {}
        for region_id in unique_regions:
            mask = np.all(region_ids == region_id, axis=1)
            region_positions = chunk_positions[mask]
            region_masses = chunk_masses[mask]

            region_key = tuple(region_id)  # 保持 region_key 为 tuple 格式
            if region_key not in local_kdtrees:
                local_kdtrees[region_key] = {"positions": {}, "masses": {}}
            local_kdtrees[region_key]["positions"].setdefault(ptype, []).append(region_positions)
            local_kdtrees[region_key]["masses"].setdefault(ptype, []).append(region_masses)
        return local_kdtrees



    # 确定 Halo 的区域 ID
    region_counts = int(box_size // region_size)
    group_data = {}

    # 初始化分组数据

    for ptype in particle_data:
        positions = particle_data[ptype]["positions"]
        masses = particle_data[ptype]["masses"]
        num_particles = positions.shape[0]

        print(f"Processing particle type: {ptype}, Total particles: {num_particles}")

        with tqdm_joblib(tqdm(desc=f"Building KDTree for {ptype}", total=num_particles // chunk_size + 1, disable=tqdm_disabled)) as progress_bar:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_chunk)(
                    start,
                    min(start + chunk_size, num_particles),
                    ptype,
                    positions,
                    masses,
                )
                for start in range(0, num_particles, chunk_size)
            )

        # 合并处理结果
        for local_kdtrees in results:
            for region_key, data in local_kdtrees.items():
                group_key = tuple(np.floor(np.array(region_key) / group_size).astype(int))  # 使用 tuple 格式
                if group_key not in group_data:
                    group_data[group_key] = {}
                if region_key not in group_data[group_key]:
                    if "gas" not in data["positions"]:
                        print(f"Region {region_key} does not contain gas particles.")
                    group_data[group_key][region_key] = {"positions": {}, "masses": {}}

                for ptype in data["positions"]:
                    group_data[group_key][region_key]["positions"].setdefault(ptype, []).extend(data["positions"][ptype])
                    group_data[group_key][region_key]["masses"].setdefault(ptype, []).extend(data["masses"][ptype])

        
    # 将分组数据写入磁盘
    for group_key, group in group_data.items():
        save_group_to_disk(group, group_key, output_path, region_counts)

def get_neighbor_regions(halo_center, box_size, region_size, search_radius):
    """
    获取包含中心区域及所有实际与搜索半径覆盖到的区域编号。

    Parameters:
    - halo_center: Halo 中心坐标 (Mpc)。
    - box_size: 模拟盒子的大小 (Mpc)。
    - region_size: 区域的大小 (Mpc)。
    - search_radius: 搜索半径 (Mpc)。

    Returns:
    - neighbor_regions: 包含需要加载的区域编号的列表。
    """
    region_id = (halo_center // region_size).astype(int)
    neighbor_regions = []

    # 确定偏移范围
    min_offset = -np.ceil(search_radius / region_size).astype(int)
    max_offset = np.ceil(search_radius / region_size).astype(int)

    for dx in range(min_offset, max_offset + 1):
        for dy in range(min_offset, max_offset + 1):
            for dz in range(min_offset, max_offset + 1):
                offset = np.array([dx, dy, dz])
                # 计算当前偏移区域的边界
                region_min = (region_id + offset) * region_size
                region_max = region_min + region_size

                # 考虑周期边界条件
                region_min = np.mod(region_min, box_size)
                region_max = np.mod(region_max, box_size)

                # 检查球是否与区域重叠
                overlap = True
                for dim in range(3):
                    # 考虑盒子边界跨越的情况
                    if region_min[dim] < region_max[dim]:  # 正常区域
                        if not (region_min[dim] - search_radius <= halo_center[dim] <= region_max[dim] + search_radius):
                            overlap = False
                            break
                    else:  # 跨越边界区域
                        if not (halo_center[dim] <= region_max[dim] + search_radius or halo_center[dim] >= region_min[dim] - search_radius):
                            overlap = False
                            break

                if overlap:
                    region_key = tuple(np.mod(region_id + offset, box_size // region_size).astype(int))
                    neighbor_regions.append(f"region_{region_key[0]}_{region_key[1]}_{region_key[2]}")

    return neighbor_regions

#####################group
def load_data_from_regions_group(region_kdtree_path, neighbor_regions, group_size=5):
    """
    从分组存储的 HDF5 文件中加载指定的 KDTree 数据。

    Parameters:
    - region_kdtree_path: HDF5 文件路径。
    - neighbor_regions: 邻近区域列表。
    - group_size: 每组包含的区域数量，用于分组。

    Returns:
    - kdtrees: 包含加载的 KDTree 数据的字典。
    """
    # 计算需要加载的分组
    groups_to_load = set(
        tuple(np.floor(np.array([int(x) for x in region.split('_')[1:]]) / group_size).astype(int))
        for region in neighbor_regions
    )

    region_data = {}

    for group_key in groups_to_load:
        # 构造 HDF5 文件路径
        group_file = f"{region_kdtree_path}/kdtree_group_{'_'.join(map(str, group_key))}.h5"
        # 检查文件是否存在
        if not os.path.exists(group_file):
            print(f"Group file {group_file} not found.")
            continue

        # 打开 HDF5 文件并加载数据
        with h5py.File(group_file, "r") as f:
            for region_key in f.keys():                                                                                                                                                                                                                                                                                                                                                                                                                                      
                # 确保区域与邻近区域列表对齐
                if region_key in neighbor_regions:
                    if region_key not in region_data:
                        region_data[region_key] = {}

                    # 遍历粒子类型
                    for ptype in f[region_key]:
                        # 检查是否包含 positions 和 masses 数据
                        if "positions" in f[region_key][ptype] and "masses" in f[region_key][ptype]:
                            positions = f[f"{region_key}/{ptype}/positions"][:]
                            masses = f[f"{region_key}/{ptype}/masses"][:]

                            # 创建 KDTree 并存储
                            region_data[region_key][ptype] = {
                                "positions": positions,  # 加载 positions
                                "masses": masses
                            }
                        else:
                            region_data[region_key][ptype] = {
                                "positions": np.empty((0, 3)), # 空矩阵
                                "masses": np.empty((0,))
                            }
    return region_data 

def build_data_for_halos_with_figm(
    particle_data, 
    box_size, 
    halo_data, 
    output_path, 
    region_kdtree_path, 
    slice_indices,
    region_size,
    chunk_size=1000000, 
    n_jobs=-1, 
    C_search=5,
    group_size=5
):
    """
    为每个 Halo 构建一个 KDTree，并利用区域 KDTree 数据计算 f_IGM。
    
    参数：
    - particle_data: 包含粒子坐标和质量的数据字典。
    - box_size: 模拟盒子的大小 (Mpc)。
    - halo_data: 包含 Halo 的位置、R_200 和 ID 的字典。
        e.g., halo_data = {"positions": [...], "R200": [...], "ids": [...]}
    - output_path: HDF5 文件存储路径。
    - region_kdtree_path: 区域 KDTree 数据的存储路径。
    - slice_indices: Halo 划分切片索引。
    - region_size: 区域划分大小 (Mpc)。
    - group_size: 每组包含的区域数量。
    - chunk_size: 每次处理的粒子数量。
    - n_jobs: 并行线程数。
    """

    def save_halos_to_hdf5(halo_data_by_id, file_index, num_halos, output_path):
        """
        将 Halo 的粒子数据按粒子类型存储到指定的 HDF5 文件中，并保存 Halo 总数为属性。
        """
        file_name = os.path.join(output_path, f"halos_kdtree_{file_index}_{C_search}RV.h5")
        with h5py.File(file_name, "w") as f:
            for halo_id, halo_data in halo_data_by_id.items():
                halo_group = f.create_group(str(halo_id))
                for ptype, pdata in halo_data.items():
                    ptype_group = halo_group.create_group(ptype)
                    ptype_group.create_dataset("positions", data=pdata["positions"], compression="gzip")
                    ptype_group.create_dataset("masses", data=pdata["masses"], compression="gzip")
            f.attrs["Halo_num"] = num_halos

    def process_halo_tree(halo_id, halo_center, halo_r200, region_kdtree_path, group_size, region_size):
        """
        处理单个 Halo 的粒子数据，加载其覆盖区域的 KDTree 数据并构建 Halo 的整体 KDTree。
        """
        # Step 1: 计算 Halo 覆盖的邻近区域
        neighbor_regions = get_neighbor_regions(halo_center, box_size, region_size, C_search * halo_r200)
        # 检查生成的邻近区域
        if len(neighbor_regions) == 0:
            logging.warning(f"Warning: No neighbor regions found for halo {halo_id}.")
        # Step 2: 加载邻近区域的 KDTree 数据
        region_data = load_data_from_regions_group(region_kdtree_path, neighbor_regions, group_size)
        # 检查加载的区域数据
        if len(region_data) == 0:
             logging.warning(f"Warning: No neighbor regions found for halo {halo_id}.")
        # Step 3: 整合所有相关区域的粒子数据
        combined_positions = {}
        combined_masses = {}
        for region_key in region_data:
            for ptype in region_data[region_key]:
                positions = region_data[region_key][ptype]["positions"]  # 获取粒子位置
                masses = region_data[region_key][ptype]["masses"]        # 获取粒子质量
                if positions.size == 0:
                    logging.warning(f"Region {region_key}, type {ptype} has no positions to combine.")
                if ptype not in combined_positions:
                    combined_positions[ptype] = []
                    combined_masses[ptype] = []

                combined_positions[ptype].append(positions)
                combined_masses[ptype].append(masses)

        # 合并各粒子类型的所有位置和质量
        for ptype in combined_positions:
            try:
                combined_positions[ptype] = np.concatenate(combined_positions[ptype], axis=0)
                combined_masses[ptype] = np.concatenate(combined_masses[ptype], axis=0)
            except ValueError as e:
                print(f"Error combining data for type {ptype}: {e}")
                combined_positions[ptype] = np.empty((0, 3))
                combined_masses[ptype] = np.empty((0,))

        # Step 4: 为 Halo 构建整体 KDTree
        halo_particle_data = {}
        for ptype in combined_positions:
            # 创建 KDTree
            positions = combined_positions[ptype]
            masses = combined_masses[ptype]
            if len(positions) > 0:
                halo_kdtree = cKDTree(positions, boxsize=box_size)
                indices = halo_kdtree.query_ball_point(halo_center, C_search * halo_r200)
                if len(indices) > 0:
                    halo_particle_data[ptype] = {
                        "positions": positions[indices],
                        "masses": masses[indices],
                    }
        
        if halo_particle_data:
            return halo_id, halo_particle_data
        else:
            logging.warning(f"Warning: Halo {halo_id} has no particle data.")
            return halo_id, {}



    # 获取 Halo 数据
    halo_positions = halo_data["positions"]
    halo_radii = halo_data["R200"]
    halo_ids = halo_data["HaloIDs"]

    # 并行处理 Halo 数据
    with tqdm(desc="Processing Halos", total=len(halo_ids), disable=tqdm_disabled) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_halo_tree)(
                halo_id,
                halo_positions[halo_idx],
                halo_radii[halo_idx],
                region_kdtree_path,
                group_size,
                region_size
            )
            for halo_idx, halo_id in enumerate(halo_ids)
        )
        progress_bar.update(len(halo_ids))

    # 整理 Halo 数据
    halo_data_by_id = {halo_id: data for halo_id, data in results}

    # 保存到 HDF5 文件
    for i in range(len(slice_indices) - 1):
        start_idx, end_idx = slice_indices[i], slice_indices[i + 1]
        file_halos = {halo_ids[halo_idx]: halo_data_by_id[halo_ids[halo_idx]] for halo_idx in range(start_idx, end_idx)}
        save_halos_to_hdf5(file_halos, i, len(halo_ids), output_path)

    print("KDTree construction and saving completed.")



def load_kdtree_by_halo_id(output_path, halo_id, slice_indices, C_search = 5):
    """
    根据 halo_id 和划分的 slice_indices 加载对应的 KDTree 数据。

    参数：
    - output_path: HDF5 文件的存储路径。
    - halo_id: 当前 Halo 的 ID。
    - slice_indices: Halo 切片索引数组。

    返回：
    - kdtree_data: 当前 Halo 的 KDTree 数据。
    """
    # 确定 Halo 所属分区
    file_index = None
    for i in range(len(slice_indices) - 1):
        if slice_indices[i] <= halo_id <= slice_indices[i + 1]:
            file_index = i
            break

    if file_index is None:
        raise ValueError(f"Halo ID {halo_id} does not belong to any slice.")

    # 构造 HDF5 文件路径
    file_name = os.path.join(output_path, f"halos_kdtree_{file_index}_{C_search}RV.h5")

    # 检查文件是否存在
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"KDTree file {file_name} not found for Halo ID {halo_id}.")

    # 加载 HDF5 文件中的数据
    with h5py.File(file_name, "r") as f:
        if str(halo_id) not in f:
            raise KeyError(f"Halo ID {halo_id} not found in KDTree file {file_name}.")
        halo_data = f[str(halo_id)]
        kdtree_data = {
            ptype: {
                "positions": halo_data[ptype]["positions"][:],
                "masses": halo_data[ptype]["masses"][:],
            }
            for ptype in halo_data.keys()
        }

    return kdtree_data







#########Storage




###########inener
def get_particles_within_R200_inner(halo_center, R200, particle_data, box_size = 100):
    """
    提取 R200 范围内的粒子。
    """
    result = {}
    for ptype in particle_data:
        if "positions" not in particle_data[ptype] or "masses" not in particle_data[ptype]:
            logging.warning(f"Missing 'positions' or 'masses' for particle type '{ptype}'. Skipping.")
            continue

        positions = particle_data[ptype]["positions"]
        masses = particle_data[ptype]["masses"]

        if positions.ndim != 2 or positions.shape[1] != 3:
            logging.error(f"Invalid positions shape for particle type '{ptype}': {positions.shape}")
            continue

        if len(positions) == 0:
            logging.warning(f"No particles found for particle type '{ptype}'. Skipping.")
            continue

        # 计算 R200 范围内的粒子
        kdtree = cKDTree(positions, boxsize=box_size)
        indices = kdtree.query_ball_point(halo_center, R200)

        selected = {
            "positions": positions[indices],
            "masses": masses[indices],
        }

        # 如果是 gas 粒子类型，分类冷/热、稠密/稀疏
        if ptype == "gas" and "T" in pdata and "Density" in pdata:
            temperature = pdata["T"][indices]
            density_code = pdata["Density"][indices]
            density_cgs = density_code * to_cgs_factor
            rho_ratio = density_cgs / rho_mean

            cold_mask = temperature < 1e4
            hot_mask = ~cold_mask
            condensed_mask = rho_ratio >= rho_thresh
            IGM_mask = rho_ratio < rho_thresh

            selected["temperature"] = temperature
            selected["density_cgs"] = density_cgs

            # 分类质量
            selected["mass_cold_condensed"] = np.sum(masses[indices][cold_mask & condensed_mask])
            selected["mass_hot_condensed"] = np.sum(masses[indices][hot_mask & condensed_mask])
            selected["mass_cold_diff"] = np.sum(masses[indices][cold_mask & IGM_mask])
            selected["mass_hot_diff"] = np.sum(masses[indices][hot_mask & IGM_mask])
        
        result[ptype] = selected

    return result


# 函数：提取 R200 范围内的粒子
def get_particles_within_R200(halo_center, R200, particle_data, box_size = 100):
    """
    提取 R200 范围内的粒子。
    """
    result = {}
    for ptype in particle_data:
        if "positions" not in particle_data[ptype] or "masses" not in particle_data[ptype]:
            logging.warning(f"Missing 'positions' or 'masses' for particle type '{ptype}'. Skipping.")
            continue

        positions = particle_data[ptype]["positions"]
        masses = particle_data[ptype]["masses"]

        if positions.ndim != 2 or positions.shape[1] != 3:
            logging.error(f"Invalid positions shape for particle type '{ptype}': {positions.shape}")
            continue

        if len(positions) == 0:
            logging.warning(f"No particles found for particle type '{ptype}'. Skipping.")
            continue

        # 计算 R200 范围内的粒子
        kdtree = cKDTree(positions, boxsize=box_size)
        indices = kdtree.query_ball_point(halo_center, R200)

        result[ptype] = {
            "positions": positions[indices],
            "masses": masses[indices],
        }

    return result
    

def process_halo(
    halo_id,
    halo_center,
    halo_r200,
    particle_types,
    rho_crit_Msun_Mpc3,
    region_kdtree_path,
    num_valid_halos,
    slice_indices,
    C_search = 5,
    C_R200=1,
    if_Cal_R200=False,
    box_size = 100
):
    """
    处理单个 Halo，计算其内的气体质量。

    Parameters:
    - halo_id: 当前 Halo 的 ID。
    - halo_center: Halo 的中心坐标。
    - halo_r200: Halo 的 R200 半径。
    - particle_types: 包含的粒子类型 (如 'gas', 'dm', 等)。
    - region_kdtree_path: KDTree 文件的存储路径。
    - rho_crit_Msun_Mpc3: 临界密度。
    - slice_count: HDF5 文件的分块数量。
    - C_R200: 计算范围的倍数 (默认 R200 的 1 倍)。
    - if_Cal_R200: 是否重新计算 R200 和 M200 (默认 False)。

    Returns:
    - result: 包含 Halo ID 和气体质量的字典，或 None。
    """


    try:
        # 加载 KDTree 数据
        kdtree_data = load_kdtree_by_halo_id(region_kdtree_path, halo_id, slice_indices, C_search)
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Failed to load KDTree for halo_id {halo_id}: {e}")
        return None

    if not kdtree_data:
        logging.warning(f"No KDTree data available for halo_id {halo_id}. Skipping.")
        return None
    # 检查是否需要重新计算 R200 和 M200
    if if_Cal_R200:
        calc_R200, calc_M200 = calculate_R200_with_particles(
            halo_center, kdtree_data, rho_crit_Msun_Mpc3, C_R200 * halo_r200
        )
        if calc_R200 is None:
            logging.warning(f"Failed to calculate R200 for halo_id {halo_id}. Skipping.")
            return None
        halo_r200 = calc_R200

    # 提取 Halo 内的粒子
    particles_within_R200 = get_particles_within_R200(halo_center, C_R200 * halo_r200, kdtree_data, box_size)

    # 计算气体质量
    gas_mass_within_R200 = np.sum(particles_within_R200.get("gas", {}).get("masses", []))

    return {
        "halo_id": halo_id,
        "gas_mass_within_R200": gas_mass_within_R200,
    }



# 主函数：计算 f_IGM
def calculate_f_IGM_with_regions_and_kdtree(
    base_path,
    snapshots,
    output_file,
    region_kdtree_path,
    label="f",
    box_size=100.0,
    region_size=2.0,
    chunk_size=1000000,
    n_jobs=-1,
    C_search = 5,
    C_R200 = 1,
    filter_M200_low = 0,
    filter_M200_up = 15,
    if_Cal_R200 = False,
    slice_count = 64,
    group_size = 5,
    log_T_thre = 4, 
    log_rho_thre = 3.5
):
    """
    主函数：结合区域 KDTree 和并行处理计算 f_IGM。
    """
    redshifts = []
    f_IGM_values = []
    f_CGM_values = []
    f_Stellar_values = []
    f_hot_condensed_values = []
    f_cold_condensed_values = []
    f_cold_diff_values = []
    f_hot_diff_values =[]
    f_BH_values = []
    # 创建一个 HDF5 文件用于存储每个 Halo 的气体质量
    with h5py.File(output_file, "w") as hdf5_file:
        for i in range(36, 38, 2):
            snapfile = base_path + snapshots[i]
            groupfile = base_path + snapshots[i + 1]
            
            print(f"\nProcessing Snapshot: {snapshots[i]} and {snapshots[i + 1]}")

            # 提取当前的 scale factor a 和 Hubble 参数 h
            try:
                with h5py.File(f"{snapfile}.0.hdf5", "r") as f_snap:
                    a = f_snap["Header"].attrs["Time"]
                    z = 1 / a - 1
                    redshifts.append(z)
                    try:
                        h = f_snap["Header"].attrs["HubbleParam"]
                        Omega_0 = f_snap["Header"].attrs["Omega0"]
                        OmegaLambda = f_snap["Header"].attrs["OmegaLambda"]
                    except:
                        h = f_snap["Parameters"].attrs["HubbleParam"]
                        Omega_0 = f_snap["Parameters"].attrs["Omega0"]
                        OmegaLambda = f_snap["Parameters"].attrs["OmegaLambda"]
            except OSError:
                print(f"Error reading header for {snapfile}. Skipping.")
                continue
            box_label = "Box_" + snapshots[i].split("_")[-1] + label
            region_kdtree_path = os.path.join(kdtree_output_path, box_label)   
            print(f"region_kdtree_path: {region_kdtree_path}")
            if not os.path.exists(region_kdtree_path):
                os.makedirs(region_kdtree_path)
            # 计算临界密度
            G = 6.67430e-8  # 引力常数 (cm^3 g^-1 s^-2)
            H0_cgs = (h * 100) * 1e5 / 3.08567758e24  # H0 转为 cgs 单位 (s^-1)
            rho_crit = 3 * H0_cgs**2 / (8 * np.pi * G)
            rho_crit_Msun_Mpc3 = rho_crit * (3.08567758e24)**3 / 1.98847e43  # 转为 1e10 Msun/Mpc^3
            # 加载粒子数据并应用标度
            try:
                sphmass = parallel_load_snapshot(snapfile, "sphmass", a, h)
                sphpos = parallel_load_snapshot(snapfile, "sphpos", a, h)
                dmpos = parallel_load_snapshot(snapfile, "dmpos", a, h)
                dmmass = parallel_load_snapshot(snapfile, "dmmass", a, h)
                starpos = parallel_load_snapshot(snapfile, "starpos", a, h)
                starmass = parallel_load_snapshot(snapfile, "starmass", a, h)
                bhpos = parallel_load_snapshot(snapfile, "bhpos", a, h) if label == "f" else None
                bhmass = parallel_load_snapshot(snapfile, "bhmass", a, h) if label == "f" else None
                sphT = parallel_load_snapshot(snapfile, "T", a, h)
                sphrho = parallel_load_snapshot(snapfile, "rho", a, h)

                # 输出 sphpos 和 halopos 的最大值
                print(f"Max sphpos for snapshot {i/2}: {np.max(sphpos):.4f} Mpc")
                print(f"Min sphpos for snapshot {i/2}: {np.min(sphpos):.4f} Mpc")
            except (KeyError, ValueError) as e:
                print(f"Error loading snapshot: {e}")
                continue

            # 加载 Halo 数据并应用标度
            try:
                halopos = parallel_load_snapshot(groupfile, "halopos", a, h)
                haloRV = parallel_load_snapshot(groupfile, "haloRV", a, h)
                haloMV = parallel_load_snapshot(groupfile, "haloMV", a, h)
                halomass = parallel_load_snapshot(groupfile, "halomass", a, h)
                # 将 Halo 数据按照质量 haloMV 降序排序
                sorted_indices = np.argsort(-haloMV)  # 负号表示降序排列

                # 根据排序结果重新排列所有属性
                halopos = halopos[sorted_indices]
                haloRV = haloRV[sorted_indices]
                haloMV = haloMV[sorted_indices]
                halomass = halomass[sorted_indices]
                # 为 Halo 分配统一的 HaloID，按照排序后的顺序
                haloIDs = np.arange(len(halopos), dtype=int)

                # 输出 halopos 的最大值和最小值
                print(f"Max halopos for snapshot {i/2}: {np.max(halopos):.4f} Mpc")
                print(f"Min halopos for snapshot {i/2}: {np.min(halopos):.4f} Mpc")
            except (KeyError, ValueError) as e:
                print(f"Error loading FoF file: {e}")
                continue

            # 物理距离
            box_size_physical = box_size / h * a
            region_size_physical = region_size / h * a
            # 组装粒子数据
            particle_data = {
                "gas": {"positions": sphpos, "masses": sphmass, "T": sphT, "density": sphrho},
                "dm": {"positions": dmpos, "masses": dmmass},
                "star": {"positions": starpos, "masses": starmass},
            }
            if label == "f":
                particle_data["bh"] = {"positions": bhpos, "masses": bhmass}
            # 动态生成粒子种类列表
            particle_types = list(particle_data.keys())

            region_counts = int(box_size_physical // region_size_physical)
            print(f"region counts: {region_counts}")

            # 转换为 g/cm^3
            #density_cgs = density_code * 6.7699e-31

            # 使用前面算出的 rho_mean（g/cm^3）
            rho_mean_0 = 4.2149e-31
            rho_mean = rho_mean_0*(1+z)**3
            # 提取温度与质量
            temperature = particle_data["gas"]["T"]
            masses = particle_data["gas"]["masses"]
            density_cgs = particle_data["gas"]["density"] * 6.7699e-31 # 或 PartType0["Density"]
            # 计算 rho / rho_mean 及其对数
            rho_ratio = density_cgs / rho_mean
            # 定义密度临界值
            rho_ratio_thresh = 10 ** log_rho_thre
            # 温度划分
            cold_mask = sphT < 10**log_T_thre
            hot_mask = sphT >= 10**log_T_thre

            # 密度划分：临界密度为 log10(rho_b / rho_mean) = 5.6

            # 四个子类掩码
            cold_condensed = (temperature < 10**log_T_thre) & (rho_ratio >= rho_ratio_thresh)
            hot_condensed  = (temperature >= 10**log_T_thre) & (rho_ratio >= rho_ratio_thresh)
            cold_IGM       = (temperature < 10**log_T_thre) & (rho_ratio < rho_ratio_thresh)
            hot_IGM        = (temperature >= 10**log_T_thre) & (rho_ratio < rho_ratio_thresh)

            # 分别计算质量
            mass_cold_condensed = np.sum(masses[cold_condensed])
            mass_hot_condensed = np.sum(masses[hot_condensed])
            mass_cold_IGM = np.sum(masses[cold_IGM])
            mass_hot_IGM = np.sum(masses[hot_IGM])
            print(f"z={z:.2f}, mean T={np.mean(temperature):.2e}, median rho_ratio={np.median(rho_ratio):.2e}")
            print(f"Cold condensed gas mass: {mass_cold_condensed:.4e}")
            print(f"Hot condensed gas mass: {mass_hot_condensed:.4e}")
            print(f"Cold IGM gas mass: {mass_cold_IGM:.4e}")
            print(f"Hot IGM gas mass: {mass_hot_IGM:.4e}")
            #####清理kdtree文件
            def clear_output_files(output_path, files):
                """
                清理指定路径中的所有 HDF5 文件。
                """
                if not files:
                    print(f"No files found in {output_path} to delete.")
                else:
                    for file in files:
                        os.remove(file)
                        print(f"Removed old file: {file}")
            def check_and_clear_region_kdtree_files(region_kdtree_path, region_counts):
                """
                检查 HDF5 文件中的 Halo 数是否匹配，如果不匹配，则清理 HDF5 文件。
                """
                kdtree_files = glob.glob(f"{region_kdtree_path}/kdtree_*.h5")

                if not kdtree_files:
                    print("No KDTree files found. Rebuilding KDTree...")
                    return True

                for file in kdtree_files:
                    try:
                        with h5py.File(file, "r") as f:
                            if "region_counts" in f.attrs and f.attrs["region_counts"] == region_counts:
                                continue
                            else:
                                print(f"Region counts mismatch in {file}. Clearing all KDTree files...")
                                #clear_output_files(region_kdtree_path, kdtree_files)
                                return True
                    except Exception as e:
                        print(f"Error reading {file}: {e}. Clearing all KDTree files...")
                        clear_output_files(region_kdtree_path, kdtree_files)
                        return True

                print("KDTree files are up-to-date.")
                return False

            if check_and_clear_region_kdtree_files(region_kdtree_path, region_counts):
                build_kdtrees_by_region_parallel_and_save_group(
                particle_data, 
                box_size_physical, 
                region_size_physical, 
                halopos, 
                region_kdtree_path, 
                chunk_size=chunk_size, 
                n_jobs=n_jobs, 
                group_size=group_size
                )           
                print("Regional KDTree rebuilt and saved.")
            else:
                print("Regional KDTree files are up-to-date. Skipping KDTree build.")
    
            ##############检查是否需要清清理之前的kdtree文件
            def check_and_clear_halo_files(output_path, num_valid_halos, valid_halo_ids, required_keys=["positions", "masses"]):
                """
                检查所有 HDF5 文件中的 Halo 数是否匹配，以及是否包含完整的 Halo IDs 和粒子数据。
                如果不匹配或数据缺失，则清理 HDF5 文件。

                参数：
                - output_path: HDF5 文件的存储路径。
                - num_valid_halos: 预期的 Halo 总数。
                - valid_halo_ids: 预期的 Halo ID 列表。
                - required_keys: 每个粒子类型必须包含的键列表。
                """
                halo_kdtree_files = glob.glob(f"{output_path}/halos_kdtree_*.h5")

                if not halo_kdtree_files:
                    print("No KDTree files found. Rebuilding KDTree...")
                    return True

                # 用于存储所有切片文件中存在的 Halo IDs
                all_existing_halo_ids = set()

                for file in halo_kdtree_files:
                    try:
                        with h5py.File(file, "r") as f:
                            # 1. 检查 Halo 数量是否匹配
                            if "Halo_num" not in f.attrs or f.attrs["Halo_num"] != num_valid_halos:
                                print(f"Halo number mismatch in {file}. Expected {num_valid_halos}, found {f.attrs.get('Halo_num', 'None')}. Clearing...")
                                clear_output_files(output_path, halo_kdtree_files)
                                print("Clearing finished! Rebuilding...")
                                return True

                            # 2. 收集文件中的 Halo IDs
                            existing_halo_ids = set(map(int, f.keys()))
                            all_existing_halo_ids.update(existing_halo_ids)

                            # 3. 检查每个 Halo 是否包含所需的键
                            for halo_id in existing_halo_ids:
                                halo_group = f[str(halo_id)]
                                for ptype in halo_group.keys():  # 遍历每种粒子类型
                                    ptype_group = halo_group[ptype]
                                    if not all(key in ptype_group for key in required_keys):
                                        print(f"Missing required keys in {file}, halo {halo_id}, type {ptype}.")
                                        clear_output_files(output_path, halo_kdtree_files)
                                        print("Clearing finished! Rebuilding...")
                                        return True

                    except Exception as e:
                        print(f"Error reading {file}: {e}. Clearing all KDTree files...")
                        clear_output_files(output_path, halo_kdtree_files)
                        print("Clearing finished! Rebuilding...")
                        return True

                # 4. 检查是否所有预期的 Halo IDs 都存在
                missing_halo_ids = set(valid_halo_ids) - all_existing_halo_ids
                if missing_halo_ids:
                    if len(missing_halo_ids) > 100:
                        print(
                            f"Halo IDs mismatch across all files. {len(missing_halo_ids)} Halo IDs are missing. Clearing..."
                            f"Showing first 100 IDs: {list(missing_halo_ids)[:100]}"
                        )
                    else:
                        print(f"Halo IDs mismatch across all files. Missing Halo IDs: {missing_halo_ids}. Clearing...")
                    clear_output_files(output_path, halo_kdtree_files)
                    print("Clearing finished! Rebuilding...")
                    return True

                print("All KDTree files are consistent and up-to-date.")
                return False

            M_filter_low = 10**filter_M200_low/1e10
            M_filter_up = 10**filter_M200_up/1e10
            print(f"M_filter_low = {M_filter_low:.3e} (10^{filter_M200_low})")
            print(f"M_filter_up  = {M_filter_up:.3e} (10^{filter_M200_up})")
            # 计算有效 Halo 数
            print(f"M_200: {haloMV[:10]}")
            print(f"M_FoF: {halomass[:10]}")
            valid_halos = [
                (HaloID, center, R200, M200, halomass)
                for idx, (HaloID, center, R200, M200, halomass) in enumerate(zip(haloIDs, halopos, haloRV, haloMV, halomass))
                if halomass >= 0
                #if M_filter_low <= M200 <= M_filter_up
            ]


            # 构建一个目标 ID 列表
            target_halo_ids = set([
                HaloID for (HaloID, _, _, M200, halomass) in valid_halos 
                if M_filter_low <= halomass <= M_filter_up
            ])
            print(f"Number of halos in mass range: {len(target_halo_ids)}")
            num_valid_halos = len(valid_halos)
            print(f"Number of valid halos: {num_valid_halos}")
            for idx, (HaloID, halo_center, halo_r200, halo_m200) in enumerate(valid_halos[:5]):
                print(f"Halo {idx}: ID={HaloID}, Center={halo_center}, R200={halo_r200}, M200={halo_m200}, M_FoF={halomass}")

            # 构建 Halo 数据，ID 仍基于原始总数排序
            halo_data = {
                "positions": np.array([center for _, center, _, _ in valid_halos]),
                "R200": np.array([R200 for _, _, R200, _ in valid_halos]),
                "HaloIDs": np.array([HaloID for HaloID, _, _, _ in valid_halos]),  # 原始的 Halo ID 对应有效 Halo
            }
            

            if num_valid_halos // 100000 == 0:
                slice_indices = np.unique(np.linspace(0, num_valid_halos, slice_count + 1, dtype=int))
            else:
                linear_fraction = 0.00001
                num_linear_slices = int(slice_count * linear_fraction)

                # 修正起始和终止点
                linear_indices = np.linspace(0, int(num_valid_halos * linear_fraction), num_linear_slices, dtype=int)
                log_indices = np.floor(
                    np.logspace(
                        np.log10(max(int(num_valid_halos * linear_fraction), 1)),  # 确保 log10 的输入大于 0
                        np.log10(num_valid_halos),
                        slice_count - num_linear_slices + 1,
                    )
                ).astype(int)

                # 合并线性和对数切片索引
                slice_indices = np.unique(np.concatenate([linear_indices, log_indices]))

            # 确保起始为 0，结束为 num_halos
            if slice_indices[0] != 0:
                slice_indices = np.insert(slice_indices, 0, 0)
            if slice_indices[-1] != num_valid_halos:
                slice_indices = np.insert(slice_indices, -1, num_valid_halos)
            print(slice_indices)
            # 检查 KDTree 文件是否需要重建
            if check_and_clear_halo_files(region_kdtree_path, num_valid_halos, halo_data["HaloIDs"]):
                build_data_for_halos_with_figm(
                    particle_data=particle_data,
                    box_size=box_size_physical,
                    halo_data=halo_data,
                    output_path=region_kdtree_path,
                    region_kdtree_path=region_kdtree_path,
                    slice_indices=slice_indices,
                    region_size=region_size_physical,
                    n_jobs=n_jobs,
                    C_search=5,
                    group_size=group_size
                )
                
                print("Halo kdtree rebuilt and saved.")
            else:
                print("Halo KDTree files are up-to-date. Skipping KDTree build.")
            # Halo 数据的并行处理
            # 并行处理
            with tqdm_joblib(tqdm(desc="Processing Halos", total=num_valid_halos, disable = tqdm_disabled)) as progress_bar:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(process_halo)(
                        halo_id,
                        halo_center, 
                        true_R200, 
                        particle_types, 
                        rho_crit_Msun_Mpc3, 
                        region_kdtree_path,
                        num_valid_halos,
                        slice_indices = slice_indices,
                        C_search = C_search,
                        C_R200=C_R200, 
                        if_Cal_R200=if_Cal_R200,
                        box_size = box_size_physical
                    )
                    for halo_id, halo_center, true_R200, true_M200 in valid_halos
                )
            # 提取结果
            #halo_gas_masses = [result["gas_mass_within_R200"] for result in results if result is not None]
            halo_gas_masses = [
                result["gas_mass_within_R200"]
                for result in results 
                if result is not None and result["halo_id"] in target_halo_ids
            ]
            # 保存每个 Snapshot 的 Halo 气体质量
            dataset_name = f"snapshot_{snapshots[i]}_gas_mass_filter"
            hdf5_file.create_dataset(dataset_name, data=np.array(halo_gas_masses))

            # 计算 f_IGM
            total_gas_mass = np.sum(sphmass)
            total_stellar_mass = np.sum(starmass)
            total_bh_mass = np.sum(bhmass) if label == "f" else 0.0
            total_baryon_mass = total_gas_mass + total_stellar_mass + total_bh_mass
            total_halo_gas_mass = np.sum(halo_gas_masses)
            IGM_mass = total_baryon_mass - total_halo_gas_mass - total_stellar_mass - total_bh_mass
            
            f_CGM = total_halo_gas_mass / total_baryon_mass
            f_star = total_stellar_mass / total_baryon_mass
            f_bh = total_bh_mass / total_baryon_mass
            f_IGM = IGM_mass / total_baryon_mass
            f_cold_condensed = mass_cold_condensed / total_baryon_mass
            f_cold_diff = mass_cold_IGM / total_baryon_mass
            f_hot_condensed = mass_hot_condensed / total_baryon_mass
            f_hot_diff = mass_hot_IGM / total_baryon_mass

            f_IGM_values.append(f_IGM)
            f_Stellar_values.append(f_CGM)
            f_cold_condensed_values.append(f_cold_condensed)
            f_cold_diff_values.append(f_cold_diff)
            f_hot_condensed_values.append(f_hot_condensed)
            f_hot_diff_values.append(f_hot_diff)
            f_BH_values.append(f_bh)
            #print(type(z))  # 如果是 <class 'numpy.ndarray'>，就不能直接格式化
            #print(f"DEBUG: z = {z}, type = {type(z)}")
            #sys.stdout.flush()
            #print("Redshift z =", z)
            #print("  f_IGM       =", f_IGM)
            #print("  f_CGM       =", f_CGM)
            #print("  f_star      =", f_star)
            #print("  f_BH        =", f_bh)
            #print("  f_cold_dense=", f_cold_condensed)
            #print("  f_hot_dense =", f_hot_condensed)
            #print("  f_cold_IGM  =", f_cold_diff)
            #print("  f_hot_IGM   =", f_hot_diff)

            def safe_float(val):
                """将各种浮点类型安全转换为 Python float"""
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        return val.item()
                    else:
                        raise ValueError(f"Expected scalar array, got shape {val.shape}")
                elif isinstance(val, (np.float32, np.float64)):
                    return float(val)
                elif isinstance(val, float):
                    return val
                else:
                    raise TypeError(f"Cannot convert {type(val)} to float")

            # 用法举例（推荐你包成函数调用）：
            print(
                f"Redshift z={safe_float(z):.2f}\n"
                f"  f_IGM          = {safe_float(f_IGM)}\n"
                f"  f_CGM          = {safe_float(f_CGM)}\n"
                f"  f_star         = {safe_float(f_star)}\n"
                f"  f_BH           = {safe_float(f_bh):.2e}\n"
                f"  f_cold_dense   = {safe_float(f_cold_condensed)}\n"
                f"  f_cold_IGM     = {safe_float(f_cold_diff)}\n"
                f"  f_hot_dense    = {safe_float(f_hot_condensed)}\n"
                f"  f_hot_IGM      = {safe_float(f_hot_diff)}\n"
            )


            sys.stdout.flush()
            def save_f_igm_incrementally(z, f_IGM, C_R200, label_FM200_low, label_FM200_up, label_AGN, output_path):
                """
                增量保存或更新单个 (z, f_IGM) 数据到指定格式的 .txt 文件。

                Parameters:
                - z: 当前红移。
                - f_IGM: 当前的 f_IGM。
                - C_R200: R200 的倍数。
                - label_FM200: 质量过滤标识。
                - label_AGN: 数据集标识（如 'fiducial' 或 'noAGN'）。
                - output_path: 文件保存路径。
                """
                file_name = os.path.join(output_path, f"f_igm_z_{label_AGN}_{C_R200}_M{label_FM200_low}_{label_FM200_up}_temp_MPI.txt")
                temp_results = {}

                # 如果文件存在，读取已有的内容
                if os.path.exists(file_name):
                    with open(file_name, "r") as file:
                        for line in file:
                            if line.startswith("#"):
                                continue  # 跳过表头
                            existing_z, existing_f_IGM = map(float, line.split())
                            temp_results[existing_z] = existing_f_IGM

                # 更新或添加新的 (z, f_IGM) 值
                temp_results[z] = f_IGM

                # 按红移排序后写回文件
                with open(file_name, "w") as file:
                    file.write("# Redshift\tf_IGM\n")  # 写入表头
                    for sorted_z in sorted(temp_results, reverse=True):
                        file.write(f"{sorted_z:.6f}\t{temp_results[sorted_z]:.6f}\n")

                print(f"Saved/Updated f_IGM data for z={z:.6f} to {file_name}")


                
            #主进程管理数据的集中写入

            def save_f_components_incrementally(z, f_IGM, f_CGM, f_star, f_BH, f_cold_condensed, f_cold_diff, f_hot_condensed, f_hot_diff,
                                                C_R200, label_FM200_low, label_FM200_up, label_AGN, output_path):
                """
                增量保存或更新一个红移下的多个 gas phase 比例（f_IGM、f_CGM 等）到文件。
                """
                file_name = os.path.join(output_path, f"f_IGM_components_z_{label_AGN}_{C_R200}_M{label_FM200_low}_{label_FM200_up}_T{log_T_thre}_rho{log_rho_thre}_temp_MPI_FoF.txt")
                temp_results = {}
                print(f"[DEBUG] z={z:.6f}  TO BE WRITTEN:")
                print(f"  f_cold_dense = {f_cold_condensed}")
                print(f"  f_cold_diff  = {f_cold_diff}")
                print(f"  f_hot_dense  = {f_hot_condensed}")
                print(f"  f_hot_diff   = {f_hot_diff}")
                z_rounded = round(z, 6)
                # 如果文件存在，先读取旧数据
                if os.path.exists(file_name):
                    with open(file_name, "r") as file:
                        for line in file:
                            if line.startswith("#"):
                                continue
                            parts = line.strip().split()
                            if len(parts) >= 9:
                                existing_z = round(float(parts[0]), 6)
                                temp_results[existing_z] = list(map(float, parts[1:9]))

                # 检查是否覆盖
                if z_rounded in temp_results:
                    print(f"[INFO] Overwriting entry at z = {z_rounded:.6f}")
                else:
                    print(f"[INFO] Inserting new entry at z = {z_rounded:.6f}")

                # 插入或更新当前红移的数据
                temp_results[z_rounded] = [
                    f_IGM, f_CGM, f_star, f_BH,
                    f_cold_condensed, f_cold_diff,
                    f_hot_condensed, f_hot_diff
                ]

                # 按红移降序写回
                with open(file_name, "w") as file:
                    file.write("# Redshift\tf_IGM\tf_CGM\tf_star\tf_BH\tf_cold_dense\tf_cold_diff\tf_hot_dense\tf_hot_diff\n")
                    for sorted_z in sorted(temp_results.keys(), reverse=True):
                        values = temp_results[sorted_z]
                        file.write(f"{sorted_z:.6f}\t" + "\t".join(f"{v:.6e}" for v in values) + "\n")

                print(f"[SAVED] Updated multi-phase f_IGM data for z = {z_rounded:.6f} → {file_name}")
            #label_FM200 = str(filter_M200).replace(".", "p")
            label_FM200_low = str(filter_M200_low).replace(".", "p")
            label_FM200_up = str(filter_M200_up).replace(".", "p")
            ####仅保存f_IGM
            #save_f_igm_incrementally(z, f_IGM, C_R200, label_FM200_low, label_FM200_up label, kdtree_output_path)
            ####保存所有成分
            save_f_components_incrementally(
                z=z,
                f_IGM=f_IGM,
                f_CGM=f_CGM,
                f_star=f_star,
                f_BH=f_bh,
                f_cold_condensed=f_cold_condensed,
                f_cold_diff=f_cold_diff,
                f_hot_condensed=f_hot_condensed,
                f_hot_diff=f_hot_diff,
                C_R200=C_R200,
                label_FM200_low=label_FM200_low,
                label_FM200_up=label_FM200_up,
                label_AGN=label,
                output_path=kdtree_output_path
            )
            
#return redshifts, f_IGM_values
    return redshifts, f_IGM_values, f_CGM_values, f_Stellar_values, f_BH_values, f_cold_condensed_values, f_cold_diff_values, f_hot_condensed_values, f_hot_diff_values

def consolidate_f_igm_results(file_path, output_file):
    """
    从增量保存的文件中整理和校验所有 f_IGM 数据，生成最终的单个结果文件。

    Parameters:
    - file_path: 原始增量保存的文件路径。
    - output_file: 最终整理后的结果文件路径。
    """
    results = []
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue  # 跳过表头
            z, f_IGM = map(float, line.split())
            results.append((z, f_IGM))

    # 按红移排序并写入最终文件
    results.sort(key=lambda x: x[0])
    with open(output_file, "w") as file:
        file.write("# Redshift\tf_IGM\n")
        for z, f_IGM in results:
            file.write(f"{z:.6f}\t{f_IGM:.6f}\n")
    print(f"Consolidated results saved to {output_file}")
start_time = time.time()
# 获取当前机器的核心数
num_cores = multiprocessing.cpu_count()
print(f"core_num: {num_cores}")

default_CR200 = 1
default_FM200_low = 0
default_FM200_up = 15.5
default_logT = 4
default_logrho = 3.5
if len(sys.argv) == 1:
    # 无外部输入时使用默认值
    C_R200 = default_CR200
    filter_M200_low = default_FM200_low 
    filter_M200_up = default_FM200_up 
    log_T_thre = default_logT
    log_rho_thre = default_logrho
elif len(sys.argv) == 2:
    C_R200 = sys.argv[1]
    filter_M200_low = default_FM200_low
    filter_M200_up = default_FM200_up 
    log_T_thre = default_logT
    log_rho_thre = default_logrho
elif len(sys.argv) == 3:
    C_R200 = sys.argv[1]
    filter_M200_low = ast.literal_eval(sys.argv[2])
    filter_M200_up = default_FM200_up 
    log_T_thre = default_logT
    log_rho_thre = default_logrho
elif len(sys.argv) == 4:
    C_R200 = sys.argv[1]
    filter_M200_low = ast.literal_eval(sys.argv[2])
    filter_M200_up = ast.literal_eval(sys.argv[3])
    log_T_thre = default_logT
    log_rho_thre = default_logrho
elif len(sys.argv) == 5:
    C_R200 = sys.argv[1]
    filter_M200_low = ast.literal_eval(sys.argv[2])
    filter_M200_up = ast.literal_eval(sys.argv[3])
    log_T_thre = ast.literal_eval(sys.argv[4])
    log_rho_thre = default_logrho
elif len(sys.argv) == 6:
    C_R200 = sys.argv[1]
    filter_M200_low = ast.literal_eval(sys.argv[2])
    filter_M200_up = ast.literal_eval(sys.argv[3])
    log_T_thre = ast.literal_eval(sys.argv[4])
    log_rho_thre = ast.literal_eval(sys.argv[5])
else:
    print("  python3 f_igm_z.py 1")
    print("  python3 f_igm_z.py R/R_200, M_critical_low, M_critical_up, log_T_threshold, log_rho_threshold(rathip rho/rho_mean)")
    print("Example: mpirun -np 4 f_igm_z_halo_paralell_MPI_type_formal.py 1, 0, 15, 4, 3.5")
filter_M200_low = float(filter_M200_low)
filter_M200_up = float(filter_M200_up)
C_R200 = int(C_R200)
log_T_thre = float(log_T_thre)
log_rho_thre = float(log_rho_thre)

print(f"log_T_thre: {log_T_thre}")
print(f"log_rho_thre: {log_rho_thre}")
sys.stdout.flush()
box_size=100.0
region_size=5.0
C_search = 5
# 动态设置 n_jobs，例如使用总核心数减去1，保留一个核心用于系统任务
num_cores = max(1, num_cores - 1)
if_Cal_R200 = False
label_n = "n"
label_f = "f"
label_noAGN = "noBH"
label_fiducial = "fiducial_v1"
label_name = [label_noAGN, label_fiducial]
# 计算 noAGN 和 fiducial 的 f_IGM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    if os.path.exists(f"{storage_path}/halo_gas_{C_R200}_r200n.hdf5"):
        os.remove(f"{storage_path}/halo_gas_{C_R200}_r200n.hdf5")
    if os.path.exists(f"{storage_path}/halo_gas_{C_R200}_r200f.hdf5"):
        os.remove(f"{storage_path}/halo_gas_{C_R200}_r200f.hdf5")
comm.Barrier()

if size < 2:
    raise RuntimeError("This script requires at least 2 MPI processes.")
if rank == 0:
    # Task for noAGN
    logfile = f"logfile_figm_z_P_{C_R200}_{label_noAGN}_{filter_M200_low}_{filter_M200_up}_FoF.txt"
    sys.stdout = open(logfile, "w")
    redshifts_noBH, f_IGM_noBH, f_CGM_noBH, f_star_noBH, f_BH_noBH, f_cold_condensed_noBH, \
    f_cold_diff_noBH, f_hot_condensed_noBH, f_hot_diff_noBH = \
    calculate_f_IGM_with_regions_and_kdtree(
        base_path_noAGN, 
        snapshots_noAGN, 
        f"{storage_path}/halo_gas_{C_R200}_r200n_FoF.hdf5", 
        kdtree_output_path, 
        label = label_n,
        box_size=box_size,
        region_size=region_size,
        chunk_size=1000000,
        n_jobs=num_cores,
        C_search = 5,
        C_R200 = C_R200,
        filter_M200_low = filter_M200_low,
        filter_M200_up = filter_M200_up,
        if_Cal_R200 = False,
        slice_count = 64,
        group_size = 5,
        log_T_thre = log_T_thre, 
        log_rho_thre = log_rho_thre
    )
    comm.Barrier()


elif rank == 1:
    # Task for fiducial
    logfile = f"logfile_figm_z_P_{C_R200}_{label_fiducial}_{filter_M200_low}_{filter_M200_up}_FoF.txt"
    sys.stdout = open(logfile, "w")
    redshifts_fiducial, f_IGM_fiducial, f_CGM_fiducial, f_star_fiducial, f_BH_fiducial , f_cold_condensed_fiducial, \
    f_cold_diff_fiducial, f_hot_condensed_fiducial, f_hot_diff_fiducial=  \
    calculate_f_IGM_with_regions_and_kdtree(
        base_path_fiducial, 
        snapshots_fiducial,
        f"{storage_path}/halo_gas_{C_R200}_r200f_FoF.hdf5",
        kdtree_output_path, 
        label = label_f,
        box_size=box_size,
        region_size=region_size,
        chunk_size=1000000,
        n_jobs=num_cores,
        C_search = 5,
        C_R200 = C_R200,
        filter_M200_low = filter_M200_low,
        filter_M200_up = filter_M200_up,
        if_Cal_R200 = False,
        slice_count = 64,
        group_size = 5,
        log_T_thre = log_T_thre, 
        log_rho_thre = log_rho_thre
        )
    comm.Barrier()


end_time = time.time()
run_time = (end_time - start_time)/3600
print(f"code finised in {run_time} hours (Combied)")
comm.Barrier()
if rank == 0:
    # Consolidate results after all tasks are finished
    label_FM200_low = str(filter_M200_low).replace(".", "p")
    label_FM200_up = str(filter_M200_up).replace(".", "p")
    # Paths for temporary and final results
    temp_file_noAGN = f"{kdtree_output_path}/f_IGM_components_z_{label_n}_{C_R200}_M{label_FM200_low}_{label_FM200_up}_T{log_T_thre}_rho{log_rho_thre}_temp_MPI_FoF.txt"
    temp_file_fiducial = f"{kdtree_output_path}/f_IGM_components_z_{label_f}_{C_R200}_M{label_FM200_low}_{label_FM200_up}_T{log_T_thre}_rho{log_rho_thre}_temp_MPI_FoF.txt"
    final_file_noAGN = f"{storage_path}/f_IGM_components_final_{label_n}_{C_R200}_M{label_FM200_low}_{label_FM200_up}_T{log_T_thre}_rho{log_rho_thre}_MPI_FoF.txt"
    final_file_fiducial = f"{storage_path}/f_IGM_components_final_{label_f}_{C_R200}_M{label_FM200_low}_{label_FM200_up}_T{log_T_thre}_rho{log_rho_thre}_MPI_FoF.txt"

    # Copy temp file to final location (optional: sort if needed)
    import shutil
    shutil.copy(temp_file_noAGN, final_file_noAGN)
    shutil.copy(temp_file_fiducial, final_file_fiducial)

    print("All results consolidated and saved.")

    # Load consolidated results from file
    def load_multi_column_results(file_path):
        with open(file_path, "r") as file:
            lines = [line for line in file if not line.startswith("#")]
        data = np.loadtxt(lines)
        return data.T  # columns: z, f_IGM, f_CGM, f_cold_condensed, f_cold_diff, f_hot_condensed, f_hot_diff, f_BH

    # Load both data sets
    z_noAGN, f_IGM_noAGN, f_CGM_noAGN, f_star_noAGN , f_BH_noAGN , f_cold_condensed_noAGN, f_cold_diff_noAGN, f_hot_condensed_noAGN, f_hot_diff_noAGN= load_multi_column_results(final_file_noAGN)
    z_fiducial, f_IGM_fiducial, f_CGM_fiducial, f_star_fiducial, f_BH_fiducial, f_cold_condensed_fiducial, f_cold_diff_fiducial, f_hot_condensed_fiducial, f_hot_diff_fiducial = load_multi_column_results(final_file_fiducial)

    # Plot multiple components
    plt.figure(figsize=(12, 7))
    plt.plot(z_noAGN, f_IGM_noAGN, 'o-', label="IGM (noAGN)")
    plt.plot(z_noAGN, f_CGM_noAGN, 's--', label="CGM (noAGN)")
    plt.plot(z_fiducial, f_IGM_fiducial, 'o-', label="IGM (fiducial)")
    plt.plot(z_fiducial, f_CGM_fiducial, 's--', label="CGM (fiducial)")
    plt.xlabel("Redshift z", fontdict=font_label)
    plt.ylabel("Fraction", fontdict=font_label)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.gca().invert_xaxis()
    plt.grid()
    plt.legend(prop=font_legend)
    plt.tight_layout()
    plt.savefig(f"f_IGM_CGM_evolution_{C_R200}.png")
    print("Plot saved to f_IGM_CGM_evolution.png")

    # Optional: add a second figure for subcomponents
    plt.figure(figsize=(12, 7))
    plt.plot(z_fiducial, f_cold_condensed_fiducial, label="cold_dense (fiducial)")
    plt.plot(z_fiducial, f_hot_condensed_fiducial, label="hot_dense (fiducial)")
    plt.plot(z_fiducial, f_cold_diff_fiducial, label="cold_diffuse (fiducial)")
    plt.plot(z_fiducial, f_hot_diff_fiducial, label="hot_diffuse (fiducial)")
    plt.plot(z_fiducial, f_BH_fiducial, label="BH (fiducial)")
    plt.plot(z_fiducial, f_star_fiducial, label="Star (fiducial)")
    plt.xlabel("Redshift z", fontdict=font_label)
    plt.ylabel("Fraction", fontdict=font_label)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.gca().invert_xaxis()
    plt.grid()
    plt.legend(prop=font_legend)
    plt.tight_layout()
    plt.savefig("f_subcomponents_fiducial.png")
    print("Plot saved to f_subcomponents_fiducial.png")

    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 3600:.2f} hours")

    plt.figure(figsize=(12, 7))
    plt.plot(z_fiducial, f_BH_fiducial, label="BH (fiducial)")
    plt.plot(z_fiducial, f_star_fiducial, label="Star (fiducial)")
    plt.xlabel("Redshift z", fontdict=font_label)
    plt.ylabel("Fraction", fontdict=font_label)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.gca().invert_xaxis()
    plt.grid()
    plt.legend(prop=font_legend)
    plt.tight_layout()
    plt.savefig("f_BH_star_fiducial.png")
    print("Plot saved to f_BH_star_fiducial.png")

#sys.stdout.close()