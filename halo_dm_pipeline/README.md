# Halo DM / CGM Analysis Pipeline

这套流程用于从大尺度模拟 snapshot / FoF catalogue 中提取 halo 局部物理量，并进一步开展：

- density profile 计算与比较
- DM vs impact parameter 分析
- 1D sightline 投影统计
- 2D observer-style map 处理与可视化

当前目录包含：

- `data_halo_storing_with_stellar_MPI_input.py`
- `density_profile_rv_morebins.py`
- `DM_Impact_factor_morebin.py`
- `Halo_DM_1D_map_joblib_withstellar.py`
- `Halo_DM_map_process_thread_P_joblib_observer_z.py`
- `run_halo_dm_pipeline.py`
- `config.yaml`

## Unified Runner

为了避免每次手动拼接命令，现在目录里增加了一个最小侵入的统一调度层：

- `run_halo_dm_pipeline.py`
- `config.yaml`

设计原则是：

- 不改你现有的科学主脚本
- 只在外层统一管理参数、步骤顺序、MPI 包装方式和工作目录

支持的步骤名：

- `prepare`
- `density_profile`
- `dm_impact`
- `map1d`
- `map2d`
- `all`

最常用的查看方式：

```bash
cd halo_dm_pipeline
python3 run_halo_dm_pipeline.py all --dry-run
```

这会打印将要执行的命令，但不会真正运行。

如果要实际运行某一个步骤，例如只做数据准备：

```bash
cd halo_dm_pipeline
python3 run_halo_dm_pipeline.py prepare
```

如果要按配置顺序运行全部已启用步骤：

```bash
cd halo_dm_pipeline
python3 run_halo_dm_pipeline.py all
```

## Config File

样例配置在：

`config.yaml`

它分成两部分：

- `pipeline`
  控制 Python 解释器、工作目录、MPI launcher、MPI 进程数等
- `steps`
  控制每个步骤是否启用、使用哪个脚本、是否用 MPI、参数如何传入

例如：

- `steps.prepare.args.snapshot_number`
- `steps.map1d.args.mass_range`
- `steps.map2d.args.snap_num`

如果你的环境支持 `PyYAML`，runner 会直接读取 `config.yaml`。
如果没有安装 `PyYAML`，可以：

- 安装 `PyYAML`
- 或者把配置改写成 JSON，再通过 `--config` 传入

## Workflow

推荐的整体流程是：

1. `data_halo_storing_with_stellar_MPI_input.py`
   从原始 snapshot 与 FoF catalogue 中抽取 halo 周围的局部数据，生成后续分析所需的 HDF5 输入。

2. `density_profile_rv_morebins.py`
   读取局部 halo 数据，计算多质量区间、多 box size 的 density profile，并输出曲线图。

3. `DM_Impact_factor_morebin.py`
   基于局部 halo 数据，沿不同 impact parameter 计算 DM 分布与统计量。

4. `Halo_DM_1D_map_joblib_withstellar.py`
   采用 1D sightline 方式，对每个 halo 在不同半径上统计 DM、stellar column density、metallicity 等物理量。

5. `Halo_DM_map_process_thread_P_joblib_observer_z.py`
   采用 2D mapping / observer-style 方式，对 halo 做二维投影，输出更丰富的 map 量，例如 DM、stellar mass、SFR、金属丰度、形成时间等。

## 1. Halo Local Data Extraction

脚本：

`data_halo_storing_with_stellar_MPI_input.py`

主要作用：

- 从 `L100N1024_Fiducial_v1` 与 `L100N1024_NoBH` 的 snapshot / FoF catalogue 中读取 halo 与粒子数据
- 提取 gas / DM / stellar 以及与恒星相关的额外字段
- 输出到按 snapshot 编号组织的目录中，供后续分析脚本调用

命令格式：

```bash
python data_halo_storing_with_stellar_MPI_input.py <snapshot_number>
```

示例：

```bash
python data_halo_storing_with_stellar_MPI_input.py 19
```

当前默认输出目录：

```bash
/sqfs/work/hp240141/z6b340/results/Halo_data_2D/snap_<snapshot_number>
```

主要提取字段包括：

- gas: `Density`, `Temperature`, `SmoothingLength`, `StarFormationRate`, `Metallicity`, `HI`, `HII`, `H2I`, `ElectronAbundance`, `CELibOxygen`, `CELibIron`
- DM: `Coordinates`, `Masses`
- stellar: `Coordinates`, `Masses`, `Metallicity`, `StellarFormationTime`
- FoF halo: `GroupMass`, `GroupPos`, `Group_R_Crit200`, `Group_M_Crit200`, `Group_R_Mean200`, `Group_M_Mean200`

## 2. Density Profile Analysis

脚本：

`density_profile_rv_morebins.py`

主要作用：

- 读取由 halo local data 生成的 HDF5 文件
- 按不同 halo 质量区间计算 density profile
- 比较不同 box size / 不同反馈模型下的 profile 差异
- 输出多组 profile 曲线和误差带

适合展示的示意图：

- density profile 结果图

建议后续加入：

- `assets/density_profile_example.png`

## 3. DM vs Impact Parameter

脚本：

`DM_Impact_factor_morebin.py`

主要作用：

- 基于 halo local data 计算不同 impact parameter 下的 DM
- 沿多条 LoS 穿过 halo / CGM 区域
- 统计不同质量区间与不同 box 的 DM 随 `b / R200` 的变化

命令行参数：

```bash
python DM_Impact_factor_morebin.py [a_] [h_] [Boxsize] [Boxname] [factor_RV]
```

支持的输入包括：

- scale factor `a_`
- 哈勃参数 `h_`
- 盒子尺度 `Boxsize`
- 数据集名称 `Boxname`
- 统计半径倍数 `factor_RV`

适合展示的图片：

- LoS / impact parameter 几何示意图
- DM vs `b/R200` 结果图

建议后续加入：

- `assets/impact_parameter_geometry.png`
- `assets/impact_parameter_dm_result.png`

## 4. 1D Sightline Projection

脚本：

`Halo_DM_1D_map_joblib_withstellar.py`

主要作用：

- 以 1D sightline 方式对每个 halo 做投影统计
- 在不同半径上统计：
  - `DM`
  - `stellar column density`
  - `metallicity`
- 支持 MPI + joblib 并行
- 支持共享 HDF5 写入或 per-rank merge 两种输出方式

命令格式：

```bash
python Halo_DM_1D_map_joblib_withstellar.py <mass_range> <halo_num> [--snap-num N] [--radial-bin-mode inner|uniform] [--agn-info f|n|both] [--h5-write-mode append|overwrite] [--mpi-io-mode mpio|merge]
```

代码中当前输出 HDF5 文件名形式为：

```bash
halos_data_<mass_bin_label>_200[_inner].h5
```

适合展示的图片：

- 1D sightline 方法示意图
- 单个 halo 的 DM distribution 示例

建议后续加入：

- `assets/oned_sightline_method.png`
- `assets/oned_dm_distribution_example.png`

## 5. 2D Observer-style Mapping

脚本：

`Halo_DM_map_process_thread_P_joblib_observer_z.py`

主要作用：

- 对 halo 做 2D observer-style mapping
- 计算二维图上的多种物理量，包括：
  - `DM`
  - `stellar mass`
  - `SFR`
  - `metallicity`
  - `SFT`
  - `n_HI` 等中性氢相关量

命令格式：

```bash
python Halo_DM_map_process_thread_P_joblib_observer_z.py [snap_num] [feedback_on]
```

当前默认参数：

- `snap_num = 20`
- `feedback_on = 1`

输出目录按 snapshot 编号组织，例如：

- `.../Halo_data_2D/snap_<snap_num>/DM_map_100f_observer/`
- `.../Halo_data_2D/snap_<snap_num>/DM_map_100n_observer/`

适合展示的图片：

- 2D mapping 方法示意图
- 多物理量二维投影结果图

建议后续加入：

- `assets/twod_mapping_method.png`
- `assets/twod_mapping_example.png`

## Relationships Between Scripts

这几个脚本之间的关系可以概括为：

- `data_halo_storing_with_stellar_MPI_input.py` 是上游数据准备脚本，负责从原始 simulation 数据中抽取 halo 局部数据。
- `density_profile_rv_morebins.py` 主要面向径向 profile 分析。
- `DM_Impact_factor_morebin.py` 主要面向 impact parameter 统计。
- `Halo_DM_1D_map_joblib_withstellar.py` 面向 1D sightline 投影。
- `Halo_DM_map_process_thread_P_joblib_observer_z.py` 面向 2D observer map。

也就是说：

- 第一份脚本负责“制备数据”
- 后面四份脚本负责“用不同方法分析同一批 halo 局部数据”

## Figures in README

你在本地说明中已经有以下几类示意图 / 结果图：

- density profile 示例图
- impact parameter 几何图
- DM vs impact parameter 结果图
- 1D-distribution 方法图
- 1D DM distribution 结果图
- 2D-mapping 方法图
- 2D map 结果图

目前我已经把 README 结构和图片落点安排好了，但我还不能直接把聊天附件里的图片文件写进仓库。

如果你之后把这些 PNG 文件放到本地目录，例如：

```bash
halo_dm_pipeline/assets/
```

我就可以继续帮你把 README 里的图片链接补成真正可显示的 GitHub 图片。
