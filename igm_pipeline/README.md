# IGM/CGM Evolution Pipeline

这份说明对应当前版本的：

- `f_igm_z_halo_paralell_MPI_type_formal_FoF.py`
- `run_IGM_z_evolve_MPI_type_formal_FoF.sh`

## Overview

这个流程主要用于：

- 计算不同红移下的 `f_IGM` 和 `f_CGM`
- 统计不同成分的质量分数，例如 `gas`、`star`、`bh`
- 区分冷致密、冷弥散、热致密、热弥散等气体相
- 对比 `noBH` 和 `fiducial` 两套模拟

当前这份脚本默认使用 FoF catalogue 中给出的 `R200/M200`。
代码内部保留了重新计算 `R200/M200` 的能力，通过 `if_Cal_R200` 控制，但当前命令行入口中固定为 `False`。

## MPI Layout

- `rank 0` 处理 `noBH`
- `rank 1` 处理 `fiducial`
- 至少需要 `2` 个 MPI rank

## Command Line

命令格式：

```bash
mpirun -np 2 python f_igm_z_halo_paralell_MPI_type_formal_FoF.py [C_R200] [logM_low] [logM_up] [logT] [logrho]
```

参数表：

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---:|---|
| `C_R200` | `int` | `1` | 统计半径，单位为 `R200` 倍数。例如 `1` 表示统计 `1 R200` 内的 halo 气体。 |
| `logM_low` | `float` | `0` | halo 质量下限，按 `log10(M200/Msun)` 理解。 |
| `logM_up` | `float` | `15.5` | halo 质量上限，按 `log10(M200/Msun)` 理解。 |
| `logT` | `float` | `4.0` | 温度阈值，用于气体相分类。 |
| `logrho` | `float` | `3.5` | 密度阈值，按 `rho / rho_mean` 的对数划分气体相。 |

最简单的例子：

```bash
mpirun -np 2 python f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 10 15.5 4.0 3.5
```

## Batch Script

对应 PBS 运行脚本：

```bash
qsub run_IGM_z_evolve_MPI_type_formal_FoF.sh
```

当前示例命令会运行：

```bash
mpirun ${NQSV_MPIOPTS} -genv I_MPI_DEBUG 5 -np 2 -genv I_MPI_PIN_DOMAIN=node \
  python3 $PBS_O_WORKDIR/f_igm_z_halo_paralell_MPI_type_formal_FoF.py 1 10 15.5 4.0 3.5
```

## Outputs

输出内容包括：

- 每个红移的 `f_IGM`
- 每个红移的 `f_CGM`
- `f_star`
- `f_BH`
- `f_cold_dense`
- `f_cold_diff`
- `f_hot_dense`
- `f_hot_diff`

输出文件会写入脚本中的 `storage_path` 和 `kdtree_output_path` 对应目录，文件名中包含：

- 数据集标签：`n` 或 `f`
- `C_R200`
- 质量区间
- `logT`
- `logrho`

## Notes

- 当前 FoF 版本入口默认读取 catalogue 中提供的 `R200`
- 如果后续需要开放“重新计算 R200”的用户接口，建议再补一个显式命令行参数
- 该脚本会同时生成中间 temp 结果文件和最终整理后的结果文件
