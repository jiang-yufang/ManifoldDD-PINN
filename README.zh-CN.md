# 基于物理信息神经网络的高维流形上椭圆方程的区域分解方法（DDM-PINN）

**论文官方实现：**

> *Domain Decomposition Methods with Physics-Informed Neural Networks for Elliptic Equations on Manifolds*（流形上椭圆方程的物理信息神经网络区域分解方法）
>
> 作者：Yufang Jiang（蒋雨芳）, Lizhen Qin（秦丽珍）, Feng Wang（王锋）

[![arXiv](https://img.shields.io/badge/arXiv-2607.04285-b31b1b.svg)](https://arxiv.org/abs/2607.04285)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2607.04285-blue.svg)](https://doi.org/10.48550/arXiv.2607.04285)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[简体中文](README.zh-CN.md) | [English](README.md)**

论文（本仓库同时附有预印本 PDF：`DDM_PINN_arXiv.pdf`）针对紧致黎曼流形上的椭圆方程，提出了两种基于**物理信息神经网络（PINN）**的数值**区域分解方法（DDM）**。本仓库包含用于生成论文**第 4 节**全部数值实验结果的完整 PyTorch 实现。

---

## 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [仓库结构](#仓库结构)
- [环境要求](#环境要求)
- [安装](#安装)
- [复现实验](#复现实验)
  - [运行命令](#运行命令)
  - [实验配置](#实验配置)
  - [输出与结果](#输出与结果)
- [注意事项](#注意事项)
- [引用](#引用)
- [许可证](#许可证)

---

## 项目简介

考虑如下模型问题：

```
-Δu + b·u = f   在 d 维紧致黎曼流形 M 上，
```

其中 Δ 为 Laplace–Beltrami 算子。将流形分解为有限个相互重叠的子域，每个子域位于一个局部坐标卡中，并在每个子域上训练一个 PINN 近似局部解。各子域问题通过迭代过程耦合：**串行（Schwarz 型）** 或带**单位分解**的**并行**方式，与经典重叠型区域分解方法完全对应（见论文中的算法 3.1 与算法 3.2）。

方法在四个有界或无界流形上进行了数值验证，维数范围为 **5 到 10**：

| 流形 | 维数 | 边界 | 论文章节 |
|---|---|---|---|
| S⁵（ℝ⁶ 中的单位球面） | 5 | 无 | §4.1 |
| S¹⁰（ℝ¹¹ 中的单位球面） | 10 | 无 | §4.2 |
| B³ × S³（乘积流形） | 6 | 有（S² × S³） | §4.3 |
| CP³（复射影空间） | 6 | 无 | §4.4 |

## 主要特性

- **流形上两种 DDM 变体**，均基于 PyTorch 从零实现：
  - 串行（Schwarz 交替）区域分解方法 —— 论文算法 3.1；
  - 带单位分解的并行区域分解方法 —— 论文算法 3.2。
- **各流形局部坐标下的 Laplace–Beltrami 算子**：
  - S⁵ / S¹⁰ 的球面 Laplace 算子（球极坐标）；
  - B³ × S³ 的混合（欧氏 + 球面）Laplace 算子；
  - CP³ 复坐标下的 Laplace 算子（CP³ 不是任何欧氏空间的子流形）。
- **自动选择计算设备** —— 有 CUDA GPU 时自动使用 GPU，否则回退到 CPU。
- **可复现的统计结果** —— 每个主程序独立运行 **5 次**（随机初始化），并输出外层迭代第 0、5、10、15、20、100 步时相对 L² 误差的均值与标准差。
- **自动绘图与日志** —— 自动生成相对误差收敛曲线、‖uⁿ − u^∞‖ 曲线、逐次实验的 CSV 记录以及统计 CSV。
- **灵活的神经网络选项** —— 支持普通全连接 PINN 与残差（ResNet 风格）网络；深度、宽度、激活函数、优化器、学习率调度、采样比例与损失权重均可调节。

## 仓库结构

```
.
├── DDM_PINN_arXiv.pdf            # 配套论文（arXiv 预印本）
├── requirements.txt              # Python 依赖
├── README.md                     # README（英文）
├── README.zh-CN.md               # README（简体中文）
├── LICENSE                       # MIT 许可证
│
├── 4.1exp/                       # §4.1  S⁵ 上的数值实验
│   ├── HdDDM1.py                 #   主程序（串行 DDM，两个子域）
│   ├── pinn_models1.py           #   训练类、球面 Laplace 算子、损失函数
│   ├── basic_functions.py        #   网络结构、球/球体上随机采样
│   ├── test_functions.py         #   误差评估与绘图辅助函数
│   ├── data_processing.py        #   第 0、5、10、15、20、100 步的均值/标准差统计
│   ├── record_result.py          #   CSV 结果记录
│   ├── error_curve/              #   输出：相对 L² 误差曲线
│   ├── un-u_inf_curve/           #   输出：‖uⁿ − u^∞‖ 曲线
│   ├── result/                   #   输出：CSV 日志与统计结果
│   ├── models/                   #   输出：训练好的模型
│   ├── history_preds/            #   输出：保存的预测结果
│   └── history_error/            #   输出：保存的误差历史
│
├── 4.2exp/                       # §4.2  S¹⁰ 上的数值实验
│   └── ...                       #   与 4.1exp 布局一致（主程序：HdDDM1.py）
│
├── 4.3exp/                       # §4.3  B³ × S³ 上的数值实验
│   ├── HdDDM2.py                 #   主程序（主程序：HdDDM2.py）
│   └── ...                       #   与 4.1exp 布局一致
│
├── 4.4exp_serial/                # §4.4  CP³ 上的串行 DDM
│   ├── HdDDM3.py                 #   主程序（串行 DDM，四个子域）
│   └── ...                       #   与 4.1exp 布局一致
│
└── 4.4exp_parallel/              # §4.4  CP³ 上的并行 DDM（单位分解）
    ├── HdDDM4.py                 #   主程序（并行 DDM，四个子域）
    └── ...                       #   与 4.1exp 布局一致
```

> `error_curve/`、`un-u_inf_curve/`、`result/`、`models/`、`history_preds/`、`history_error/` 等子目录由主程序在运行时自动创建并写入结果。

## 环境要求

实验开发与运行环境如下：

| 依赖 | 版本 |
|---|---|
| Python | 3.10.14 |
| torch | ≥ 2.0.0（实际测试版本为 `2.1.0.dev20230621+cu117`，对应 CUDA 11.7 的 GPU 版本） |
| numpy | 1.26.4 |
| matplotlib | 3.9.2 |

全部 Python 依赖已写入 [`requirements.txt`](requirements.txt)。代码支持 CUDA GPU 加速并自动回退到 CPU，但**强烈建议使用 CUDA GPU** —— 完整实验的计算量很大（参见[注意事项](#注意事项)）。

## 安装

```bash
# 1.（推荐）创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3.（仅 GPU）若 pip 安装的不是 CUDA 版 PyTorch，可按需指定 CUDA 版本安装，
#    例如 CUDA 11.7：
#    pip install torch --index-url https://download.pytorch.org/whl/cu117
```

## 复现实验

每个主程序都会进行 **5 次**随机初始化的独立重复实验，并自动生成全部曲线图、日志与统计结果。请在各自实验文件夹内运行：

### 运行命令

| 论文章节 | 流形 | 文件夹 | 命令 |
|---|---|---|---|
| §4.1 | S⁵ | `4.1exp` | `cd 4.1exp && python HdDDM1.py` |
| §4.2 | S¹⁰ | `4.2exp` | `cd 4.2exp && python HdDDM1.py` |
| §4.3 | B³ × S³ | `4.3exp` | `cd 4.3exp && python HdDDM2.py` |
| §4.4（串行） | CP³ | `4.4exp_serial` | `cd 4.4exp_serial && python HdDDM3.py` |
| §4.4（并行） | CP³ | `4.4exp_parallel` | `cd 4.4exp_parallel && python HdDDM4.py` |

示例：

```bash
cd 4.1exp
python HdDDM1.py
```

### 实验配置

各主程序中硬编码的默认超参数如下表所示（与论文中的设置对应）：

| 章节 | 文件夹 | 流形 | b | R | 网络 | 采样点数 | 内部点比例 | 内部损失权重（→ λ_BC = 1 − 权重） | 重叠参数 s |
|---|---|---|---|---|---|---|---|---|---|
| 4.1 | `4.1exp` | S⁵ | 1.0 | 1.2 | 全连接 PINN，500 × 4 | 2000 | 0.6 | 0.2（→ 0.8） | — |
| 4.2 | `4.2exp` | S¹⁰ | 1.0 | 1.2 | 全连接 PINN，500 × 6 | 5000 | 0.9 | 0.1（→ 0.9） | — |
| 4.3 | `4.3exp` | B³ × S³ | 0.0 | 1.2 | Res-PINN，500 × 2 | 2000 | 0.1 | 0.005（→ 0.995） | — |
| 4.4 串行 | `4.4exp_serial` | CP³ | 4.0 | 1.2 | Res-PINN，500 × 2 | 2000 | 0.5 | 0.003（→ 0.997） | 1 |
| 4.4 并行 | `4.4exp_parallel` | CP³ | 4.0 | 1.2 | Res-PINN，500 × 2 | 2000 | 0.5 | 0.003（→ 0.997） | 0.1 |

所有实验的公共设置：

- 损失函数：`L = interior_weight · L_PDE + (1 − interior_weight) · L_BC`，论文中的 λ_BC 即边界项权重，λ_BC = 1 − interior_weight（见论文公式 (3.4)）。
- 内层迭代：每个外层步训练 5000 个 epoch，Adam 优化器，lr = 1e-3，指数学习率衰减（γ = 0.9999），每完成一个外层步 lr × 0.9。
- 激活函数：tanh。采样方式：球体/球面上均匀随机点（Monte Carlo）。
- 外层迭代：100 步。重复实验：5 次独立随机初始化。

问题设置（真解与源项）：

| 流形 | 真解 u | 源项 f |
|---|---|---|
| S⁵ / S¹⁰ | (1 − ‖x‖²)/(1 + ‖x‖²) | (n + b)(1 − ‖x‖²)/(1 + ‖x‖²) |
| B³ × S³ | sin(π yₚ) + (1 − ‖x‖²)/(1 + ‖x‖²) | (b + π²)sin(π yₚ) + (b + n′)(1 − ‖x‖²)/(1 + ‖x‖²) |
| CP³ | (a₀ + Σⱼ aⱼ‖wⱼ‖²)/(1 + ‖w‖²) | (4n + 4 + b)u − 4Σⱼ aⱼ，其中 a = (1, 2, −1, −2) |

### 输出与结果

每次运行主程序会在对应实验文件夹内生成以下文件：

| 输出 | 位置 | 说明 |
|---|---|---|
| 相对 L² 误差曲线 | `error_curve/` | `loss&error_curve_improved_all_<n>.png`（各子域）、`..._max_<n>.png`（子域最大值），以及 5 次实验的均值版本 |
| ‖uⁿ − u^∞‖ 曲线 | `un-u_inf_curve/` | `un-u_inf__relative_l2_Over_Step_all_<n>.png`、`..._max_<n>.png`，以及均值版本 |
| 逐次实验日志 | `result/hdddm*_test_results.csv` | 每次求解的具体参数、时间戳、求解耗时与相对 L² 误差 |
| 统计结果 | `result/output.csv` | 外层迭代第 0、5、10、15、20、100 步时各子域最大相对 L² 误差的均值与标准差（第 0 步即随机初始化的相对误差） |
| 模型存档 | `models/` | `models_<n>.pth`（全部子域网络的 state dict） |
| 预测结果 | `history_preds/` | `historys_<n>.pth`（各外层步在测试点上的预测值） |
| 误差历史 | `history_error/` | `d_er_<n>.pkl` / `di_er_<n>.pkl`（逐步误差历史，供 `data_processing.py` 统计使用） |

## 注意事项

- **计算开销。** 每次实验运行 100 个外层步 × 5000 个内层 epoch ×（2–4）个子域网络，并重复 5 次；单个完整实验在单张 GPU 上可能需要数小时。论文中的结果是在 GPU 集群上得到的。
- **`run_count.txt`。** 主程序会在当前工作目录维护一个运行计数器 `run_count.txt`，请务必在实验文件夹内运行（如 `cd 4.1exp`），以保证计数器与输出文件位于正确位置。
- **网络结构。** §4.1、§4.2 使用普通全连接 PINN，§4.3、§4.4 使用残差网络变体（`Res_PINN`，带 shortcut 连接的 ResNet 风格结构）；两种结构均可通过 `md` 选项（`'PINN'` / `'Res'`）选择。

## 引用

如果您的研究使用了本代码，请引用：

```bibtex
@article{Jiang2026PDE,
  author        = {Jiang, Yufang and Qin, Lizhen and Wang, Feng},
  title         = {{Domain Decomposition Methods with Physics-Informed Neural Networks for Elliptic Equations on Manifolds}},
  journal       = {arXiv preprint},
  year          = {2026},
  eprint        = {2607.04285},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2607.04285}
}
```

预印本 PDF 已随仓库发布：[`DDM_PINN_arXiv.pdf`](DDM_PINN_arXiv.pdf)。

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。
