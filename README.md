# 基于机器学习的致密双星并合引力波信号搜寻

本仓库对应太极实验室 2026 年度"大学生创新实践训练计划"题目 （一）基于机器学习的致密双星并合引力波信号搜寻。

## 1. 项目概述

本项目的任务是：给定一段 LIGO 探测器的时间序列数据，判断其中**是否包含致密双星旋近阶段的引力波信号**。这是一个二分类问题——区分有信号和纯噪声。**另外，所有的 notebook 都是已经运行过且保存完结果的，如要查看任务完成情况，可直接查看 notebook**

项目分三个阶段完成：

- **Phase 0 (本地跑通案例)**：安装环境，并成功运行github上下载的'baseline_sugon.ipynb'
- **阶段 1（Phase 1, Baseline）**：编写代码复现原库的 GWData-Bootcamp BBH 信号搜寻 baseline，使用 CNN 模型，验证复现后的数据管线可用，且效果一致。
- **阶段 2（Phase 2, BNS + ResNet）**：将搜寻目标从 BBH 迁移到 BNS，同时将 CNN 替换为 1D ResNet，绘制 BNS 搜寻任务的 ROC 曲线。


## 2. 文件结构

```
BBH_to_BNS/
├── README.md
├── deep_learning/
│   └── baseline/                   # 原始 Bootcamp baseline 复现目录
│       ├── main.py                 # 原版 CNN 模型定义  + 训练逻辑
│       ├── data_prep_bbh.py        # BBH 波形生成 + 噪声注入 + 白化
│       ├── utils.py                # 工具类
│       ├── baseline_sugon.ipynb    # 原版且跑通的 notebook
│       └── checkpoints_cnn/
│           ├── model_e52.pt
│           └── train_loss_history_cnn.npy
└── src/                            # 本项目主要实验与改进代码
    ├── train_gw_search.py          # 统一训练 + 评估脚本（阶段 1 / 阶段 2）
    ├── model_resnet_v2.py          # 1D ResNet-18
    ├── data_prep_bbh.py            # BBH 数据生成（含原版 bug 修复）
    ├── data_prep_bns.py            # BNS 数据生成（IMRPhenomD_NRTidalv2 + APR4_EPP）
    ├── main.py                     # baseline MyNet 入口
    ├── utils.py                    # 工具函数
    ├── requirements.txt
    ├── 01_baseline_bootcamp_run.ipynb  #复现且跑通的 BBH 的 notebook
    ├── 02_bns_resnet_search.ipynb      #改成 BNS + ResNet 的 notebook
    ├── checkpoints_baseline/       # 阶段 1 结果
    │   ├── best_model_e87.pt
    │   ├── last_model_e100.pt
    │   ├── loss_history_bbh_cnn.png
    │   ├── roc_bbh.png
    │   └── train_loss_history.npy
    └── checkpoints_bns_resnet/     # 阶段 2 结果
        ├── best_model_e95.pt
        ├── last_model_e100.pt
        ├── loss_history_bns_resnet.png
        ├── roc_bns.png
        └── train_loss_history.npy
```

## 3. 数据管线

### 3.1 数据从哪里来

样本由程序按需模拟生成。样本会先预生成到内存数据池，再在每个 epoch 从池中重采样。单个样本的生成过程如下：

```
参数采样 → 波形模板生成 → 探测器响应 → SNR 归一化 → 噪声叠加 → 白化 → 裁切
```

具体步骤：

1. **参数采样**：随机生成双星系统的物理参数——分量质量 (m₁, m₂)、倾角 (ι)、偏振角 (ψ)、天区位置 (α, δ) 等。BBH 质量范围 5–100 M☉（metric 分布），BNS 质量范围 1.0–2.0 M☉（均匀分布）。

2. **波形模板生成**：
   - **BBH**：代码通过 LALSimulation 接口生成 h₊(t) 和 h×(t)；底层所用的 BBH 近似模型为 `IMRPhenomD`。IMRPhenomD 是一个面向 nonprecessing/aligned-spin black-hole binaries 的频域 phenomenological inspiral-merger-ringdown 模型（Khan et al. 2016, Phys. Rev. D 93, 044007）。
   - **BNS**：使用 `IMRPhenomD_NRTidalv2`(Dietrich et al. 2019, Phys. Rev. D 100, 044003)，即在 BBH 基础模型上加入由数值相对论标定的潮汐修正。潮汐参数 Λ₁、Λ₂ 由 LALSimulation 的 TOV 相关接口数值计算；实现中使用的 EOS 名称为 `APR4_EPP`，它是 LALSuite 支持的命名 EOS。若追溯其 APR/APR4 背景，可分别参考 Akmal, Pandharipande & Ravenhall (1998, Phys. Rev. C 58, 1804) 与 Read et al. (2009, Phys. Rev. D 79, 124032)。具体计算链为：EOS → TOV 积分 → R(m), k₂(m) → Λ = (2/3) k₂ C⁻⁵；关于 k₂ 与潮汐形变的物理背景，可参考 Hinderer (2008, ApJ 677, 1216) 以及 Flanagan & Hinderer (2008, Phys. Rev. D 77, 021502)。

3. **探测器响应**：将 h₊ 和 h× 投影到具体探测器（H1/L1），施加天线函数 F₊、F× 和光程时延。

4. **SNR 归一化**：计算注入信号在给定 PSD 下的最优匹配滤波 SNR，并将其缩放到目标 SNR。训练脚本默认使用 `snr_train=20`，本文实验配置也采用这一设置。

5. **噪声叠加**：从 Advanced LIGO design sensitivity PSD 生成高斯有色噪声，将信号叠加到噪声上。每个波形模板配 25 个独立噪声实现（Nnoise=25），增加数据多样性。

6. **白化**：用同一个 PSD 对频域数据按 `1 / sqrt(PSD)` 加权，再变回时域，使有色噪声变成近似白噪声。这一步和匹配滤波中的噪声加权思想一致。

7. **裁切**：数据生成时实际观测长度是 2s（`T_obs = T × safe`，其中 `T=1s`、`safe=2`）。baseline 路径保留完整的 `(2, 16384)` 输入；ResNet 路径再截取中间 1s，得到 `(2, 8192)`。

### 3.2 数据集划分

本项目不采用预先落盘的固定训练集/验证集/测试集，而是使用预生成数据池 + 每 epoch 重采样的方式：

- **训练池**：训练脚本默认预生成 `pool_size=5000` 个样本，本文实验配置也使用 5000；其中一半是纯噪声（label=0），一半是信号+噪声（label=1）。每个 epoch 再从池中有放回随机抽取 `nsample=100` 个样本参与训练。
- **测试池**：训练脚本默认使用训练池大小的 1/5 作为测试池；在本文实验配置中对应为 1000 个样本，每个 epoch 从中抽样做评估。
- **ROC 评估集**：训练结束后，针对 SNR=5、10、15、20 分别重新独立生成 1000 个样本，用来计算 ROC 曲线和 AUC。

这种方式的好处是：训练阶段每个 epoch 看到的样本组合都不同，具有一定的数据增强效果，同时避免了在每个 epoch 的训练循环里反复执行波形生成等 CPU 密集型计算。代价是评估结果会随着重采样产生波动，因此前期 test loss 可能明显震荡。

### 3.3 输入维度

| 阶段 | 模型 | 输入形状 | 含义 |
|------|------|---------|------|
| 阶段 1（Phase 1） | MyNet (2D CNN) | (batch, 1, 2, 16384) | 1通道 × 2探测器 × 2s×8192Hz |
| 阶段 2（Phase 2） | ResNet1D | (batch, 2, 8192) | 2通道(探测器) × 1s×8192Hz |

## 4. 模型架构

### 4.1 阶段 1：MyNet（Baseline CNN）

GWData-Bootcamp 提供的原版模型，是一个 8 层 2D 卷积网络。所有卷积核在"探测器维度"上大小为 1，本质上是对每个探测器独立做 1D 卷积，最后在 flatten 层混合两个探测器的信息。

```
Conv2d(1→8, 1×32) → ELU → BN → MaxPool(1×8)
Conv2d(8→16, 1×16) → ELU → BN
Conv2d(16→16, 1×16) → ELU → BN
Conv2d(16→32, 1×16) → ELU → BN
Conv2d(32→64, 1×8) → ELU → BN → MaxPool(1×6)
Conv2d(64→64, 1×8) → ELU → BN
Conv2d(64→128, 1×4) → ELU → BN
Conv2d(128→128, 1×4) → ELU → BN → MaxPool(1×4)
Flatten → Linear(20224→64) → ELU → Dropout(0.5) → Linear(64→2)
```

参数量约 1.3M。输出 2 维 logits，配合 CrossEntropyLoss。

### 4.2 阶段 2：ResNet1D

标准 ResNet-18 结构适配 1D 时间序列输入。将两个探测器（H1, L1）作为 2 个输入通道，直接做 1D 卷积。

```
Conv1d(2→32, k=7, s=2) → BN → ReLU → MaxPool(k=3, s=2)
Layer1: 2 × ResBlock(32→64)
Layer2: 2 × ResBlock(64→128, stride=2)
Layer3: 2 × ResBlock(128→256, stride=2)
Layer4: 2 × ResBlock(256→512, stride=2)
AdaptiveAvgPool1d(1) → Dropout(0.5) → Linear(512→2)
```

其中每个 ResBlock 包含两层 Conv1d-BN-ReLU 和一条 shortcut 连接。当前默认 `ResNet1D` 的参数量约 3.84M。

### 4.3 训练配置

| 参数 | 本文实验配置 | 训练脚本默认值 |
|------|------|------|
| 优化器 | Adam | Adam |
| 初始学习率 | 0.003 | 0.003 |
| 学习率调度 | CosineAnnealingLR | CosineAnnealingLR |
| 损失函数 | CrossEntropyLoss | CrossEntropyLoss |
| Epochs | 100 | 100 |
| Batch size | 32 | 32 |
| 每 epoch 抽样数 | 100 | 100 |
| 训练池大小 | 5000 | 5000 |
| 训练 SNR | 20 | 20 |

## 5. 结果解读

### 5.1 Loss 曲线

以下结果解读以阶段 2（BNS + ResNet）为例：

- **前 50 个 epoch test loss 剧烈震荡**：原因主要来自预生成数据池上的重采样随机性——每个 epoch 看到的样本组合不同，某些 batch 碰巧抽到模型不擅长的样本组合，导致 test loss 出现孤立的峰。这不是模型发散。
- **train loss 始终很低**：每个 epoch 只抽取 100 个训练样本，模型很快就能拟合当前 batch。
- **50 epoch 之后趋于稳定**：CosineAnnealingLR 将学习率逐渐降到接近 0，模型停止大幅更新。
- **最终 test accuracy ≈ 1.0**：在训练 SNR=20 条件下，白化后的信号信噪比很高，分类任务本身比较容易。

### 5.2 ROC 曲线

**阶段 1（BBH + CNN）ROC**：

| SNR | AUC |
|-----|-----|
| 20 | 1.00 |
| 15 | 1.00 |
| 10 | 0.90 |
| 5 | 0.60 |

**阶段 2（BNS + ResNet）ROC**：

| SNR | AUC |
|-----|-----|
| 20 | 1.00 |
| 15 | 0.99 |
| 10 | 0.84 |
| 5 | 0.60 |

**分析**：

1. **AUC 整体随 SNR 升高而增大**。SNR=20 时两个阶段都达到 AUC≈1.0；SNR=5 时都接近随机（0.60），说明 1s 窗口内低 SNR 信号的 detection 接近极限。

2. **当前结果不能直接归因于“BNS 比 BBH 更难”**：在这组实验里，SNR=10 处 BNS+ResNet 的 AUC（0.84）低于 BBH+CNN（0.90）；但这里同时改变了信号类型和模型架构，因此不能把差异直接归因于波形物理。更稳妥的说法是：BNS 在 1s 窗口内通常只覆盖 inspiral 的一小段，频率扫过范围和 SNR 积累往往少于更重的 BBH；不过要单独验证这种物理差异对检测性能的影响，需要在控制模型不变的前提下做对照实验。

3. **SNR=5 时两者都接近随机**：AUC≈0.60，说明在如此低的信噪比下，单靠 1s 的数据几乎不可能可靠地检测信号。实际的 LIGO 搜寻使用匹配滤波对更长的数据段进行相干积分，有效 SNR 远高于单个 1s 窗口。

## 6. BBH → BNS 迁移

### 6.1 改了什么

| 项目 | BBH (原版) | BNS (修改后) |
|------|-----------|-------------|
| 分量质量 | 5–100 M☉ | 1.0–2.0 M☉ |
| 质量分布 | metric (按 Fisher 信息) | uniform |
| 波形近似 | IMRPhenomD | IMRPhenomD_NRTidalv2 |
| 潮汐参数 | 无 | 通过 APR4_EPP EOS + LALSimulation TOV 的数值计算 |
| 波形生成起始频率 | 从 12 Hz 起，按需下调以保证长度 | 从 par.fmin 起，按需下调至最低 5 Hz |
| `get_fmin` | point-particle PN chirp time | 在原始 point-particle chirp-time 表达式上加入 leading-order tidal PN 修正（参考 Vines, Flanagan & Hinderer 2011） |

### 6.2 BNS 波形生成的物理设定

**EOS 选择**：本实现固定使用 LALSuite 中的命名 EOS `APR4_EPP`。为避免混淆，这里不再将其直接写成“Akmal et al. (1998) 的 EOS”；更稳妥的说法是：其 APR/APR4 的物理背景可追溯到 Akmal, Pandharipande & Ravenhall (1998, Phys. Rev. C 58, 1804) 与 Read et al. (2009, Phys. Rev. D 79, 124032)，而 `APR4_EPP` 作为具体命名 EOS 由 LALSuite 直接支持，并在 GW170817 的 EOS 比较工作中作为独立模型出现。采用固定 EOS 而非随机 EOS 选择，是为了避免质量分布与 EOS 选择之间的人为耦合。

**Λ(m) 计算**：调用 LALSimulation 的 `SimNeutronStarEOSByName` → `CreateSimNeutronStarFamily` → `SimNeutronStarRadius` + `SimNeutronStarLoveNumberK2`，对给定质量的中子星进行 TOV 数值计算，得到 R(m) 与 k₂(m)，再据 Λ = (2/3) k₂ C⁻⁵ 计算无量纲潮汐形变参数。不使用拟合公式；同一双星系统的两颗星使用同一个 EOS。

**质量范围**：[1.0, 2.0] M☉。在本实现所用的 APR4_EPP 下，非旋最大质量约为 2.16 M☉，因此上限 2.0 M☉ 仍处于稳定 TOV 解范围内。下限 1.0 M☉ 则覆盖了常见 BNS 质量区间的低端。

### 6.3 为什么没有用 TaylorF2

最初尝试替换为 TaylorF2。但 TaylorF2 是频域波形（`SimInspiralChooseFDWaveform`），经 IFFT 后时域信号的长度、归一化方式和 `ref_idx` 与下游注入管线不兼容，导致信号实质上未被注入——loss 收敛到 ln(2)，accuracy 锁死在 50%。最终改用 IMRPhenomD_NRTidalv2，保持时域接口不变的同时加入了潮汐效应。

## 7. 对原版代码的修复

在复现和迁移过程中，检查了 GWData-Bootcamp 原版 `data_prep_bbh.py`（credit: Dr. Hunter Gabbard），发现其中存在两处需要修正的问题，并在本项目中进行了对应修复。

### 7.1 多探测器索引问题

原版代码在 `gen_bbh()` 的探测器循环中先设定 `j = 0`，但循环内部未对 `j` 进行递增，导致多探测器情形下始终写入第 0 个通道。对于双探测器配置（如 H1/L1），其直接后果是：后一个探测器的数据会覆盖前一个探测器写入到 `ts[0]` 的内容，而 `ts[1]` 保持初始化时的零值。该问题会影响多探测器训练样本的正确生成。

修复方法：将原先的手动索引方式改为 `enumerate`，确保每个探测器的数据写入各自对应的数组通道。

### 7.2 倾角参数语义错误

原版代码在 `gen_par()` 中先生成真实倾角 `iota`，但在存入参数结构体时使用的是 `np.cos(iota)`，随后在 `gen_bbh()` 中又将 `par.iota` 直接传入 `SimInspiralChooseTDWaveform`。然而，LALSimulation 接口在此处要求的参数是倾角 $\iota$ 本身（以弧度计），而不是 $\cos\iota$。因此，原版代码虽然可以运行，但这里传入的物理参数语义并不正确，会使生成波形所对应的倾角定义发生偏移。

修复方法：在参数结构体中直接保存真实倾角 `iota`，并在后续波形生成时按接口要求传入该角度本身。

### 7.3 说明

需要说明的是，上述两处问题可以直接由公开源码与 LALSimulation 接口定义核对确认；但从公开可查的代码版本看，这两种写法并不只出现在 GWData-Bootcamp 中，在 Hunter Gabbard 公开的其他相关仓库代码中也可见到类似实现。因此，本文将其表述为“原版代码中的可确认问题及修复”，而不进一步将其简单归因于某一次特定迁移过程。

## 8. 运行方式

### 环境准备

```bash
cd "src"
pip install -r requirements.txt
```

### 阶段 1：BBH baseline

```bash
python train_gw_search.py --phase baseline --epochs 100 --pool_size 5000
```

### 阶段 2：BNS + ResNet

```bash
python train_gw_search.py --phase bns_resnet --epochs 100 --pool_size 5000
```

### 完整参数列表

```text
--phase         baseline / bns_resnet
--epochs        训练轮数 (默认 100)
--lr            学习率 (默认 0.003)
--batch_size    批大小 (默认 32)
--nsample       每 epoch 抽取样本数 (默认 100)
--pool_size     预生成数据池大小 (默认 5000)
--snr_train     训练 SNR (默认 20.0)
--model         ResNet 版本: full / small (默认 full)
--device        cuda / cpu / auto (默认 auto)
--outdir        输出目录 (默认 ./checkpoints_{phase}/)
```

## 9. 产出清单

以下文件路径均相对于 `src/` 目录：

| 产出 | 文件 | 说明 |
|------|------|------|
| BBH baseline ROC | `checkpoints_baseline/roc_bbh.png` | 4 个 SNR 下的 ROC 曲线 |
| BBH baseline loss | `checkpoints_baseline/loss_history_bbh_cnn.png` | 训练/测试 loss 和 accuracy |
| BBH baseline 模型 | `checkpoints_baseline/best_model_e87.pt` | 最佳 test loss 对应的权重 |
| BNS+ResNet ROC | `checkpoints_bns_resnet/roc_bns.png` | 4 个 SNR 下的 ROC 曲线 |
| BNS+ResNet loss | `checkpoints_bns_resnet/loss_history_bns_resnet.png` | 训练/测试 loss 和 accuracy |
| BNS+ResNet 模型 | `checkpoints_bns_resnet/best_model_e95.pt` | 最佳 test loss 对应的权重 |