# -*- coding: utf-8 -*-
"""
config/settings.py — 全局配置（论文方法 → 本项目参数的对照表）

所有可调节超参数集中在此文件。main.py 会加载这里的定义。
论文原始设置（大数据集 + PyTorch + GPU）对应的轻量级取值已标注 [paper] 与 [project]。

运行环境约束：Windows / 无 GPU / 无 CUDA / Python 3.10+ / 仅轻量依赖。
因此：PyTorch -> numpy 手写 LSTM；PyEMD -> 自实现 CEEMDAN；
      香港 60 站 3 年 5 分钟实测 -> 内置合成数据（可一键替换）。
"""

import os

# ----------------------------------------------------------------------------
# 0. 路径配置
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")          # 测试数据目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")   # 输出目录
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# 合成数据文件（main.py 首次运行会自动生成，之后可直接替换为你自己的数据）
SYNTH_POWER_FILE = os.path.join(DATA_DIR, "pv_power.csv")        # 各站点功率时序
SYNTH_META_FILE = os.path.join(DATA_DIR, "site_metadata.csv")    # 各站点装机容量

# ----------------------------------------------------------------------------
# 1. 数据规模（合成数据）— [paper] 60 站 / 3 年 / 5 分钟；[project] 轻量内置
# ----------------------------------------------------------------------------
N_SITES = 8                 # 站点数 [paper] 60
SAMPLE_INTERVAL_MIN = 5     # 采样间隔（分钟）[paper] 5
N_DAYS = 14                 # 合成数据天数 [paper] ~1095 天(3年)
SEEDS = list(range(2021, 2031))   # 10 次重复实验的随机种子 [paper] 2021-2030
SEED = SEEDS[0]                   # 演示运行取第一个种子；重复实验依次遍历 SEEDS

# ----------------------------------------------------------------------------
# 2. 预测任务窗口
# ----------------------------------------------------------------------------
INPUT_STEPS = 12            # 历史输入窗口 L（12 步 = 60 分钟）[paper] 12
HORIZON = 24                # 预测步长 H（未来 24 步 = 120 分钟）[paper] 48
                              # 轻量实现取 24 步以控制时长，语义一致（多步超前预测）
QUANTILES = [0.05, 0.5, 0.95]   # 分位数（点预测 + 90% 区间）[paper] 同

# ----------------------------------------------------------------------------
# 3. CEEMDAN 分解参数 — [paper] 白噪声幅值 0.2*std, 集成 100 次, 最大层数 12
# ----------------------------------------------------------------------------
CEEMDAN_NOISE_AMP = 0.2     # 白噪声幅值系数（相对原序列标准差）[paper] 0.2
CEEMDAN_ENSEMBLE = 3        # 集成次数 [paper] 100（轻量 -> 3，详见 README 简化说明）
CEEMDAN_MAX_IMF = 12        # 最大 IMF 层数 [paper] 12
CEEMDAN_MAX_SIFT = 12       # 单次筛分最大迭代次数（停止准则 SD<tol 或达上限）
CEEMDAN_SD_TOL = 0.2        # 筛分停止准则阈值
# 频率分组：IMF 阶次 -> 分量类别
# [paper] 1-5 高频, 6-9 中频, 10-12 低频, 残差单独成一类 -> 共 4 类
FREQ_GROUP_HIGH = list(range(1, 6))    # 高频波动
FREQ_GROUP_MID = list(range(6, 10))    # 中频变化
FREQ_GROUP_LOW = list(range(10, 13))   # 低频趋势
N_FREQ = 4                  # 合并后的频率分量数（高/中/低/残差）

# ----------------------------------------------------------------------------
# 4. 共享权重多通道 LSTM（轻量 numpy 实现，替代 PyTorch）
# ----------------------------------------------------------------------------
LSTM_HIDDEN = 20            # LSTM 隐藏维度 [paper] 64（轻量 -> 20，兼顾精度与 10 分钟预算）
LEARNING_RATE = 3e-3        # Adam 初始学习率 [paper] 1e-3（提高到 3e-3 以避免欠拟合）
BATCH_SIZE = 64             # 批量大小 [paper] 64
MAX_EPOCHS = 20             # 最大训练轮数 [paper] 200（轻量 -> 20，配合早停，保证 10 分钟内）
EARLY_STOP_PATIENCE = 5     # 验证集损失连续不下降即早停 [paper] 20
WEIGHT_DECAY = 1e-4         # L2 正则系数 [paper] 1e-4
LOSS_W_POINT = 0.1          # 点预测分支权重 λ [paper] 0.1
# 分位数回归头输出维度 = HORIZON * len(QUANTILES)

# ----------------------------------------------------------------------------
# 5. 数据集划分（按时间顺序）— [paper] 7 : 1.5 : 1.5
# ----------------------------------------------------------------------------
SPLIT_TRAIN = 0.7
SPLIT_VAL = 0.15
# 测试集占比 = 1 - TRAIN - VAL

# ----------------------------------------------------------------------------
# 6. 评估参数
# ----------------------------------------------------------------------------
VPP_CAPACITY_KWP = 2230.8   # VPP 总装机容量(kWp) [paper] 2230.8（仅用于 PINAW 归一化）
MAPE_LOW_POWER_RATIO = 0.05 # MAPE 剔除聚合功率低于总装机 5% 的时段 [paper] 同
NOMINAL_COVERAGE = 0.90     # 名义区间覆盖率 [paper] 90%（对应分位数 0.05/0.95）

# ----------------------------------------------------------------------------
# 7. 可视化与随机性
# ----------------------------------------------------------------------------
FIG_DPI = 120
SHOW_PLOTS = False          # 无界面环境设为 False，仅保存图片到 results/

# 汇总：把所有配置导出为 dict，方便保存进 summary.json 与日志
def as_dict():
    return {
        "N_SITES": N_SITES,
        "SAMPLE_INTERVAL_MIN": SAMPLE_INTERVAL_MIN,
        "N_DAYS": N_DAYS,
        "SEED": SEED,
        "INPUT_STEPS": INPUT_STEPS,
        "HORIZON": HORIZON,
        "QUANTILES": QUANTILES,
        "CEEMDAN_NOISE_AMP": CEEMDAN_NOISE_AMP,
        "CEEMDAN_ENSEMBLE": CEEMDAN_ENSEMBLE,
        "CEEMDAN_MAX_IMF": CEEMDAN_MAX_IMF,
        "N_FREQ": N_FREQ,
        "LSTM_HIDDEN": LSTM_HIDDEN,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_EPOCHS": MAX_EPOCHS,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LOSS_W_POINT": LOSS_W_POINT,
        "SPLIT_TRAIN": SPLIT_TRAIN,
        "SPLIT_VAL": SPLIT_VAL,
        "VPP_CAPACITY_KWP": VPP_CAPACITY_KWP,
        "MAPE_LOW_POWER_RATIO": MAPE_LOW_POWER_RATIO,
        "NOMINAL_COVERAGE": NOMINAL_COVERAGE,
    }
