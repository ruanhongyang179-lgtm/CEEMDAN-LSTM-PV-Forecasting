# -*- coding: utf-8 -*-
"""
data_generator.py — 合成分布式光伏（PV）测试数据生成器

论文真实数据为「香港科大 60 个屋顶光伏站点、3 年、5 分钟采样」的实测数据集
（Dryad, DOI:10.5061/dryad.m37pvmd99），体积庞大且需联网下载。
本项目以内置合成数据等价替代，并刻意复现论文方法依赖的关键物理现象：

  1. 昼夜清晰天空功率曲线（低频趋势）—— 每日 0 功率的夜间 + 白天钟形曲线；
  2. 天气过程引起的中频变化（天与天之间的整体辐照度差异）；
  3. 云团移动造成的站点间异步波动 + 传播时延（高频）—— 相邻站点功率跌落
     存在 1~3 个采样步的滞后，构成论文强调的“跨站点时延耦合”；
  4. 逆变器测量噪声（高频白噪声）。

生成结果（均为纯文本 CSV，无第三方表格库依赖）：
  data/pv_power.csv        列 = [timestamp, site_1, ..., site_N]，单位 kW
  data/site_metadata.csv   列 = [site_id, capacity_kwp, x_position]

用户可直接用自己同格式的 CSV 替换上述文件（见 README「如何替换数据」）。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import datetime as dt
import numpy as np

import config.settings as cfg

# 仅依赖标准库与 numpy；不使用 pandas，保持轻量。


def _clear_sky(hour, sunrise=5.5, sunset=18.5):
    """归一化清晰天空辐照形状 [0,1]：夜间为 0，白天钟形。"""
    if hour <= sunrise or hour >= sunset:
        return 0.0
    x = (hour - sunrise) / (sunset - sunrise)  # 0..1
    return float(np.sin(np.pi * x) ** 1.5)


def generate_pv_data(seed=cfg.SEED, n_sites=cfg.N_SITES, n_days=cfg.N_DAYS,
                     interval_min=cfg.SAMPLE_INTERVAL_MIN,
                     return_meta=False):
    """生成合成多站点 PV 功率时序。

    返回
    ----
    timestamps : list[datetime], 长度 T
    power      : np.ndarray (T, n_sites) 单位 kW
    capacities : np.ndarray (n_sites,) 单位 kWp
    """
    rng = np.random.default_rng(seed)
    steps_per_day = int(24 * 60 / interval_min)           # 5 分钟 -> 288
    T = steps_per_day * n_days
    start = dt.datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [start + dt.timedelta(minutes=interval_min * i) for i in range(T)]

    # 站点地理位置（沿一条直线排布，便于云团有序传播）
    x_pos = np.arange(n_sites, dtype=float)
    # 各站点装机容量(kWp)：在 [30, 60] 间离散取值，制造站点容量差异
    capacities = rng.integers(30, 61, size=n_sites).astype(float)

    hours = np.array([(ts.hour + ts.minute / 60.0) for ts in timestamps])
    # (1) 清晰天空日曲线（所有站点共享形状）
    sky = np.array([_clear_sky(h) for h in hours])         # (T,)
    # (2) 天尺度天气因子（中频）：每天一个缓慢随机游走 [0.55, 1.0]
    day_idx = (np.arange(T) // steps_per_day)
    n_unique_days = int(day_idx.max()) + 1
    daily_factor = np.empty(n_unique_days)
    val = 0.8
    for d in range(n_unique_days):
        val += rng.uniform(-0.08, 0.08)
        val = min(max(val, 0.55), 1.0)
        daily_factor[d] = val
    weather = daily_factor[day_idx]                         # (T,)

    # (3) 云团场：若干云团沿 x 方向以速度 v 移动，产生站点间时延耦合
    occ = np.ones((T, n_sites))                            # 遮挡乘子，1 表示无遮挡
    n_blobs = max(8, n_days * 3)
    t_arr = np.arange(T)
    for _ in range(n_blobs):
        c0 = rng.uniform(-2, n_sites + 2)                  # 初始中心位置
        v = rng.uniform(0.18, 0.45)                        # 速度(站点/步) -> 相邻站延迟约 2~5 步
        t0 = rng.integers(0, max(1, T - steps_per_day // 2))
        dur = rng.integers(steps_per_day // 6, steps_per_day // 2)
        intensity = rng.uniform(0.45, 0.9)                 # 最大遮挡强度
        sigma = rng.uniform(0.6, 1.4)                      # 空间尺度
        center = c0 + v * (t_arr - t0)
        g = np.exp(-((x_pos[None, :] - center[:, None]) ** 2) / (2 * sigma ** 2))  # (T,N)
        active = (t_arr >= t0) & (t_arr < t0 + dur)
        a = intensity * g * active[:, None]
        occ = occ * (1.0 - np.clip(a, 0.0, 1.0))
    occ = np.clip(occ, 0.0, 1.0)

    # (4) 组装功率 = 容量 × 清晰天空 × 天气 × 云遮挡
    power = np.empty((T, n_sites))
    for i in range(n_sites):
        base = capacities[i] * sky * weather
        p = base * occ[:, i]
        p = np.where(p < 0, 0.0, p)
        power[:, i] = p

    # (5) 测量噪声（高频白噪声，约 ±2.5% 满量程量级，按容量缩放）
    noise = rng.normal(0.0, 1.0, size=power.shape)
    noise_amp = 0.025 * capacities[None, :]
    power = power + noise * noise_amp
    power = np.clip(power, 0.0, None)

    if return_meta:
        return timestamps, power, capacities, x_pos
    return timestamps, power, capacities


def save_synthetic_data(power, capacities, x_pos, timestamps,
                        power_file=cfg.SYNTH_POWER_FILE,
                        meta_file=cfg.SYNTH_META_FILE):
    """将生成数据落盘为 CSV（含格式说明所需的字段）。"""
    os.makedirs(os.path.dirname(power_file), exist_ok=True)
    n_sites = power.shape[1]
    with open(power_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp"] + [f"site_{i+1}" for i in range(n_sites)])
        for t, row in zip(timestamps, power):
            w.writerow([t.strftime("%Y-%m-%d %H:%M:%S")] + [f"{v:.4f}" for v in row])
    with open(meta_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "capacity_kwp", "x_position"])
        for i in range(len(capacities)):
            w.writerow([f"site_{i+1}", f"{capacities[i]:.2f}", f"{x_pos[i]:.4f}"])
    return power_file, meta_file


def ensure_data(exist_ok=True):
    """若数据文件缺失则生成；返回 (timestamps, power, capacities)。"""
    if (os.path.exists(cfg.SYNTH_POWER_FILE) and
            os.path.exists(cfg.SYNTH_META_FILE) and exist_ok):
        return load_data()
    timestamps, power, capacities, x_pos = generate_pv_data(return_meta=True)
    save_synthetic_data(power, capacities, x_pos, timestamps)
    return timestamps, power, capacities


def load_data(power_file=cfg.SYNTH_POWER_FILE, meta_file=cfg.SYNTH_META_FILE):
    """加载数据（兼容用户自带同格式 CSV）。

    要求格式：
      pv_power.csv     : 第一列 timestamp(可解析)，其余各列为站点功率(kW)
      site_metadata.csv: 列 site_id, capacity_kwp(, 可选 x_position)
    返回 timestamps(list), power(ndarray T×N), capacities(ndarray N)
    """
    with open(power_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        site_cols = [c for c in header if c != "timestamp"]
        rows = []
        ts = []
        for r in reader:
            ts.append(dt.datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S"))
            rows.append([float(x) for x in r[1:]])
    power = np.array(rows, dtype=float)
    # 读取容量
    cap_map = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            cap_map[r[0].strip()] = float(r[1])
    capacities = np.array([cap_map.get(c, np.nan) for c in site_cols], dtype=float)
    if np.any(np.isnan(capacities)):
        capacities = power.max(axis=0) * 1.05
        print("[warn] 部分站点容量缺失，已用峰值功率×1.05 估计。")
    return ts, power, capacities


if __name__ == "__main__":
    ts, pw, cap = ensure_data()
    print("power shape:", pw.shape, "capacities:", cap)
