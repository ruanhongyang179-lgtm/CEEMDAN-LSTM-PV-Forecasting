# -*- coding: utf-8 -*-
"""
preprocessing.py — 预处理与跨站点多通道张量构建（论文 Protocol 第 3 节）

流程：
  1) 按时间顺序划分 训练/验证/测试（避免未来信息泄漏，论文 7:1.5:1.5）；
  2) 对每个 (站点, 频率分量) 用「仅训练集」统计做标准化（论文 Eq.10）；
  3) 滑窗构造跨站点多通道三维张量 Z(t) ∈ R^{N×L×4}（论文 Eq.9-11）；
     目标为未来 H 步的标准化分量 Y(t) ∈ R^{N×H×4}；
  4) 提供滞后互相关系数（论文 Eq.8），用于量化云团传播时延。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config.settings as cfg


def temporal_split(T, train=cfg.SPLIT_TRAIN, val=cfg.SPLIT_VAL):
    """返回 (train_end, val_end) 样本索引切分点。"""
    c1 = int(T * train)
    c2 = int(T * (train + val))
    return c1, c2


def build_windows(components, L=cfg.INPUT_STEPS, H=cfg.HORIZON,
                  train=cfg.SPLIT_TRAIN, val=cfg.SPLIT_VAL):
    """从 (T,N,4) 分量序列构造滑窗样本，并按时间切分。

    返回 dict:
      'X_train/val/test' : (B, N, L, 4)  标准化输入
      'Y_train/val/test' : (B, N, H, 4)  标准化目标（未来分量）
      'scaler'           : 标准化器（mean/std），用于反归一化
    注意：窗口起始索引按全局时间点切分，保证测试窗口不窥探训练段。
    """
    T, N, K = components.shape
    c1, c2 = temporal_split(T, train, val)
    starts = np.arange(0, T - L - H + 1)
    # 窗口归属：训练窗口的整个预测段须落在训练段内；测试窗口起始在 val 之后
    train_mask = starts + H <= c1
    val_mask = (starts >= c1) & (starts + H <= c2)
    test_mask = starts >= c2

    # 标准化：仅用训练窗口覆盖的观测（按 (站点,频率) 统计）
    scaler = {"mean": np.zeros((N, K)), "std": np.zeros((N, K))}
    for i in range(N):
        for k in range(K):
            train_idx = starts[train_mask]  # 训练窗口的历史段
            vals = components[train_idx[:, None] + np.arange(L), i, k].ravel()
            m = vals.mean()
            s = vals.std()
            scaler["mean"][i, k] = m
            scaler["std"][i, k] = s if s > 1e-8 else 1.0

    def make_batch(mask):
        s_idx = starts[mask]
        B = len(s_idx)
        if B == 0:
            return None
        X = np.empty((B, N, L, K))
        Y = np.empty((B, N, H, K))
        for b, s in enumerate(s_idx):
            x_win = components[s:s + L, :, :]           # (L,N,K)
            y_win = components[s + L:s + L + H, :, :]   # (H,N,K)
            X[b] = np.transpose(x_win, (1, 0, 2))       # -> (N,L,K)
            Y[b] = np.transpose(y_win, (1, 0, 2))       # -> (N,H,K)
        # 标准化（逐 (i,k)）
        for i in range(N):
            for k in range(K):
                X[:, i, :, k] = (X[:, i, :, k] - scaler["mean"][i, k]) / scaler["std"][i, k]
                Y[:, i, :, k] = (Y[:, i, :, k] - scaler["mean"][i, k]) / scaler["std"][i, k]
        return X, Y

    out = {"scaler": scaler}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        res = make_batch(mask)
        if res is None:
            out[f"X_{name}"] = np.empty((0, N, L, K))
            out[f"Y_{name}"] = np.empty((0, N, H, K))
        else:
            out[f"X_{name}"], out[f"Y_{name}"] = res
    return out


def lag_cross_correlation(comp_a, comp_b, max_lag=cfg.INPUT_STEPS):
    """归一化互相关（论文 Eq.8），返回 lags 与对应相关系数，及峰值滞后步数。"""
    a = (comp_a - comp_a.mean()) / (comp_a.std() + 1e-12)
    b = (comp_b - comp_b.mean()) / (comp_b.std() + 1e-12)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for lag in lags:
        if lag >= 0:
            x = a[:len(a) - lag]
            y = b[lag:]
        else:
            x = a[-lag:]
            y = b[:len(b) + lag]
        if len(x) < 2:
            corr.append(0.0)
        else:
            corr.append(float(np.corrcoef(x, y)[0, 1]))
    corr = np.array(corr)
    peak_lag = int(lags[np.argmax(np.abs(corr))])
    return lags, corr, peak_lag


def site_correlation_matrix(components):
    """站点功率相关性矩阵：对 4 类分量求和还原站点功率后计算（用于可视化 Figure 4a）。"""
    T, N, K = components.shape
    site_power = components.sum(axis=2)  # (T,N)
    C = np.corrcoef(site_power, rowvar=False)  # (N,N)
    return C
