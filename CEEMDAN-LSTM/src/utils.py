# -*- coding: utf-8 -*-
"""utils.py — 通用工具：评估指标、日志、计时。"""
import time
import numpy as np


def rmse(y_true, y_pred):
    """均方根误差 (论文 Eq.24)。y_true, y_pred: 一维数组。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """平均绝对误差 (论文 Eq.25)。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, low_power_mask=None):
    """平均绝对百分比误差 (论文 Eq.26)。

    low_power_mask: 布尔数组，True 表示对应时刻为“低功率时段”，
    计算时将被剔除（与论文一致：剔除聚合功率 < 总装机 5% 的时段）。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true)
    valid = np.ones_like(denom, dtype=bool)
    if low_power_mask is not None:
        valid = valid & (~np.asarray(low_power_mask, dtype=bool))
    # 额外保护：剔除分母为 0 的点
    valid = valid & (denom > 1e-9)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)


def picp(y_true, lower, upper):
    """区间预测覆盖率 (论文 Eq.27)：观测值落入 [lower, upper] 的比例。"""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def pinaw(lower, upper, capacity):
    """归一化区间平均宽度 (论文 Eq.28)：区间宽度 / VPP 总装机。"""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean(upper - lower) / capacity)


def pinball_loss(y_true, y_pred, tau):
    """分位数回归 pinball 损失 (论文 Eq.20)。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


class Timer:
    """简单的计时上下文管理器。"""

    def __init__(self, name="task"):
        self.name = name
        self.t0 = None
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        print(f"[计时] {self.name}: {self.elapsed:.2f}s")
        return False


def log(msg):
    """带时间戳的阶段日志。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
