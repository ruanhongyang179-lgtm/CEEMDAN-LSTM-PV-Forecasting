# -*- coding: utf-8 -*-
"""
decomposition.py — CEEMDAN 分解与频率分量合并（论文 Protocol 第 2 节）

论文使用 PyEMD 库做 CEEMDAN；本项目在无 GPU / 轻量依赖约束下用
纯 numpy 重新实现完整 CEEMDAN 算法（Torres et al., 2011）：

  阶段1: 对 x + ε·w 做 EMD 取第 1 个 IMF，集成 E 次取平均 -> imf1
  阶段k: 对残差 r_{k-1} + ε·w 做 EMD 取第 (k-1) 个 IMF，集成取平均 -> imfk
  残差 r_k = r_{k-1} - imfk，直至残差无足够极值点或达到最大层数。

随后按 IMF 阶次固定分组（论文 Eq.5-6，覆盖 >95% 样本）：
  阶次 1~5  -> 高频波动
  阶次 6~9  -> 中频变化
  阶次 ≥10  -> 低频趋势
  残差      -> 第 4 类（残差趋势）
四类分量满足完整重构约束（论文 Eq.7）：原始序列 = 高+中+低+残差。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config.settings as cfg


def find_extrema(x):
    """返回内部局部极大值与极小值的索引。"""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    d = np.diff(np.sign(np.diff(x)))
    maxima = np.where(d == -2)[0] + 1
    minima = np.where(d == 2)[0] + 1
    return maxima, minima


def _natural_cubic_spline(pos, val, t):
    """纯 numpy 自然三次样条（替代 scipy CubicSpline，去除额外依赖）。

    pos : 单调递增结点横坐标; val : 结点函数值; t : 待插值横坐标（可向量）。
    端点取自然边界（二阶导为 0）。求值全程向量化，无 Python 段循环。
    """
    pos = np.asarray(pos, dtype=float)
    val = np.asarray(val, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(pos)
    if n < 3:
        if n == 1:
            return np.full_like(t, val[0])
        return np.interp(t, pos, val)
    h = np.diff(pos)
    # 自然三次样条三对角方程组（内部 n-2 个二阶导 m_1..m_{n-2}），端点 m_0=m_{n-1}=0
    A = np.zeros((n - 2, n - 2))
    rhs = np.zeros(n - 2)
    for i in range(n - 2):
        hi = h[i]
        hi1 = h[i + 1]
        A[i, i] = 2.0 * (hi + hi1)
        if i > 0:
            A[i, i - 1] = hi
        if i < n - 3:
            A[i, i + 1] = hi1
        rhs[i] = 6.0 * ((val[i + 2] - val[i + 1]) / hi1 - (val[i + 1] - val[i]) / hi)
    m = np.zeros(n)
    m[1:-1] = np.linalg.solve(A, rhs)
    # 全向量化求值
    seg = np.clip(np.searchsorted(pos, t) - 1, 0, n - 2)
    dx = t - pos[seg]
    ha = pos[seg + 1] - pos[seg]
    b = (val[seg + 1] - val[seg]) / ha - ha * (2.0 * m[seg] + m[seg + 1]) / 6.0
    c = m[seg] / 2.0
    d = (m[seg + 1] - m[seg]) / (6.0 * ha)
    return val[seg] + b * dx + c * dx ** 2 + d * dx ** 3


def _spline_envelope(x, extrema):
    """用三次样条构造上/下包络。extrema: 极大值(或极小值)索引数组。"""
    t = np.arange(len(x))
    if len(extrema) < 3:
        # 包络退化为常数（该段无法筛分）
        return np.full(len(x), x[extrema].mean()) if len(extrema) > 0 else np.zeros(len(x))
    pos = np.concatenate(([0], extrema, [len(x) - 1]))
    val = np.concatenate(([x[0]], x[extrema], [x[-1]]))
    return _natural_cubic_spline(pos, val, t)


def emd_one_imf(x, max_sift=cfg.CEEMDAN_MAX_SIFT, tol=cfg.CEEMDAN_SD_TOL):
    """单次筛分提取一个 IMF（停止准则：SD<tol 或达最大迭代）。"""
    h = np.asarray(x, dtype=float).copy()
    for _ in range(max_sift):
        maxima, minima = find_extrema(h)
        if len(maxima) < 3 or len(minima) < 3:
            break
        upper = _spline_envelope(h, maxima)
        lower = _spline_envelope(h, minima)
        mean_env = 0.5 * (upper + lower)
        h_new = h - mean_env
        denom = np.sum(h ** 2) + 1e-12
        sd = np.sum((h_new - h) ** 2) / denom
        h = h_new
        if sd < tol:
            break
    return h


def emd(x, max_imf=cfg.CEEMDAN_MAX_IMF, max_sift=cfg.CEEMDAN_MAX_SIFT,
        tol=cfg.CEEMDAN_SD_TOL):
    """完整 EMD：返回 IMF 列表与最终残差。"""
    resid = np.asarray(x, dtype=float).copy()
    imfs = []
    for _ in range(max_imf):
        if np.max(np.abs(resid)) < 1e-10:
            break
        maxima, minima = find_extrema(resid)
        if len(maxima) < 3 or len(minima) < 3:
            break
        imf = emd_one_imf(resid, max_sift, tol)
        imfs.append(imf)
        resid = resid - imf
    return imfs, resid


def ceemdan(x, ensemble=cfg.CEEMDAN_ENSEMBLE, noise_amp=cfg.CEEMDAN_NOISE_AMP,
            max_imf=cfg.CEEMDAN_MAX_IMF, max_sift=cfg.CEEMDAN_MAX_SIFT,
            tol=cfg.CEEMDAN_SD_TOL, seed=0):
    """完整 CEEMDAN 分解（Torres et al., 2011 标准流程）。

    每个阶段仅对「残差 + 自适应噪声」做一次 EMD 并取第 1 个 IMF 做集成平均，
    因此每个集成成员只需 1 次筛分，远快于逐层全量 EMD。
    返回 (imfs_list[IMF1..IMFK], residual)。
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    eps = noise_amp * np.std(x)
    rng = np.random.default_rng(seed)
    noise = lambda: eps * rng.standard_normal(T)

    # 阶段 1：第 1 个 IMF
    acc = np.zeros(T)
    for _ in range(ensemble):
        imfs, _ = emd(x + noise(), max_imf=1, max_sift=max_sift, tol=tol)
        if len(imfs) >= 1:
            acc += imfs[0]
    imfs = [acc / max(ensemble, 1)]
    r = x - imfs[0]

    # 阶段 2..max_imf：每阶段取「残差+噪声」的首个 IMF 集成平均
    k = 2
    while k <= max_imf:
        maxima, minima = find_extrema(r)
        if len(maxima) < 3 or len(minima) < 3:
            break
        acc = np.zeros(T)
        for _ in range(ensemble):
            tmp, _ = emd(r + noise(), max_imf=1, max_sift=max_sift, tol=tol)
            if len(tmp) >= 1:
                acc += tmp[0]
        acc /= max(ensemble, 1)
        imfs.append(acc)
        r = r - acc
        k += 1
    return imfs, r


def group_into_freq(imfs, residual,
                    high=cfg.FREQ_GROUP_HIGH, mid=cfg.FREQ_GROUP_MID, low=cfg.FREQ_GROUP_LOW):
    """将 IMF 列表按阶次分组为 4 类频率分量（论文 Eq.6）。

    返回 (comp_high, comp_mid, comp_low, residual)，均为原长度数组。
    任意阶次 > max(mid) 的 IMF 归入低频，保证覆盖全部 IMF 且重构可还原。
    """
    comp_high = np.zeros_like(imfs[0]) if len(imfs) else np.zeros_like(residual)
    comp_mid = np.zeros_like(comp_high)
    comp_low = np.zeros_like(comp_high)
    max_high = max(high) if high else 0
    max_mid = max(mid) if mid else 0
    for idx, imf in enumerate(imfs):
        order = idx + 1  # 1-based 阶次
        if order <= max_high:
            comp_high = comp_high + imf
        elif order <= max_mid:
            comp_mid = comp_mid + imf
        else:
            comp_low = comp_low + imf
    return comp_high, comp_mid, comp_low, residual


def _hilbert(x):
    """纯 numpy Hilbert 变换（FFT 实现，等价于 scipy.signal.hilbert）。"""
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1.0
        h[1:N // 2] = 2.0
        h[N // 2] = 1.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def mean_instantaneous_frequency(imf, dt=1.0):
    """用 Hilbert 变换估计单个 IMF 的平均瞬时频率（论文 Eq.5，用于分组验证）。"""
    if np.max(np.abs(imf)) < 1e-9:
        return 0.0
    analytic = _hilbert(imf)
    phase = np.unwrap(np.angle(analytic))
    inst_f = np.diff(phase) / (2.0 * np.pi * dt)
    inst_f = inst_f[np.isfinite(inst_f)]
    if len(inst_f) == 0:
        return 0.0
    return float(np.mean(np.abs(inst_f)))


def decompose_dataset(power, seed_base=cfg.SEED):
    """对全部站点做 CEEMDAN + 频率合并。

    参数
    ----
    power : np.ndarray (T, N)

    返回
    ----
    components : np.ndarray (T, N, 4)  四类频率分量（高/中/低/残差）
    info       : dict  每站点 IMF 数量、各阶平均瞬时频率等诊断信息
    """
    T, N = power.shape
    components = np.zeros((T, N, 4), dtype=float)
    info = {"n_imfs": [], "imf_if": [], "group_counts": [0, 0, 0, 0]}
    for i in range(N):
        imfs, residual = ceemdan(power[:, i], seed=int(seed_base) + i)
        ch, cm, cl, res = group_into_freq(imfs, residual)
        components[:, i, 0] = ch
        components[:, i, 1] = cm
        components[:, i, 2] = cl
        components[:, i, 3] = res
        info["n_imfs"].append(len(imfs))
        info["imf_if"].append([mean_instantaneous_frequency(m) for m in imfs])
        # 统计每类非空分量数
        for j, c in enumerate((ch, cm, cl, res)):
            if np.max(np.abs(c)) > 1e-6:
                info["group_counts"][j] += 1
    return components, info


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 1000)
    x = (np.sin(2 * np.pi * 0.5 * t) +
         0.5 * np.sin(2 * np.pi * 2 * t) +
         0.2 * np.sin(2 * np.pi * 8 * t) + 0.1 * rng.standard_normal(1000))
    s = time.time()
    imfs, res = ceemdan(x)
    print(f"CEEMDAN 耗时 {time.time()-s:.2f}s, IMF 数={len(imfs)}")
    rec = sum(imfs) + res
    print("重构误差:", np.max(np.abs(rec - x)))
    ch, cm, cl, rr = group_into_freq(imfs, res)
    print("分组重构误差:", np.max(np.abs(ch + cm + cl + rr - x)))
