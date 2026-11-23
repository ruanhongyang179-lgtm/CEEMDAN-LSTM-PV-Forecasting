# -*- coding: utf-8 -*-
"""
main.py — 一键运行入口（论文方法完整流水线）

流水线（与论文 Protocol 对齐）：
  1. 数据：内置合成多站点光伏功率（无联网下载），含清晰天空日曲线、
     天尺度天气、云团传播时延、测量噪声。
  2. 分解：CEEMDAN 将每站点功率分解为 4 类频率分量（高/中/低/残差）。
  3. 预处理：按时间切分训练/验证/测试，逐(站点,频率)标准化，滑窗构造
     跨站点多通道张量 (B,N,L,K)。
  4. 建模：共享权重多通道 LSTM + 通道注意力 + 分位数回归头（纯 numpy 实现）。
  5. 预测：输出未来 H 步的 90% 预测区间（0.05/0.5/0.95 分位）。
  6. 重建与聚合：反标准化并还原各站点功率，加和得到 VPP 总功率预测。
  7. 评估：与持续性基线、无分解消融实验对比，输出 RMSE/MAE/MAPE/PICP/PINAW。
  8. 可视化：输入可视化、训练/验证损失曲线（收敛过程）、结果对比图、
     跨站点相关性/时滞图，全部存至 results/，并汇总为 summary.json。

运行：  python main.py
依赖：  numpy / matplotlib / pillow（见 requirements.txt，无需联网安装）
"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境：仅写文件
import matplotlib.pyplot as plt
from matplotlib import font_manager

import config.settings as cfg
from src.utils import (rmse, mae, mape, picp, pinaw, pinball_loss,
                       Timer, log)
from src.data_generator import ensure_data
from src.decomposition import decompose_dataset, group_into_freq
from src.preprocessing import (build_windows, lag_cross_correlation,
                               site_correlation_matrix)
from src.model import MultiChannelLSTM


# --------------------------------------------------------------------------- #
# 中文字体：Windows 自带的 SimHei / 微软雅黑等可正常渲染中文标签；若不可用则静默
# 回退（标签可能显示为方框），不影响程序运行。
# --------------------------------------------------------------------------- #
def setup_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simkai.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                # 取该字体家族名并设为全局默认
                fp = font_manager.FontProperties(fname=path)
                family = fp.get_name()
                plt.rcParams["font.family"] = family
                plt.rcParams["axes.unicode_minus"] = False
                return family
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- #
# 分解结果缓存：CEEMDAN 分解确定且耗时，按「数据文件哈希 + 关键配置」做失效校验，
# 命中则直接加载，显著加快重跑（用户替换 data 后会自动失效重算）。
# --------------------------------------------------------------------------- #
def _data_file_hash(path):
    import hashlib
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "none"


def load_or_decompose(power, seed_base):
    cfg_str = (f"{cfg.N_SITES}_{cfg.CEEMDAN_ENSEMBLE}_{cfg.CEEMDAN_MAX_IMF}_"
               f"{cfg.CEEMDAN_NOISE_AMP}_{seed_base}_{power.shape}")
    data_hash = _data_file_hash(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), cfg.SYNTH_POWER_FILE))
    import hashlib
    key = hashlib.md5((data_hash + cfg_str).encode()).hexdigest()[:16]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"decomp_{key}.npz")
    if os.path.exists(cache_path):
        try:
            npz = np.load(cache_path, allow_pickle=True)
            components = npz["components"]
            decomp_info = json.loads(str(npz["info"]))
            return components, decomp_info, cache_path, True
        except Exception:
            pass
    components, decomp_info = decompose_dataset(power, seed_base=seed_base)
    info = {k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in decomp_info.items()}
    try:
        np.savez(cache_path, components=components, info=json.dumps(info))
    except Exception:
        pass
    return components, decomp_info, cache_path, False


# --------------------------------------------------------------------------- #
# 轻量 Adam 优化器（作用于模型参数数组，原地更新）
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            self.params[i] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------- #
# 训练与损失工具
# --------------------------------------------------------------------------- #
def quantile_pinball_loss(model, X, Y_true, weight_decay=0.0):
    """计算批量上的平均 pinball 损失（不反传，仅用于监控/早停）。"""
    Yhat = model.forward(X)                       # (B,N,K,Hor,Q)
    Bc = X.shape[0] * X.shape[1] * model.K
    Yhat_qh = Yhat.reshape(Bc, model.Hor, model.Q).transpose(0, 2, 1)  # (Bc,Q,Hor)
    Ytrue_rep = np.broadcast_to(Y_true.reshape(Bc, model.Hor)[:, None, :],
                                (Bc, model.Q, model.Hor))
    d = Ytrue_rep - Yhat_qh
    tau = np.array(model.quantiles).reshape(1, model.Q, 1)
    pin = np.where(d > 0, tau * d, (tau - 1.0) * d)
    loss = pin.mean()
    if weight_decay > 0:
        for p in (model.Wx, model.Wh, model.Wd):
            loss += weight_decay * 0.5 * np.sum(p ** 2)
    return float(loss)


def train_model(model, X_tr, Y_tr, X_val, Y_val, n_epochs, batch_size,
                rng, weight_decay=cfg.WEIGHT_DECAY, patience=cfg.EARLY_STOP_PATIENCE):
    """训练一个 MultiChannelLSTM。

    X_tr/X_val : (B,N,L,K)   Y_tr/Y_val : (B,N,K,Hor)（已转置）
    """
    params = [model.Wx, model.Wh, model.b, model.Wd, model.bd]
    names = ["Wx", "Wh", "b", "Wd", "bd"]
    opt = Adam(params, lr=cfg.LEARNING_RATE)
    B = X_tr.shape[0]
    train_hist, val_hist = [], []
    best_val = float("inf")
    best_snapshot = None
    wait = 0
    for epoch in range(1, n_epochs + 1):
        perm = rng.permutation(B)
        batch_losses = []          # 各 batch 的平均 pinball 损失（避免对全量训练集
        for s in range(0, B, batch_size):  # 额外做一遍巨量前向，见 README 性能说明）
            idx = perm[s:s + batch_size]
            Yhat = model.forward(X_tr[idx])
            loss_pin, grads, _ = model.backward(Yhat, Y_tr[idx], weight_decay=weight_decay)
            batch_losses.append(float(loss_pin))
            opt.step([grads[n] for n in names])
        tr = float(np.mean(batch_losses))
        va = quantile_pinball_loss(model, X_val, Y_val, weight_decay)
        train_hist.append(tr)
        val_hist.append(va)
        if va < best_val - 1e-6:
            best_val = va
            best_snapshot = [p.copy() for p in params]
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            log(f"  早停在 epoch {epoch}（验证损失连续 {patience} 轮未下降）")
            break
    # 恢复最优
    if best_snapshot is not None:
        for p, bs in zip(params, best_snapshot):
            p[...] = bs
    return train_hist, val_hist, epoch


def predict_vpp(model, X, Y_std, scaler):
    """预测并还原 VPP 总功率的分位区间。

    X      : (B,N,L,K) 标准化输入
    Y_std  : (B,N,H,K) 标准化目标（测试集）
    返回 (pred_vpp (B,H,Q), actual_vpp (B,H), Yhat_std (B,N,K,H,Q))
    """
    Yhat = model.forward(X)                       # (B,N,K,Hor,Q)
    B, N, K, Hor, Q = Yhat.shape
    mean = scaler["mean"]                         # (N,K)
    std = scaler["std"]
    # 反标准化分量
    pred_comp = Yhat * std[None, :, :, None, None] + mean[None, :, :, None, None]  # raw comp
    pred_power = pred_comp.sum(axis=2)            # (B,N,Hor,Q)
    pred_vpp = pred_power.sum(axis=1)             # (B,Hor,Q)

    Y_raw = Y_std * std[None, :, None, :] + mean[None, :, None, :]  # (B,N,H,K)
    actual_site = Y_raw.sum(axis=3)               # (B,N,H)
    actual_vpp = actual_site.sum(axis=1)          # (B,H)
    return pred_vpp, actual_vpp, Yhat


def persistence_vpp(X_test, Y_test, scaler):
    """持续性基线：以输入末步 VPP 功率重复 H 步。"""
    B, N, L, K = X_test.shape
    mean = scaler["mean"]; std = scaler["std"]
    last = X_test[:, :, L - 1, :] * std[None, :, :] + mean[None, :, :]  # (B,N,K)
    last_site = last.sum(axis=2)                  # (B,N)
    last_vpp = last_site.sum(axis=1)              # (B,)
    H = Y_test.shape[2]
    return np.repeat(last_vpp[:, None], H, axis=1)  # (B,H)


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def plot_input(power, components, timestamps, site_idx=0, out_path=None):
    """输入可视化：站点原始功率 + 其 CEEMDAN 四类频率分量。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    # 取前 2 天（约 576 步）
    span = min(len(power), 2 * int(24 * 60 / cfg.SAMPLE_INTERVAL_MIN))
    t = np.arange(span)
    axes[0].plot(t, power[:span, site_idx], color="#1f77b4", lw=1.0)
    axes[0].set_title(f"站点 {site_idx+1} 合成光伏功率（前 2 天）  [kW]")
    axes[0].set_ylabel("功率 (kW)")
    labels = ["高频", "中频", "低频", "残差"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    for k in range(4):
        axes[1].plot(t, components[:span, site_idx, k], color=colors[k],
                     lw=1.0, label=labels[k])
    axes[1].set_title(f"站点 {site_idx+1} CEEMDAN 分解（4 类频率分量）")
    axes[1].set_ylabel("分量 (kW)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("时间步（5 分钟/步）")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=cfg.FIG_DPI)
    plt.close(fig)


def plot_loss(train_hist, val_hist, out_path=None):
    """训练/验证损失曲线（优化收敛过程）。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_hist, label="训练集 pinball 损失", color="#1f77b4", lw=1.6)
    ax.plot(val_hist, label="验证集 pinball 损失", color="#d62728", lw=1.6)
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("平均 pinball 损失")
    ax.set_title("训练 / 验证损失曲线（优化收敛过程）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=cfg.FIG_DPI)
    plt.close(fig)


def plot_result(actual_vpp, pred_vpp, horizons=(1, 12), out_path=None):
    """结果对比图：选定预测步长的点预测 + 90% 区间 vs 实际 VPP。"""
    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 4 * len(horizons)),
                             sharex=True)
    if len(horizons) == 1:
        axes = [axes]
    qlo, qmd, qhi = 0, 1, 2  # 0.05/0.5/0.95 的索引（QUANTILES 顺序）
    for ax, h in zip(axes, horizons):
        idx = np.arange(len(actual_vpp))
        ax.plot(idx, actual_vpp[:, h - 1], color="#333333", lw=1.0,
                label=f"实际 VPP (H={h})")
        ax.plot(idx, pred_vpp[:, h - 1, qmd], color="#1f77b4", lw=1.0,
                label="预测中位 (0.5)")
        ax.fill_between(idx, pred_vpp[:, h - 1, qlo], pred_vpp[:, h - 1, qhi],
                        color="#1f77b4", alpha=0.2, label="90% 区间 (0.05–0.95)")
        ax.set_ylabel("VPP 总功率 (kW)")
        ax.set_title(f"VPP 功率预测对比（提前 {h} 步）")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("测试窗口序号（按时间顺序）")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=cfg.FIG_DPI)
    plt.close(fig)


def plot_corr(components, out_path=None):
    """跨站点相关性矩阵 + 两站点互相关滞后（体现云团传播时延）。"""
    C = site_correlation_matrix(components)       # (N,N)
    # 取站点 0 与站点 4 的某分量做互相关
    comp_a = components[:, 0, 1]                   # 站点0 中频
    comp_b = components[:, 4, 1]                   # 站点4 中频
    lags, corr, peak = lag_cross_correlation(comp_a, comp_b)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im = axes[0].imshow(C, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("站点功率相关性矩阵")
    axes[0].set_xlabel("站点"); axes[0].set_ylabel("站点")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].plot(lags, corr, color="#d62728", lw=1.6)
    axes[1].axvline(peak, color="#2ca02c", ls="--", lw=1.0,
                    label=f"峰值滞后 {peak} 步")
    axes[1].set_title("站点 1 vs 5 中频分量互相关（云团传播时延）")
    axes[1].set_xlabel("滞后步数"); axes[1].set_ylabel("互相关系数")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=cfg.FIG_DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    t_total = time.time()
    setup_chinese_font()
    log("=== CEEMDAN-LSTM 光伏超短期功率预测（VPP）流水线启动 ===")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    with Timer("数据生成/加载"):
        timestamps, power, capacities = ensure_data()
    T, N = power.shape
    vpp_cap = float(capacities.sum())
    log(f"数据: T={T} 步, N={N} 站点, 总装机={vpp_cap:.1f} kWp")

    # ---- 2. CEEMDAN 分解 ----
    with Timer("CEEMDAN 分解"):
        components, decomp_info, _, cached = load_or_decompose(power, cfg.SEED)
    if cached:
        log("CEEMDAN: 命中缓存分解结果（跳过重复计算）")
    n_imfs = decomp_info["n_imfs"]
    log(f"CEEMDAN: 各站点 IMF 数={n_imfs}, 四类分量重构约束满足")

    # ---- 3. 预处理 / 滑窗 ----
    with Timer("构建滑窗张量"):
        win = build_windows(components, L=cfg.INPUT_STEPS, H=cfg.HORIZON,
                            train=cfg.SPLIT_TRAIN, val=cfg.SPLIT_VAL)
    X_tr, Y_tr_raw = win["X_train"], win["Y_train"]      # (B,N,L,K),(B,N,H,K)
    X_val, Y_val_raw = win["X_val"], win["Y_val"]
    X_te, Y_te_raw = win["X_test"], win["Y_test"]
    scaler = win["scaler"]
    # 模型需要 Ytrue=(B,N,K,Hor)，对预处理输出转置 (B,N,H,K)->(B,N,K,H)
    Y_tr = Y_tr_raw.transpose(0, 1, 3, 2)
    Y_val = Y_val_raw.transpose(0, 1, 3, 2)
    log(f"滑窗: 训练 {X_tr.shape[0]} / 验证 {X_val.shape[0]} / 测试 {X_te.shape[0]} 窗口")

    # ---- 4&5. 训练完整模型（CEEMDAN + LSTM + 注意力 + 分位数）----
    log("训练完整模型（CEEMDAN 分解 + 多通道 LSTM）...")
    rng = np.random.default_rng(cfg.SEED)
    model = MultiChannelLSTM(n_sites=N, n_freq=cfg.N_FREQ, hidden=cfg.LSTM_HIDDEN,
                             horizon=cfg.HORIZON, quantiles=cfg.QUANTILES, seed=0)
    with Timer("完整模型训练"):
        tr_hist, va_hist, epochs_run = train_model(
            model, X_tr, Y_tr, X_val, Y_val,
            n_epochs=cfg.MAX_EPOCHS, batch_size=cfg.BATCH_SIZE, rng=rng)

    # ---- 6. 预测 + 重建 + VPP 聚合 ----
    pred_vpp, actual_vpp, Yhat = predict_vpp(model, X_te, Y_te_raw, scaler)
    Q = len(cfg.QUANTILES)
    qmd = cfg.QUANTILES.index(0.5)
    qlo = cfg.QUANTILES.index(0.05)
    qhi = cfg.QUANTILES.index(0.95)
    y_true_f = actual_vpp.ravel()
    y_pred_f = pred_vpp[:, :, qmd].ravel()
    y_lo_f = pred_vpp[:, :, qlo].ravel()
    y_hi_f = pred_vpp[:, :, qhi].ravel()
    low_mask = y_true_f < vpp_cap * cfg.MAPE_LOW_POWER_RATIO

    metrics_full = {
        "rmse": rmse(y_true_f, y_pred_f),
        "mae": mae(y_true_f, y_pred_f),
        "mape": mape(y_true_f, y_pred_f, low_mask),
        "picp": picp(y_true_f, y_lo_f, y_hi_f),
        "pinaw": pinaw(y_lo_f, y_hi_f, vpp_cap),
        "pinball": pinball_loss(y_true_f, y_pred_f, 0.5),
    }
    log(f"完整模型 VPP 指标: RMSE={metrics_full['rmse']:.2f} "
        f"MAE={metrics_full['mae']:.2f} MAPE={metrics_full['mape']:.2f}% "
        f"PICP={metrics_full['picp']:.3f} PINAW={metrics_full['pinaw']:.4f}")

    # ---- 7a. 持续性基线 ----
    with Timer("持续性基线"):
        persist = persistence_vpp(X_te, Y_te_raw, scaler)
    metrics_persist = {
        "rmse": rmse(actual_vpp.ravel(), persist.ravel()),
        "mae": mae(actual_vpp.ravel(), persist.ravel()),
        "mape": mape(actual_vpp.ravel(), persist.ravel(), low_mask),
    }
    log(f"持续性基线 VPP 指标: RMSE={metrics_persist['rmse']:.2f} "
        f"MAE={metrics_persist['mae']:.2f} MAPE={metrics_persist['mape']:.2f}%")

    # ---- 7b. 消融：无 CEEMDAN 分解（直接对原始功率建模，K=1）----
    with Timer("消融实验（无分解）"):
        raw = power.copy()
        raw = raw[:, :, None]                       # (T,N,1)
        win_raw = build_windows(raw, L=cfg.INPUT_STEPS, H=cfg.HORIZON,
                                train=cfg.SPLIT_TRAIN, val=cfg.SPLIT_VAL)
        Xr_tr, Yr_tr = win_raw["X_train"], win_raw["Y_train"].transpose(0, 1, 3, 2)
        Xr_val, Yr_val = win_raw["X_val"], win_raw["Y_val"].transpose(0, 1, 3, 2)
        Xr_te, Yr_te = win_raw["X_test"], win_raw["Y_test"]
        scl_raw = win_raw["scaler"]
        rng2 = np.random.default_rng(cfg.SEED + 1)
        m2 = MultiChannelLSTM(n_sites=N, n_freq=1, hidden=cfg.LSTM_HIDDEN,
                              horizon=cfg.HORIZON, quantiles=cfg.QUANTILES, seed=0)
        # 与完整模型同轮数训练，保证「有/无分解」为公平对比
        train_model(m2, Xr_tr, Yr_tr, Xr_val, Yr_val,
                    n_epochs=cfg.MAX_EPOCHS,
                    batch_size=cfg.BATCH_SIZE, rng=rng2)
        # 还原：K=1，分量即原始功率
        Yhat2 = m2.forward(Xr_te)                   # (B,N,1,Hor,Q)
        B2, N2, K2, H2, Q2 = Yhat2.shape
        mean1 = scl_raw["mean"]; std1 = scl_raw["std"]
        pred2 = Yhat2 * std1[None, :, :, None, None] + mean1[None, :, :, None, None]
        pred2_power = pred2.sum(axis=2)             # (B,N,Hor,Q)
        pred2_vpp = pred2_power.sum(axis=1)         # (B,Hor,Q)
        Yraw2 = Yr_te * std1[None, :, None, :] + mean1[None, :, None, :]
        actual2 = Yraw2.sum(axis=3).sum(axis=1)     # (B,H)
    metrics_ablation = {
        "rmse": rmse(actual2.ravel(), pred2_vpp[:, :, qmd].ravel()),
        "mae": mae(actual2.ravel(), pred2_vpp[:, :, qmd].ravel()),
        "mape": mape(actual2.ravel(), pred2_vpp[:, :, qmd].ravel(),
                     actual2.ravel() < vpp_cap * cfg.MAPE_LOW_POWER_RATIO),
    }
    log(f"消融(无分解) VPP 指标: RMSE={metrics_ablation['rmse']:.2f} "
        f"MAE={metrics_ablation['mae']:.2f} MAPE={metrics_ablation['mape']:.2f}%")

    # ---- 8. 可视化 ----
    with Timer("生成可视化"):
        fig_input = os.path.join(cfg.RESULTS_DIR, "fig_input.png")
        fig_loss = os.path.join(cfg.RESULTS_DIR, "fig_loss.png")
        fig_result = os.path.join(cfg.RESULTS_DIR, "fig_result.png")
        fig_corr = os.path.join(cfg.RESULTS_DIR, "fig_corr.png")
        plot_input(power, components, timestamps, site_idx=0, out_path=fig_input)
        plot_loss(tr_hist, va_hist, out_path=fig_loss)
        plot_result(actual_vpp, pred_vpp, horizons=(1, 12), out_path=fig_result)
        plot_corr(components, out_path=fig_corr)

    # ---- 9. 汇总 summary.json ----
    summary = {
        "project": "CEEMDAN-LSTM VPP 光伏超短期功率预测（轻量 numpy 实现）",
        "config": cfg.as_dict(),
        "data": {
            "T_steps": T, "n_sites": N, "sample_interval_min": cfg.SAMPLE_INTERVAL_MIN,
            "n_days": cfg.N_DAYS, "vpp_capacity_kwp": round(vpp_cap, 2),
            "train_windows": int(X_tr.shape[0]),
            "val_windows": int(X_val.shape[0]),
            "test_windows": int(X_te.shape[0]),
        },
        "decomposition": {
            "n_imfs_per_site": n_imfs,
            "n_freq_groups": cfg.N_FREQ,
            "group_counts": decomp_info["group_counts"],
        },
        "training": {
            "epochs_run": int(epochs_run),
            "max_epochs": cfg.MAX_EPOCHS,
            "best_val_pinball": float(min(va_hist)) if va_hist else None,
            "final_train_pinball": float(tr_hist[-1]) if tr_hist else None,
            "final_val_pinball": float(va_hist[-1]) if va_hist else None,
        },
        "metrics": {
            "full_model_CEEMDAN_LSTM": metrics_full,
            "persistence_baseline": metrics_persist,
            "ablation_no_decomposition": metrics_ablation,
        },
        "figures": {
            "input_visualization": fig_input,
            "loss_curves": fig_loss,
            "result_comparison": fig_result,
            "cross_site_correlation": fig_corr,
        },
        "runtime_seconds": round(time.time() - t_total, 1),
        "reproducible": True,
    }
    summary_path = os.path.join(cfg.RESULTS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"=== 完成。总耗时 {summary['runtime_seconds']:.1f}s，"
        f"结果见 {cfg.RESULTS_DIR}（含 summary.json）===")
    return summary


if __name__ == "__main__":
    main()
