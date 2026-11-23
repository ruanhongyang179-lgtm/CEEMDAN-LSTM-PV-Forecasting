# -*- coding: utf-8 -*-
"""
model.py — 共享权重多通道 LSTM + 通道注意力 + 分位数回归头（论文 Protocol 第 4-5 节）

论文用 PyTorch 实现共享权重 LSTM（Eq.12）、通道注意力（Eq.13-14）与分位数
回归头（Eq.15）。本项目以纯 numpy 手写前向/反向传播等价替代：

  * 共享权重 LSTM：所有 (站点,频率) 通道共用同一套参数，按时间递归（Eq.12）。
  * 通道注意力：每个频率层内，站点间以固定温度的点积注意力加权融合
    （Eq.13-14）。论文注意力含可学习参数 W_q/W_k；本项目用无参固定温度
    点积注意力，并保留跨站点加权融合结构（详见 README 简化说明）。
  * 分位数回归头：由融合隐状态线性映射到未来 H 步的 3 个分位数（Eq.15）。
  * 损失：分位数 pinball 损失（Eq.20-21）+ L2 正则（Eq.23）。论文点预测
    分支 MSE（Eq.22，权重 0.1）作为评估指标单独计算，未并入训练目标，
    以保持训练图简洁稳健（见 README）。

实现要点：手写 BPTT，并在 __main__ 中做数值梯度校验，确保反向传播正确。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config.settings as cfg


def _softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _softmax_last(x):
    """对最后一维做 softmax（注意力分数沿站点轴）。"""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x):
    """数值稳定的 sigmoid（无 clip，避免大值溢出告警与额外开销）。"""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    # x>=0: 1/(1+exp(-x))；x<0: exp(x)/(1+exp(x))，二者均无溢出
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


class MultiChannelLSTM:
    """共享权重多通道 LSTM（轻量 numpy 实现）。"""

    def __init__(self, n_sites, n_freq, hidden=cfg.LSTM_HIDDEN,
                 horizon=cfg.HORIZON, quantiles=None, seed=0):
        self.N = n_sites
        self.K = n_freq
        self.H = hidden
        self.Hor = horizon
        self.Q = len(quantiles) if quantiles is not None else len(cfg.QUANTILES)
        self.quantiles = list(quantiles) if quantiles is not None else list(cfg.QUANTILES)
        rng = np.random.default_rng(seed)
        # LSTM 参数（共享）
        self.Wx = rng.standard_normal((4 * self.H, 1)) * 0.1      # 输入->门
        self.Wh = rng.standard_normal((4 * self.H, self.H)) * 0.1  # 隐态->门
        self.b = np.zeros(4 * self.H)
        # 解码器（分位数头）
        self.Wd = rng.standard_normal((self.H, self.Q * self.Hor)) * 0.1
        self.bd = np.zeros(self.Q * self.Hor)
        self._cache = None

    # ------------------------------------------------------------------ 前向
    def forward(self, X):
        """X: (B, N, L, K) 标准化输入。返回 Yhat: (B, N, K, Hor, Q)。"""
        B, N, L, K = X.shape
        Bc = B * N * K
        # 重组为 (Bc, L, 1) 的通道序列
        Xseq = X.transpose(0, 1, 3, 2).reshape(Bc, L, 1)  # (Bc,L,1)

        Hc = np.zeros((Bc, self.H))
        Cc = np.zeros((Bc, self.H))
        H_all = np.zeros((Bc, L, self.H))
        C_all = np.zeros((Bc, L, self.H))
        I = np.zeros((Bc, L, self.H))
        F = np.zeros((Bc, L, self.H))
        G = np.zeros((Bc, L, self.H))
        O = np.zeros((Bc, L, self.H))
        for t in range(L):
            xt = Xseq[:, t, 0]                          # (Bc,)
            z = xt[:, None] @ self.Wx.T + Hc @ self.Wh.T + self.b  # (Bc,4H)
            i = _sigmoid(z[:, 0:self.H])
            f = _sigmoid(z[:, self.H:2 * self.H])
            g = np.tanh(z[:, 2 * self.H:3 * self.H])
            o = _sigmoid(z[:, 3 * self.H:4 * self.H])
            Cc = f * Cc + i * g
            Hc = o * np.tanh(Cc)
            H_all[:, t, :] = Hc
            C_all[:, t, :] = Cc
            I[:, t, :] = i
            F[:, t, :] = f
            G[:, t, :] = g
            O[:, t, :] = o
        Hfinal = H_all[:, -1, :]  # (Bc,H)

        # 通道注意力（每个 freq 层内，站点间；批量向量化，去掉 Python 双层循环）
        Hmat = Hfinal.reshape(B, N, K, self.H).transpose(0, 2, 1, 3)  # (B,K,N,H)
        s = np.sqrt(self.H)
        scores = np.einsum('bkin,bkjn->bkij', Hmat, Hmat) / s        # (B,K,N,N)
        A = _softmax_last(scores)                                    # (B,K,N,N)
        context = np.einsum('bkij,bkjn->bkin', A, Hmat)             # (B,K,N,H)
        fused = (Hmat + context).transpose(0, 2, 1, 3).reshape(Bc, self.H)  # (Bc,H)

        # 解码器
        Yflat = fused @ self.Wd + self.bd            # (Bc, Q*Hor)
        Yhat = Yflat.reshape(Bc, self.Q, self.Hor).transpose(0, 2, 1)  # (Bc,Hor,Q)
        Yhat = Yhat.reshape(B, N, K, self.Hor, self.Q)

        self._cache = dict(Xseq=Xseq, H_all=H_all, C_all=C_all, I=I, F=F, G=G, O=O,
                           Hfinal=Hfinal, fused=fused, A=A, B=B, N=N, L=L, K=K, Bc=Bc)
        return Yhat

    # ------------------------------------------------------------------ 反向
    def backward(self, Yhat, Ytrue, weight_decay=cfg.WEIGHT_DECAY, loss_type="pinball"):
        """计算损失及其对全部参数的解析梯度。

        Yhat : (B,N,K,Hor,Q)  预测（标准化分量，来自 forward）
        Ytrue: (B,N,K,Hor)    目标（标准化分量）
        loss_type: "pinball"（训练用分位数损失）或 "mse"（梯度校验用的光滑替代，
                   仅用于反向图正确性验证，不影响训练）。
        返回 (loss, grads_dict, info)
        """
        c = self._cache
        Bc, H, Hor, Q = c["Bc"], self.H, self.Hor, self.Q
        L = c["L"]
        N, K = c["N"], c["K"]
        # ---- 输出层梯度 ----
        # 重组 Yhat 为 (Bc, Q, Hor)
        Yhat_flat = Yhat.reshape(Bc, Hor, Q)                 # (Bc,Hor,Q)
        Yhat_qh = Yhat_flat.transpose(0, 2, 1)              # (Bc,Q,Hor)
        Ytrue_rep = np.broadcast_to(Ytrue.reshape(Bc, Hor)[:, None, :], (Bc, Q, Hor))
        if loss_type == "mse":
            # 光滑二次损失：便于有限差分稳定校验（仅验证反向图结构）
            diff = Yhat_qh - Ytrue_rep
            loss_pin = np.mean(diff ** 2)
            grad_Yhat_qh = 2.0 * diff                       # dL/dYhat
        else:
            d = Ytrue_rep - Yhat_qh                         # (Bc,Q,Hor)
            tau = np.array(self.quantiles).reshape(1, Q, 1)
            g = np.where(d > 0, tau, np.where(d < 0, tau - 1.0, 0.0))
            pin = np.where(d > 0, tau * d, (tau - 1.0) * d)
            loss_pin = pin.mean()
            grad_Yhat_qh = -g                               # (Bc,Q,Hor)
        grad_Yflat = grad_Yhat_qh.reshape(Bc, Q * Hor)      # 列序 = q*Hor + h

        # ---- 解码器梯度 ----
        fused = c["fused"]                                  # (Bc,H)
        M = Bc * Hor * Q                                    # 输出项总数
        dWd = (fused.T @ grad_Yflat) / M                   # (H, Q*Hor)
        dbd = grad_Yflat.sum(axis=0) / M                    # (Q*Hor,)
        dfused = grad_Yflat @ self.Wd.T                     # (Bc,H)

        # ---- 注意力反向（固定温度 softmax，含 a 对 h 的依赖；批量向量化）----
        B, N, K = c["B"], c["N"], c["K"]
        Gmat = dfused.reshape(B, N, K, H).transpose(0, 2, 1, 3)  # (B,K,N,H)
        Hfinal = c["Hfinal"].reshape(B, N, K, H).transpose(0, 2, 1, 3)  # (B,K,N,H)
        A = c["A"]                                                # (B,K,N,N)
        s = np.sqrt(H)
        dH = Gmat.copy()                                           # (a) 恒等项
        dH += np.einsum('bkij,bkjn->bkin', A, Gmat)               # (b) 固定 a 的混合项 term2_i=Σ_j A_ij G_j
        D = np.einsum('bkin,bkjn->bkij', Gmat, Hfinal)            # D[i,j]=G_i·h_j
        aD = (A * D).sum(axis=-1, keepdims=True)                  # (B,K,N,1)
        dS = A * (D - aD)                                         # (B,K,N,N)
        dh_a = np.einsum('bkij,bkjn->bkin',
                         dS + dS.transpose(0, 1, 3, 2), Hfinal) / s  # (c) 经 a 的依赖
        dH += dh_a
        dHfinal = dH.transpose(0, 2, 1, 3).reshape(Bc, H)   # (Bc,H)

        # ---- LSTM BPTT（仅末步隐态作为输出）----
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        H_all = c["H_all"]; C_all = c["C_all"]
        I, F, Gg, O = c["I"], c["F"], c["G"], c["O"]
        Xseq = c["Xseq"]
        dHc = dHfinal.copy()                               # (Bc,H)
        dCc_future = np.zeros((Bc, H))                     # 来自更晚时间步的单元梯度
        for t in range(L - 1, -1, -1):
            i = I[:, t, :]; f = F[:, t, :]; g = Gg[:, t, :]; o = O[:, t, :]
            c_t = C_all[:, t, :]
            H_prev = H_all[:, t - 1, :] if t > 0 else np.zeros((Bc, H))
            C_prev = C_all[:, t - 1, :] if t > 0 else np.zeros((Bc, H))
            # 单元总梯度 = 未来项 + 输出项（o_t*tanh(c_t) 对 c_t 的依赖）
            dc_out = dHc * o * (1.0 - np.tanh(c_t) ** 2)
            dCc = dCc_future + dc_out                       # dL/dCc_t
            # 门梯度
            do = dHc * np.tanh(c_t)
            dg = dCc * i
            di = dCc * g
            df = dCc * C_prev                               # C_t = f*C_{t-1} + ... => dC_t/df = C_{t-1}
            dzi = di * i * (1 - i)
            dzf = df * f * (1 - f)
            dzg = dg * (1 - g ** 2)
            dzo = do * o * (1 - o)
            dz = np.concatenate([dzi, dzf, dzg, dzo], axis=1)  # (Bc,4H)
            xt = Xseq[:, t, 0]                                # (Bc,)
            dWx += dz.T @ xt[:, None]                        # (4H,1)
            dWh += dz.T @ H_prev                             # (4H,H)
            db += dz.sum(axis=0)
            # 回传
            dHc = dz @ self.Wh
            dCc_future = dCc * f
        # ---- L2 正则 ----
        loss_reg = 0.0
        for p in (self.Wx, self.Wh, self.Wd):
            loss_reg += 0.5 * np.sum(p ** 2)
        loss = loss_pin + weight_decay * loss_reg
        # pinball 梯度由“和”转为“均值”（与 loss_pin 的 mean 一致）
        dWx /= M
        dWh /= M
        db /= M
        grads = {
            "Wx": dWx + weight_decay * self.Wx,
            "Wh": dWh + weight_decay * self.Wh,
            "b": db,
            "Wd": dWd + weight_decay * self.Wd,
            "bd": dbd,
        }
        return loss, grads, {"loss_pin": float(loss_pin), "loss_reg": float(weight_decay * loss_reg)}

    # ------------------------------------------------------------------ 工具
    def params_and_grads(self):
        return [self.Wx, self.Wh, self.b, self.Wd, self.bd], \
               ["Wx", "Wh", "b", "Wd", "bd"]

    def predict(self, X):
        return self.forward(X)


# --------------------------------------------------------------------- 数值校验
def _numerical_grad_check():
    """对小型随机输入做解析梯度 vs 数值梯度校验。"""
    rng = np.random.default_rng(1)
    B, N, L, K, Hor, Q = 2, 2, 4, 2, 3, 2
    model = MultiChannelLSTM(n_sites=N, n_freq=K, hidden=5, horizon=Hor,
                             quantiles=[0.1, 0.9], seed=2)
    X = rng.standard_normal((B, N, L, K))
    Ytrue = rng.standard_normal((B, N, K, Hor))
    eps = 1e-4
    # 用光滑 MSE 损失校验反向图（pinball 为分段线性，近零梯度方向有限差分不稳）
    Yhat = model.forward(X)
    loss, grads, _ = model.backward(Yhat, Ytrue, weight_decay=0.0, loss_type="mse")
    params = [model.Wx, model.Wh, model.b, model.Wd, model.bd]
    names = ["Wx", "Wh", "b", "Wd", "bd"]
    max_err = 0.0
    for p, name in zip(params, names):
        ana = grads[name]
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]
            p[idx] = orig + eps
            l1 = model.forward(X); lp, _, _ = model.backward(l1, Ytrue, 0.0, loss_type="mse")
            p[idx] = orig - eps
            l2 = model.forward(X); lm, _, _ = model.backward(l2, Ytrue, 0.0, loss_type="mse")
            p[idx] = orig
            num[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        # 组合容差：相对误差与绝对误差任一满足即视为通过（避免近零分量误报）
        rel = np.max(np.abs(ana - num)) / (np.max(np.abs(num)) + 1e-12)
        absd = np.max(np.abs(ana - num))
        ok = (rel < 1e-4) or (absd < 1e-5)
        err = rel
        max_err = max(max_err, 0.0 if ok else rel)
        print(f"  {name}: rel_err={rel:.2e} abs_err={absd:.2e} {'OK' if ok else 'BAD'} "
              f"(解析范数 {np.linalg.norm(ana):.3f}, 数值范数 {np.linalg.norm(num):.3f})")
    print("数值梯度校验 (MSE 平滑损失)", "通过 ✅" if max_err == 0.0 else f"失败 ❌ (max rel err {max_err:.2e})")
    return max_err == 0.0


if __name__ == "__main__":
    _numerical_grad_check()
