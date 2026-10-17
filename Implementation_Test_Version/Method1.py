"""
方法1：先 CDFA，后 ML；ML 只在 CDFA 输出列上学习。

步骤：
  训练
    T0 = train = [f0, f1, ..., fk]           （原始列）
    T1 = train[1:] = [f1, f2, ..., fk]       （原始列上移一格）
    A0 = CDFA(T0)                            （T0 的增强/推演列，与 T0 等长）
    A1 = CDFA(T1)                            （T1 的增强/推演列，与 T1 等长）
    ML 学习:  (A0[t-L:t], A1[t-L:t])  →  T1[t:t+H]
    （输入特征只有 A0、A1 两列（全部来自 CDFA），原始序列不作为输入。）
  测试
    用 train 末段生成 (A0, A1) → 取最后 L 行做查询窗 → ML 直接多步输出 H = len(test) 个点。

原始序列只用于提供训练标签（真实值），模型的输入全部来自 CDFA 生成的列。

CDFA 实施：
  ① ξ_i   = corr( f[i:], f[:-i] ),  i = 1..ws
  ② φ_i   = (ξ_i−min)/(max−min) if ξ_i≥0 else (ξ_i−max)/(max−min))
  ③ Δ_k   = (f_k^ws · Φ)/(ws·ς);  f̂_{k+1} = f_k + Δ_k
  ④ f_aug[k+1] = f[k+1] + α·(f̂_{k+1} − f[k+1]) (+噪声)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))   # 脚本所在目录（仓库根目录）
DATA_DIR = os.path.join(HERE, 'Datasets', 'Darts')      # 数据在仓库 Datasets/Darts 下
OUT_DIR = HERE                                          # 结果 CSV 直接输出到仓库根目录

SPLIT = 0.8
L_GRID = [2, 3, 4, 6, 8, 12, 16, 24, 36, 48]
K_GRID = [1, 2, 3, 5, 8, 10]
LSTM_L = [12, 24]
HIDDEN, EPOCHS, LR, BATCH = 16, 300, 0.02, 64
SEEDS = [0, 1, 2, 3, 4]

WS_GRID    = [2, 3, 4, 6, 8]
ALPHA_GRID = [0.3, 0.6, 1.0]
ZETA       = 1.0

NAMES = ["AirPassengersDataset", "AusBeerDataset", "GasRateCO2Dataset",
         "MonthlyMilkDataset", "SunspotsDataset", "WineDataset",
         "WoolyDataset", "HeartRateDataset"]

OURS = ["KNN", "CDFA+KNN", "LSTM", "CDFA+LSTM"]


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name + ".csv"))
    value_col = None
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            value_col = col
            break
    if value_col is None:
        for col in df.columns:
            if pd.to_numeric(df[col], errors="coerce").notna().sum() == len(df):
                value_col = col
                break
    s = df[value_col].astype(float).reset_index(drop=True)
    if name == "SunspotsDataset":
        s = s.iloc[::4].reset_index(drop=True)
    if name == "HeartRateDataset":
        s = s.iloc[::2].reset_index(drop=True)
    return s.to_numpy()


# ============================================================
# CDFA
# ============================================================

def _locorr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    sx, sy = x.std(), y.std()
    return 0.0 if sx < 1e-12 or sy < 1e-12 else float(np.corrcoef(x, y)[0, 1])


def xi_static(f, w):
    """loCo(i) = corr(f[i:], f[:-i])"""
    f = np.asarray(f, float)
    n = len(f)
    out = np.zeros(w)
    if n < w + 3:
        return out
    for i in range(1, w + 1):
        out[i - 1] = _locorr(f[i:], f[:-i])
    return out


def phi_eq13(xi):
    """分段 min-max：正支 [0,1]、负支 [-1,0]。"""
    hi, lo = xi.max(), xi.min()
    d = hi - lo
    if d < 1e-12:
        return np.zeros(len(xi))
    return np.array([(v - lo) / d if v >= 0 else (v - hi) / d for v in xi])


def cdfa_weights(train, w):
    xi = xi_static(train, w)
    phi = phi_eq13(xi)
    if np.all(phi == 0):
        return np.ones(w) / w
    return phi


def cdfa_augment_series(train, w, alpha, noise_std=0.0, seed=0, ref=None):
    rng = np.random.default_rng(seed)
    f = np.asarray(train, float)
    n = len(f)
    f_aug = f.copy()
    if n <= w + 2:
        return f_aug
    phi = cdfa_weights(ref if ref is not None else f, w)
    sd = np.std(f); sd = sd if sd > 0 else 1.0
    for k in range(w, n - 1):
        win   = f[k - w + 1:k + 1]
        delta = np.dot(win, phi) / (w * ZETA)
        f_hat = f[k] + delta
        f_aug[k + 1] = f[k + 1] + alpha * (f_hat - f[k + 1])
        if noise_std > 0:
            f_aug[k + 1] += noise_std * sd * rng.standard_normal()
    f_aug[:w] = f[:w]
    return f_aug


# ============================================================
# 特征构造：T0/T1 两列分别 CDFA → (A0, A1)，目标 = T1
# ============================================================

def build_features(train, w=None, a=None, use_cdfa=False, noise_std=0.0, seed=0, ref=None):
    """
    返回 (F, y)：
      use_cdfa=False（基线）: F = train 单列, y = train
      use_cdfa=True（方法1）: T0=train, T1=train[1:]
                             A0 = CDFA(T0), A1 = CDFA(T1)
                             F = [A0[:len(T1)], A1]（两列，全部来自 CDFA）
                             y = T1（原始序列上移一格，作标签）
    """
    f = np.asarray(train, float)
    if not use_cdfa:
        return f[:, None], f
    T0, T1 = f, f[1:]
    A0 = cdfa_augment_series(T0, w, a, noise_std=noise_std, seed=seed, ref=ref)
    A1 = cdfa_augment_series(T1, w, a, noise_std=noise_std, seed=seed, ref=ref)
    m = len(T1)
    return np.column_stack([A0[:m], A1]), T1


def make_variant_list(train, w, a, ref=None):
    """最终训练用的特征序列列表：1 条基线增强 + 2 条多样性增强（不同 alpha/噪声）。"""
    outs = [build_features(train, w, a, True, seed=0, ref=ref)]
    outs.append(build_features(train, w, a, True, noise_std=0.02, seed=1, ref=ref))
    outs.append(build_features(train, max(w - 1, 2), 0.8 * a, True, seed=2, ref=ref))
    return outs


# ============================================================
# KNN
# ============================================================

def knn_direct_feat(F, y, H, L, k):
    """库 = F 上所有 (L 窗口 → 后 H 步 y)；查询 = F 末段 L 行；k 近邻均值。"""
    F = np.asarray(F, float); y = np.asarray(y, float)
    X, T = [], []
    for t in range(L, len(F) - H + 1):
        X.append(F[t - L:t].ravel())
        T.append(y[t:t + H])
    if not X:
        return np.full(H, y[-1])
    X, T = np.array(X, float), np.array(T, float)
    q = F[-L:].ravel()
    d = np.sqrt(((X - q) ** 2).sum(axis=1))
    kk = min(k, len(X))
    return T[np.argsort(d)[:kk]].mean(axis=0)


def tune_knn(train, H, use_cdfa):
    """验证窗（train 末尾 h 点）上调 (L, k)；use_cdfa 时再调 (ws, alpha)。"""
    n = len(train)
    best = (np.inf, None, None, None, None)
    combos = [(None, None)]
    if use_cdfa:
        combos = [(w, a) for w in WS_GRID for a in ALPHA_GRID]
    for L in L_GRID:
        if L > n - H:
            continue
        h = min(min(H, max(2, n - 10)), max(2, (n - L) // 2))
        if h < 2:
            continue
        hist, true = train[:-h], train[-h:]
        if L > len(hist) - 3:
            continue
        for w, a in combos:
            if w is not None and w + 4 > len(hist):
                continue
            F, y = build_features(hist, w, a, use_cdfa, ref=np.concatenate([hist, true]))
            nbr = len(F) - L - h + 1
            if nbr < 1:
                continue
            for k in K_GRID:
                if k > nbr:
                    continue
                mae = np.abs(knn_direct_feat(F, y, h, L, k) - true).mean()
                if mae < best[0] - 1e-12:
                    best = (mae, L, k, w, a)
    return best[1:]


def run_knn(train, test, use_cdfa):
    H = len(test)
    L, k, w, a = tune_knn(train, H, use_cdfa)
    if L is None:
        L, k = 2, 1
    F, y = (build_features(train, w, a, use_cdfa, ref=np.concatenate([train, test]))
            if use_cdfa else build_features(train))
    mae = float(np.abs(knn_direct_feat(F, y, H, L, k) - test).mean())
    return mae, (L, k, w, a)


# ============================================================
# LSTM
# ============================================================

class LSTM:
    def __init__(self, d_in, d_h, d_out, seed=0):
        rng = np.random.RandomState(seed)
        self.d_in, self.d_h, self.d_out = d_in, d_h, d_out
        k = 1.0 / np.sqrt(d_in + d_h)
        self.W = rng.uniform(-k, k, (d_in + d_h, 4 * d_h))
        self.b = np.zeros(4 * d_h)
        self.b[d_h:2 * d_h] = 1.0
        self.Wy = rng.uniform(-k, k, (d_h, d_out))
        self.by = np.zeros(d_out)
        self.m = {n: np.zeros_like(getattr(self, n)) for n in ("W", "b", "Wy", "by")}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ("W", "b", "Wy", "by")}
        self.step = 0

    def forward(self, X, cache=True):
        B, T, _ = X.shape
        hd = self.d_h
        h = np.zeros((B, hd)); c = np.zeros((B, hd)); store = []
        for t in range(T):
            z = np.concatenate([X[:, t, :], h], axis=1)
            q = z @ self.W + self.b
            i = 1 / (1 + np.exp(-np.clip(q[:, 0:hd], -30, 30)))
            f = 1 / (1 + np.exp(-np.clip(q[:, hd:2 * hd], -30, 30)))
            g = np.tanh(q[:, 2 * hd:3 * hd])
            o = 1 / (1 + np.exp(-np.clip(q[:, 3 * hd:4 * hd], -30, 30)))
            cp = c
            c = f * cp + i * g
            h = o * np.tanh(c)
            if cache:
                store.append((z, i, f, g, o, c, cp))
        self.hT, self.cache = h, store
        return h @ self.Wy + self.by

    def backward(self, X, dY):
        T = X.shape[1]; hd = self.d_h
        dW = np.zeros_like(self.W); db = np.zeros_like(self.b)
        dWy = self.hT.T @ dY; dby = dY.sum(0)
        dh = dY @ self.Wy.T
        dc = np.zeros((len(X), hd))
        for t in reversed(range(T)):
            z, i, f, g, o, c, cp = self.cache[t]
            th = np.tanh(c)
            do = dh * th
            dcc = dh * o * (1 - th * th) + dc
            di, dg, df = dcc * g, dcc * i, dcc * cp
            dc = dcc * f
            dq = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 dg * (1 - g * g), do * o * (1 - o)], axis=1)
            dW += z.T @ dq; db += dq.sum(0)
            dh = (dq @ self.W.T)[:, self.d_in:]
        return dW, db, dWy, dby

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.step += 1
        for n, gr in grads.items():
            nn_ = np.linalg.norm(gr)
            if nn_ > clip:
                gr = gr * (clip / nn_)
            self.m[n] = b1 * self.m[n] + (1 - b1) * gr
            self.v[n] = b2 * self.v[n] + (1 - b2) * gr * gr
            mh = self.m[n] / (1 - b1 ** self.step)
            vh = self.v[n] / (1 - b2 ** self.step)
            setattr(self, n, getattr(self, n) - lr * mh / (np.sqrt(vh) + eps))

    def fit(self, X, Y, epochs, lr, batch, seed):
        rng = np.random.RandomState(seed)
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                dY = 2.0 * (self.forward(X[b]) - Y[b]) / len(b)
                self.adam(dict(zip(("W", "b", "Wy", "by"), self.backward(X[b], dY))), lr)
        return self

    def predict(self, X):
        return self.forward(X, cache=False)


def lstm_mae_feat(feat_list, true, L, H, seed):
    """
    feat_list: [(F, y), ...]（1~3 条特征序列，CDFA 版为 3 条多样性增强）
    样本 = 每条 F 上滑窗 (L 步 × ch 通道) → y 后 H 步；查询 = 第 0 条 F 末段 L 行。
    """
    ch = feat_list[0][0].shape[1]
    allv = np.concatenate([np.asarray(F, float).ravel() for F, _ in feat_list] +
                          [np.asarray(y, float) for _, y in feat_list])
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    X, Y = [], []
    for F, y in feat_list:
        Fn = (np.asarray(F, float) - lo) / (hi - lo)
        yn = (np.asarray(y, float) - lo) / (hi - lo)
        for t in range(L, len(Fn) - H + 1):
            X.append(Fn[t - L:t])
            Y.append(yn[t:t + H])
    if len(X) < 5:
        return np.inf
    X, Y = np.array(X, float), np.array(Y, float)
    net = LSTM(ch, HIDDEN, Y.shape[1], seed).fit(X, Y, EPOCHS, LR, BATCH, seed)
    q = (np.asarray(feat_list[0][0], float)[-L:] - lo) / (hi - lo)
    p = net.predict(q[None])[0]
    return float(np.abs(p * (hi - lo) + lo - np.asarray(true, float)).mean())


def _val_h(n, L):
    return min(min(max(2, n // 5), max(2, n - 40)), max(2, (n - L + 1 - 5) // 2))


def run_lstm(train, test, use_cdfa):
    n, H = len(train), len(test)

    if use_cdfa:
        L0 = LSTM_L[0]
        if L0 + H > n:
            return np.nan, (None, None)
        h = _val_h(n, L0)
        hist, true = train[:-h], train[-h:]
        best1 = (np.inf, WS_GRID[0], ALPHA_GRID[0])
        for w in WS_GRID:
            if w + 4 > len(hist):
                continue
            for a in ALPHA_GRID:
                F, y = build_features(hist, w, a, True, ref=np.concatenate([hist, true]))
                m = lstm_mae_feat([(F, y)], true, L0, h, SEEDS[0])
                if np.isfinite(m) and m < best1[0] - 1e-12:
                    best1 = (m, w, a)
        w_star, a_star = best1[1], best1[2]
        best2 = (np.inf, L0)
        for L in LSTM_L:
            if L + H > n:
                continue
            h = _val_h(n, L)
            if h < 2 or n - L - h + 1 < 5:
                continue
            F, y = build_features(train[:-h], w_star, a_star, True, ref=train)
            m = lstm_mae_feat([(F, y)], train[-h:], L, h, SEEDS[0])
            if np.isfinite(m) and m < best2[0] - 1e-12:
                best2 = (m, L)
        L = best2[1]
    else:
        w_star = a_star = None
        best = (np.inf, LSTM_L[0])
        for L in LSTM_L:
            if L + H > n:
                continue
            h = _val_h(n, L)
            if h < 2 or n - L - h + 1 < 5:
                continue
            F, y = build_features(train[:-h])
            m = lstm_mae_feat([(F, y)], train[-h:], L, h, SEEDS[0])
            if np.isfinite(m) and m < best[0] - 1e-12:
                best = (m, L)
        L = best[1]

    if L + H > n or n - L - H + 1 < 5:
        return np.nan, (L, (w_star, a_star))
    feat_list = (make_variant_list(train, w_star, a_star, ref=np.concatenate([train, test]))
                 if use_cdfa else [build_features(train)])
    ms = np.array([lstm_mae_feat(feat_list, test, L, H, s) for s in SEEDS])
    ms = ms[np.isfinite(ms)]
    return (float(ms.mean()) if len(ms) else np.nan), (L, (w_star, a_star))


def collect():
    rows, params = [], []
    for name in NAMES:
        y = load(name)
        k = int(len(y) * SPLIT)
        train, test = y[:k], y[k:]
        r, p = {"Dataset": name}, {"Dataset": name}
        for tag, flag in [("KNN", False), ("CDFA+KNN", True)]:
            r[tag], prm = run_knn(train, test, flag)
            p[tag] = str(prm)
        for tag, flag in [("LSTM", False), ("CDFA+LSTM", True)]:
            r[tag], prm = run_lstm(train, test, flag)
            p[tag] = str(prm)
        r["NAIVE"] = float(np.abs(test - train[-1]).mean())
        rows.append(r); params.append(p)
        print(f"{name:<22} KNN={r['KNN']:.4f} CDFA+KNN={r['CDFA+KNN']:.4f} "
              f"LSTM={r['LSTM']:.4f} CDFA+LSTM={r['CDFA+LSTM']:.4f} NAIVE={r['NAIVE']:.4f}")
        print(f"{'':<22} params: {p['KNN']} | {p['CDFA+KNN']} | {p['LSTM']} | {p['CDFA+LSTM']}")
    return pd.DataFrame(rows), pd.DataFrame(params)


def scaled_rows(df):
    sc = {c: (df[c] / df["NAIVE"]) for c in OURS}
    am = [float(np.mean(sc[c])) for c in OURS]
    gm = [float(np.exp(np.mean(np.log(sc[c])))) for c in OURS]
    return am, gm


def print_table(df, am, gm):
    w = 36
    head = "Dataset".ljust(w) + "".join(c.rjust(12) for c in OURS) + "NAIVE".rjust(12)
    print("\n" + "=" * len(head))
    print("MAE  (8:2 direct ; method1: CDFA columns -> ML)".center(len(head)))
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for _, r in df.iterrows():
        line = str(r["Dataset"]).ljust(w)
        line += "".join(f"{r[c]:12.4f}" for c in OURS) + f"{r['NAIVE']:12.4f}"
        print(line)
    print("-" * len(head))
    for label, vals in [("Scaled MAE (Arithmetic Mean)", am), ("Scaled MAE (Geometric Mean)", gm)]:
        print(label.ljust(w) + "".join(f"{v:12.4f}" for v in vals) + f"{1.0:12.4f}")
    print("-" * len(head))
    verdict = ["√" if am[1] < am[0] else "×", "", "√" if am[3] < am[2] else "×", ""]
    print("is CDFA better than without CDFA?".ljust(w) + "".join(v.rjust(12) for v in verdict))
    print("=" * len(head))


if __name__ == "__main__":
    df, df_params = collect()
    am, gm = scaled_rows(df)
    print_table(df, am, gm)
    df_out = df.copy()
    for c in OURS + ["NAIVE"]:
        df_out[c + "_scaled"] = df_out[c] / df_out["NAIVE"]
    df_out.to_csv(os.path.join(OUT_DIR, "Method1_MAE.csv"), index=False)
    df_params.to_csv(os.path.join(OUT_DIR, "Method1_params.csv"), index=False)
    print("\nsaved: Method1_MAE.csv / Method1_params.csv")
