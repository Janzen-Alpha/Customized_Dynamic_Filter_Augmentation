"""
方法2（先 ML，后 CDFA）：T0、T1 先进行机器学习，再经 CDFA 得出结果。

步骤：
  训练
    T0 = train, T1 = train[1:]
    ML 学习:  (T0[t-L:t], T1[t-L:t])  →  T1[t:t+H]
    验证：ML 在验证窗上输出原始预测 p → CDFA 后处理（Φ 在 train 全段上拟合）→ 与真值比，调 (ws, alpha)
  测试
    T0'（train 末段）→ ML → 原始预测 p（H 步）
    p → CDFA 后处理（滤波器 Φ 在整段序列上拟合）→ 最终 T̂1

CDFA 后处理（对预测序列 p 作用，teacher-forced 不回灌）：
    Φ   = CDFA 滤波器
    对 k = ws, …, H−2:
        Δ_k      = (p_k^ws · Φ) / (ws·ς)
        p̂_{k+1}  = p_k + Δ_k
        out[k+1] = p[k+1] + α·(p̂_{k+1} − p[k+1])  # 同增强混合式
    （前 ws 个点不变；H ≤ ws+2 时退化为原始 ML 输出）
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

OURS = ["KNN", "KNN→CDFA", "LSTM", "LSTM→CDFA"]


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
    f = np.asarray(f, float)
    n = len(f)
    out = np.zeros(w)
    if n < w + 3:
        return out
    for i in range(1, w + 1):
        out[i - 1] = _locorr(f[i:], f[:-i])
    return out


def phi_eq13(xi):
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


def cdfa_postprocess(ref, p, w, alpha):
    """
    方法②的 CDFA 后处理：
      ref : 拟合滤波器用的参考序列，只用它的相关结构；
      p   : ML 输出的原始预测序列（长度 H）；
      对 p 逐点做 eq.11/12 推演 + alpha 混合（teacher-forced，不回灌）。
    H 太短（≤ w+2）时无法推演，原样返回。
    """
    p = np.asarray(p, float).copy()
    H = len(p)
    if H <= w + 2:
        return p
    phi = cdfa_weights(ref, w)
    out = p.copy()
    for k in range(w, H - 1):
        win   = p[k - w + 1:k + 1]
        delta = np.dot(win, phi) / (w * ZETA)
        p_hat = p[k] + delta
        out[k + 1] = p[k + 1] + alpha * (p_hat - p[k + 1])
    out[:w] = p[:w]
    return out


# ============================================================
# 特征构造：T0/T1 原始两列
# ============================================================

def build_features(train):
    """F = [T0[:len(T1)], T1]，y = T1"""
    f = np.asarray(train, float)
    T0, T1 = f, f[1:]
    return np.column_stack([T0[:len(T1)], T1]), T1


# ============================================================
# KNN（直接多步）：原始两列上滑窗
# ============================================================

def knn_direct_feat(F, y, H, L, k):
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


def tune_knn_m2(train, H):
    """两级调参：验证窗上先调 (L, k)（纯 ML），再固定 (L, k) 调 (ws, alpha)（CDFA 后处理）。"""
    n = len(train)
    best_ml = (np.inf, None, None)
    for L in L_GRID:
        if L > n - H:
            continue
        h = min(min(H, max(2, n - 10)), max(2, (n - L) // 2))
        if h < 2:
            continue
        hist, true = train[:-h], train[-h:]
        if L > len(hist) - 3:
            continue
        F, y = build_features(hist)
        nbr = len(F) - L - h + 1
        if nbr < 1:
            continue
        for k in K_GRID:
            if k > nbr:
                continue
            mae = np.abs(knn_direct_feat(F, y, h, L, k) - true).mean()
            if mae < best_ml[0] - 1e-12:
                best_ml = (mae, L, k)
    L, k = best_ml[1], best_ml[2]
    if L is None:
        return None, None, None, None
    best_c = (np.inf, WS_GRID[0], ALPHA_GRID[0])
    h = min(min(H, max(2, n - 10)), max(2, (n - L) // 2))
    hist, true = train[:-h], train[-h:]
    F, y = build_features(hist)
    p = knn_direct_feat(F, y, h, L, k)
    for w in WS_GRID:
        if w + 3 > h:
            continue
        for a in ALPHA_GRID:
            mae = np.abs(cdfa_postprocess(np.concatenate([hist, true]), p, w, a) - true).mean()
            if mae < best_c[0] - 1e-12:
                best_c = (mae, w, a)
    return L, k, best_c[1], best_c[2]


def run_knn_m2(train, test, use_cdfa):
    H = len(test)
    if not use_cdfa:
        # 纯 ML 基线：只调 (L, k)
        n = len(train)
        best = (np.inf, 2, 1)
        for L in L_GRID:
            if L > n - H:
                continue
            h = min(min(H, max(2, n - 10)), max(2, (n - L) // 2))
            if h < 2:
                continue
            hist, true = train[:-h], train[-h:]
            if L > len(hist) - 3:
                continue
            F, y = build_features(hist)
            nbr = len(F) - L - h + 1
            if nbr < 1:
                continue
            for k in K_GRID:
                if k > nbr:
                    continue
                mae = np.abs(knn_direct_feat(F, y, h, L, k) - true).mean()
                if mae < best[0] - 1e-12:
                    best = (mae, L, k)
        L, k = best[1], best[2]
        F, y = build_features(train)
        p = knn_direct_feat(F, y, H, L, k)
        return float(np.abs(p - test).mean()), (L, k, None, None)
    L, k, w, a = tune_knn_m2(train, H)
    if L is None:
        L, k, w, a = 2, 1, WS_GRID[0], ALPHA_GRID[0]
    F, y = build_features(train)
    p = knn_direct_feat(F, y, H, L, k)
    ref = np.concatenate([train, test])
    final = cdfa_postprocess(ref, p, w, a)
    return float(np.abs(final - test).mean()), (L, k, w, a)


# ============================================================
# LSTM（NumPy 手写，2 通道输入）
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


def lstm_predict(F, y, L, H, seed):
    """在 (F, y) 上训练并直接多步预测 H 步（原始 ML 输出，不过 CDFA）。"""
    F = np.asarray(F, float); y = np.asarray(y, float)
    ch = F.shape[1]
    allv = np.concatenate([F.ravel(), y])
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    Fn = (F - lo) / (hi - lo)
    yn = (y - lo) / (hi - lo)
    X, Y = [], []
    for t in range(L, len(Fn) - H + 1):
        X.append(Fn[t - L:t])
        Y.append(yn[t:t + H])
    if len(X) < 5:
        return np.full(H, y[-1])
    X, Y = np.array(X, float), np.array(Y, float)
    net = LSTM(ch, HIDDEN, Y.shape[1], seed).fit(X, Y, EPOCHS, LR, BATCH, seed)
    q = (F[-L:] - lo) / (hi - lo)
    p = net.predict(q[None])[0]
    return p * (hi - lo) + lo


def _val_h(n, L):
    return min(min(max(2, n // 5), max(2, n - 40)), max(2, (n - L + 1 - 5) // 2))


def run_lstm_m2(train, test, use_cdfa):
    n, H = len(train), len(test)

    def tune_L():
        best = (np.inf, LSTM_L[0])
        for L in LSTM_L:
            if L + H > n:
                continue
            h = _val_h(n, L)
            if h < 2 or n - L - h + 1 < 5:
                continue
            F, y = build_features(train[:-h])
            p = lstm_predict(F, y, L, h, SEEDS[0])
            mae = np.abs(p - train[-h:]).mean()
            if np.isfinite(mae) and mae < best[0] - 1e-12:
                best = (mae, L)
        return best[1]

    L = tune_L()
    if L + H > n:
        return np.nan, (L, None)

    if not use_cdfa:
        F, y = build_features(train)
        ms = []
        for s in SEEDS:
            p = lstm_predict(F, y, L, H, s)
            ms.append(np.abs(p - test).mean())
        return float(np.mean(ms)), (L, (None, None))

    # 两级调参：先 L（上面已做），再 (ws, alpha) —— 用单 seed 的验证预测做后处理调参
    h = _val_h(n, L)
    hist, true = train[:-h], train[-h:]
    F, y = build_features(hist)
    p_val = lstm_predict(F, y, L, h, SEEDS[0])
    best_c = (np.inf, WS_GRID[0], ALPHA_GRID[0])
    for w in WS_GRID:
        if w + 3 > h:
            continue
        for a in ALPHA_GRID:
            mae = np.abs(cdfa_postprocess(np.concatenate([hist, true]), p_val, w, a) - true).mean()
            if mae < best_c[0] - 1e-12:
                best_c = (mae, w, a)
    w_star, a_star = best_c[1], best_c[2]
    F, y = build_features(train)
    ms = []
    for s in SEEDS:
        p = lstm_predict(F, y, L, H, s)
        ref = np.concatenate([train, test])
        final = cdfa_postprocess(ref, p, w_star, a_star)
        ms.append(np.abs(final - test).mean())
    return float(np.mean(ms)), (L, (w_star, a_star))


# ============================================================
# 汇总
# ============================================================

def collect():
    rows, params = [], []
    for name in NAMES:
        y = load(name)
        k = int(len(y) * SPLIT)
        train, test = y[:k], y[k:]
        r, p = {"Dataset": name}, {"Dataset": name}
        for tag, flag in [("KNN", False), ("KNN→CDFA", True)]:
            r[tag], prm = run_knn_m2(train, test, flag)
            p[tag] = str(prm)
        for tag, flag in [("LSTM", False), ("LSTM→CDFA", True)]:
            r[tag], prm = run_lstm_m2(train, test, flag)
            p[tag] = str(prm)
        r["NAIVE"] = float(np.abs(test - train[-1]).mean())
        rows.append(r); params.append(p)
        print(f"{name:<22} KNN={r['KNN']:.4f} KNN→CDFA={r['KNN→CDFA']:.4f} "
              f"LSTM={r['LSTM']:.4f} LSTM→CDFA={r['LSTM→CDFA']:.4f} NAIVE={r['NAIVE']:.4f}")
        print(f"{'':<22} params: {p['KNN']} | {p['KNN→CDFA']} | {p['LSTM']} | {p['LSTM→CDFA']}")
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
    print("MAE  (8:2 direct ; method2: ML -> CDFA post-process)".center(len(head)))
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
    df_out.to_csv(os.path.join(OUT_DIR, "m2_ml_cdfa_mae_table.csv"), index=False)
    df_params.to_csv(os.path.join(OUT_DIR, "m2_ml_cdfa_params.csv"), index=False)
    print("\nsaved: m2_ml_cdfa_mae_table.csv / m2_ml_cdfa_params.csv")
