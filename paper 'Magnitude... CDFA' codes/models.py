"""The learners.

Two groups:

  * one-step KNN, used by the rolling protocol (stocks and public panels);
  * the direct multi-step KNN and the small NumPy LSTM used by the Darts
    pipeline.

The LSTM is plain NumPy (forward, BPTT, Adam) so there is nothing to install
beyond NumPy itself.
"""
import numpy as np

K_NEIGHBORS = 5          # k in the one-step KNN
L_GRID = [2, 3, 4, 6, 8, 12, 16, 24, 36, 48]   # Darts: look-back candidates
K_GRID = [1, 2, 3, 5, 8, 10]                   # Darts: neighbour candidates
LSTM_L = [12, 24]                              # Darts: LSTM window candidates
HIDDEN, EPOCHS, LR, BATCH = 16, 300, 0.02, 64
SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# one-step KNN (rolling protocol)
# --------------------------------------------------------------------------
def knn_next(Xtr, ytr, q, k=K_NEIGHBORS):
    # Fit on the (possibly augmented) library windows, predict the next point
    # of the query window.  Distance-weighted mean of the k nearest windows.
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    model.fit(Xtr, ytr)
    return float(model.predict(np.asarray(q)[None, :])[0])


# --------------------------------------------------------------------------
# direct multi-step KNN (Darts pipeline)
# --------------------------------------------------------------------------
def knn_direct(hist, H, L, k, aug=None):
    # Library windows hist[t-L:t] with their following H points; predict with
    # the mean of the k nearest neighbours.  aug=(phi, alpha) augments first.
    n = len(hist)
    X = [hist[t - L:t] for t in range(L, n - H + 1)]
    T = [hist[t:t + H] for t in range(L, n - H + 1)]
    if not X:
        return np.full(H, hist[-1])
    X, T = np.array(X, float), np.array(T, float)
    if aug is not None:
        from cdfa import perturb
        X = perturb(X, aug[0], aug[1])
    q = np.asarray(hist, float)[-L:]
    d = np.sqrt(((X - q) ** 2).sum(axis=1))
    return T[np.argsort(d)[:k]].mean(axis=0)


def tune_knn(train, H, use_cdfa):
    # Pick (L, k) on the tail of train only: hold out the last h points, take
    # the pair with the smallest backtest MAE.
    from cdfa import ALPHA, corr_cdfa_weights
    n = len(train)
    best = (np.inf, None, None)
    for L in L_GRID:
        if L > n - H:
            continue
        h = min(min(H, max(2, n - 10)), max(2, (n - L) // 2))
        if h < 2:
            continue
        hist, true = train[:-h], train[-h:]
        if L > len(hist) - 3 or len(hist) - L - h + 1 < 1:
            continue
        aug = (corr_cdfa_weights(hist, L), ALPHA) if use_cdfa else None
        for k in K_GRID:
            if k > len(hist) - L:
                continue
            mae = np.abs(knn_direct(hist, h, L, k, aug) - true).mean()
            if mae < best[0] - 1e-12:
                best = (mae, L, k)
    return best[1], best[2]


# --------------------------------------------------------------------------
# small LSTM (Darts pipeline)
# --------------------------------------------------------------------------
class LSTM:
    # Single-layer vanilla LSTM: four gates, one output layer, Adam training.
    def __init__(self, d_in, d_h, d_out, seed=0):
        rng = np.random.RandomState(seed)
        self.d_in, self.d_h, self.d_out = d_in, d_h, d_out
        k = 1.0 / np.sqrt(d_in + d_h)
        self.W = rng.uniform(-k, k, (d_in + d_h, 4 * d_h))
        self.b = np.zeros(4 * d_h)
        self.b[d_h:2 * d_h] = 1.0                  # forget gate starts open
        self.Wy = rng.uniform(-k, k, (d_h, d_out))
        self.by = np.zeros(d_out)
        self.m = {n: np.zeros_like(getattr(self, n)) for n in ('W', 'b', 'Wy', 'by')}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ('W', 'b', 'Wy', 'by')}
        self.step = 0

    def forward(self, X, cache=True):
        # X: (batch, time, d_in).  Returns the last hidden state mapped to d_out.
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
        # Backprop through time; needs the cache left by forward.
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
        # One Adam step per parameter, with a gradient-norm clip.
        self.step += 1
        for n, gr in grads.items():
            nn = np.linalg.norm(gr)
            if nn > clip:
                gr = gr * (clip / nn)
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
                self.adam(dict(zip(('W', 'b', 'Wy', 'by'),
                                   self.backward(X[b], dY))), lr)
        return self

    def predict(self, X):
        return self.forward(X, cache=False)


def lstm_mae(hist, true, L, H, use_cdfa, seed):
    # Train on hist, score on true (length H), MAE in the original units.
    from cdfa import ALPHA, corr_cdfa_weights, perturb
    aug = (corr_cdfa_weights(hist, L), ALPHA) if use_cdfa else None
    lo, hi = float(hist.min()), float(hist.max())
    if hi <= lo:
        hi = lo + 1.0
    sn = (np.asarray(hist, float) - lo) / (hi - lo)
    X = [sn[t - L:t] for t in range(L, len(sn) - H + 1)]
    Y = [sn[t:t + H] for t in range(L, len(sn) - H + 1)]
    if len(X) < 5:
        return np.inf
    X, Y = np.array(X, float), np.array(Y, float)
    if aug is not None:
        X = perturb(X, aug[0], aug[1])
    net = LSTM(1, HIDDEN, Y.shape[1], seed).fit(X[:, :, None], Y, EPOCHS, LR,
                                                BATCH, seed)
    p = net.predict(sn[-L:][None, :, None])[0]
    return float(np.abs(p * (hi - lo) + lo - true).mean())


def run_lstm(train, test, use_cdfa):
    # Pick L on the tail of train, then average the test MAE over the seeds.
    n, H = len(train), len(test)
    best = (np.inf, LSTM_L[0])
    for L in LSTM_L:
        if L + H > n or n - L - H + 1 < 5:
            continue
        h = min(min(H, max(2, n - 40)), max(2, (n - L + 1 - 5) // 2))
        if h < 2 or n - L - h + 1 < 5:
            continue
        m = lstm_mae(train[:-h], train[-h:], L, h, use_cdfa, SEEDS[0])
        if np.isfinite(m) and m < best[0] - 1e-12:
            best = (m, L)
    L = best[1]
    if L + H > n or n - L - H + 1 < 5:
        return np.nan
    ms = np.array([lstm_mae(train, test, L, H, use_cdfa, s) for s in SEEDS])
    ms = ms[np.isfinite(ms)]
    return float(ms.mean()) if len(ms) else np.nan
