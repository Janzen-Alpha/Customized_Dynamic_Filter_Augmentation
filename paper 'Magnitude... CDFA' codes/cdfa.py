"""CDFA: the augmentation method, plus two filters it is compared with.

The weight constructions follow the method's equations:

    Eq. (3)  CCF          the localized-correlation filter
    Eq. (5)  volatility   psi_i = (sigma_i / mean sigma) ** gamma, unit mean
    Eq. (6)  redistribution  d' = d + 1/2 * alpha * Delta(phi * d)

The final block holds a correlation-based weight construction, used by the
Darts pipeline.
"""
import numpy as np

# --------------------------------------------------------------------------
# constants shared by every weight construction
# --------------------------------------------------------------------------
W = 15           # number of weight positions (= learner input length t)
ALPHA = 0.3      # single global magnitude of the perturbation
VOL_WINDOW = 10  # m_v: trailing block used for each local volatility
VOL_POWER = 1.0  # gamma in Eq. (5)
CCF_BLOCK = 300  # CCF reads the last 300 returns (= training window)


def minmax(s):
    # Scale to [0, 1]; a constant stretch just becomes zeros.
    s = np.asarray(s, float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(s)


def unit_mean(phi):
    # Rescale so the mean |weight| is 1: the dose of the perturbation is then
    # fixed by alpha alone and does not drift with the dataset.
    phi = np.asarray(phi, float)
    m = np.mean(np.abs(phi))
    return phi / m if m > 1e-12 else np.ones_like(phi)


# --------------------------------------------------------------------------
# Eq. (5): the volatility profile
# --------------------------------------------------------------------------
def volatility_profile(series, w=W, k=None, vol_window=VOL_WINDOW,
                       power=VOL_POWER):
    # series : the normalised series; only series[:k+1] is used, k is the last
    #          observed point before the evaluated period
    # returns psi of length w, unit mean
    k = len(series) - 1 if k is None else k
    x = np.asarray(series[:k + 1], float)
    if len(x) < w + vol_window + 2:
        return np.ones(w)                    # not enough history: flat profile
    r = np.diff(x)                           # first differences d_i = x_i+1-x_i
    vols = np.empty(w)
    for i in range(w):
        # block of m_v returns ending i steps before the last observation:
        # position 0 is the newest increment, position w-1 the oldest
        end = len(r) - i
        start = max(0, end - vol_window)
        block = r[start:end]
        vols[i] = np.std(block) if len(block) > 1 else 0.0
    m = vols.mean()                          # mean sigma of the window
    if m < 1e-12:
        return np.ones(w)
    return (vols / m) ** power               # sigma_i / mean sigma, raised to gamma


def cdfa_weights(series, w=W, k=None):
    # The CDFA weight vector: Eq. (5) followed by the unit-mean rescaling.
    return unit_mean(volatility_profile(series, w, k))


# --------------------------------------------------------------------------
# comparison filters
# --------------------------------------------------------------------------
def preset_b_weights(w=W):
    # Preset curve B: a linear ramp.  It is NOT rescaled to unit mean, so its
    # effective dose is smaller than CDFA's.
    return np.array([(i + 1) / w for i in range(w)])


def ccf_weights(series, w=W, k=None, block=CCF_BLOCK):
    # Eq. (3): for lag 1..w take the correlation between the returns and their
    # own lag; normalise by the largest modulus; then put it on the same dose
    # as CDFA with the unit-mean rescaling.
    k = len(series) - 1 if k is None else k
    r = np.diff(np.asarray(series[:k + 1], float))
    if len(r) > block:
        r = r[-block:]
    if len(r) < w + 20:
        return np.ones(w)
    xi = np.empty(w)
    for i in range(w):
        lag = i + 1
        a, b = r[:-lag], r[lag:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            xi[i] = 0.0
        else:
            xi[i] = np.corrcoef(a, b)[0, 1]
    m = np.max(np.abs(xi))
    if m < 1e-12:
        return np.ones(w)
    return unit_mean(xi / m)


# --------------------------------------------------------------------------
# Eq. (6): the redistribution rule
# --------------------------------------------------------------------------
def perturb(X, phi, alpha=ALPHA):
    # Split alpha * phi_i * d_i in half between the two points the increment
    # joins.  Window length is untouched, the endpoints hardly move.
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[None, :]
    m = min(len(phi), X.shape[1] - 1)
    if m <= 0:
        return X.copy()
    d = np.diff(X, axis=1)                   # d_i
    a = np.zeros_like(X)
    a[:, :m] = d[:, :m] * np.asarray(phi[:m]) * alpha   # alpha * phi_i * d_i
    adj = 0.5 * a                            # half goes to point i ...
    adj[:, 1:] += 0.5 * a[:, :-1]            # ... and half to point i+1
    return X + adj


# --------------------------------------------------------------------------
# generic augmentation controls (baselines only)
# --------------------------------------------------------------------------
def jitter(X, sigma, rng):
    # White noise scaled by the average size of the increments.
    X = np.asarray(X, float)
    sd = np.abs(np.diff(X, axis=1)).mean() * sigma
    return X + rng.normal(0, sd, X.shape)


def scaled_noise(X, sigma, rng):
    # White noise scaled by each window's own standard deviation.
    X = np.asarray(X, float)
    sd = X.std(axis=1, keepdims=True)
    return X + rng.normal(0, 1, X.shape) * sd * sigma


JITTER_SIGMA = 0.02
NOISE_SIGMA = 0.05


def augment(X, method, phi=None, rng=None):
    # One entry point so the evaluation loop does not care which method runs.
    if method == 'baseline':
        return X
    if method == 'jitter':
        return jitter(X, JITTER_SIGMA, rng)
    if method == 'scaled_noise':
        return scaled_noise(X, NOISE_SIGMA, rng)
    return perturb(X, phi)                   # cdfa / preset_b / ccf


# --------------------------------------------------------------------------
# The correlation-based weight construction used by the Darts pipeline.
# --------------------------------------------------------------------------
def xi_returns(s, w, k):
    # Correlation of the returns against their own lag, lag = 1..w.
    r = np.diff(np.asarray(s[:k + 1], float))
    out = np.zeros(w)
    if len(r) < w + 2:
        return out
    for lag in range(1, w + 1):
        if len(r) <= lag + 1:
            continue
        x, y = r[lag:], r[:-lag]
        xm, ym = x.mean(), y.mean()
        num = ((x - xm) * (y - ym)).sum()
        den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
        out[lag - 1] = num / den if den > 1e-12 else 0.0
    return out


def phi_signed(xi):
    # Piecewise min-max: each sign branch fills its own half of [-1, 1].
    hi, lo = xi.max(), xi.min()
    d = hi - lo
    if d < 1e-12:
        return np.zeros(len(xi))
    return np.array([(v - lo) / d if v >= 0 else (v - hi) / d for v in xi])


def corr_cdfa_weights(hist, w):
    # CDFA weights from the correlation field; a short history falls back to
    # the uniform kernel 1/w.
    if len(hist) - 1 < w:
        return np.ones(w) / w
    return phi_signed(xi_returns(minmax(hist), w, len(hist) - 1))
