"""Window-level augmentations used as comparators on the Stocks dataset.

Four published transforms. Each one rewrites a single look-back window x and
keeps its length T. The two frequency transforms also take the next value y,
so the window and its target stay consistent.

    window_warp      window warping      (Le Guennec et al., 2016)
    magnitude_warp   magnitude warping   (Um et al., 2017)
    freq_mask        FrAug, masking      (Chen et al., 2024)
    freq_mix         FrAug, mixing       (Chen et al., 2024)

NumPy only. SciPy is optional and supplies the cubic spline.
"""

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except ImportError:                       # fall back to linear interpolation
    CubicSpline = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _at(x, t):
    """Read x at fractional positions t (linear interpolation)."""
    return np.interp(t, np.arange(len(x), dtype=float), x)


def _randint(rng, lo, hi):
    """Uniform integer in [lo, hi); accepts both NumPy RNG APIs."""
    if hasattr(rng, "integers"):
        return int(rng.integers(lo, hi))
    return int(rng.randint(lo, hi))


def _band(n, rng, mu, protect_dc=True):
    """Random contiguous band of an rFFT spectrum of length n."""
    k = max(1, int(round(mu * n)))
    k = min(k, n - 1) if protect_dc else min(k, n)
    lo = 1 if protect_dc else 0
    start = _randint(rng, lo, n - k + 1)
    return start, start + k


# ---------------------------------------------------------------------------
# time-domain transforms
# ---------------------------------------------------------------------------
def window_warp(x, rng, frac=0.3, lo=0.5, hi=2.0):
    """Stretch or compress one random slice, then resample to length T."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    L = max(2, int(round(frac * T)))
    a = _randint(rng, 0, T - L + 1)
    s = float(rng.uniform(lo, hi))

    T2 = T + int(round(L * (s - 1.0)))
    j = np.arange(T2, dtype=float)
    end = a + L * s

    idx = j.copy()
    inside = (j >= a) & (j < end)
    idx[inside] = a + (j[inside] - a) / s
    idx[j >= end] = a + L + (j[j >= end] - end)
    idx = np.clip(idx, 0.0, T - 1.0)

    warped = _at(x, idx)
    return _at(warped, np.linspace(0.0, T2 - 1.0, T))


def magnitude_warp(x, rng, sigma=0.2, n_knots=4):
    """Multiply the window by a smooth random curve around 1."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    knots = np.linspace(0.0, T - 1.0, n_knots)
    vals = rng.normal(0.0, sigma, n_knots)
    if CubicSpline is not None:
        curve = CubicSpline(knots, vals, bc_type="natural")(np.arange(T))
    else:
        curve = np.interp(np.arange(T, dtype=float), knots, vals)
    return x * (1.0 + curve)


# ---------------------------------------------------------------------------
# frequency-domain transforms (FrAug)
# ---------------------------------------------------------------------------
def freq_mask(x, y, rng, mu=0.1, protect_dc=True):
    """Zero one random frequency band of [x, y]."""
    s = np.concatenate([np.asarray(x, dtype=float), [float(y)]])
    S = np.fft.rfft(s)
    a, b = _band(len(S), rng, mu, protect_dc)
    S[a:b] = 0.0
    out = np.fft.irfft(S, n=len(s))
    return out[:-1], float(out[-1])


def freq_mix(x, y, x2, y2, rng, mu=0.1, protect_dc=True):
    """Swap one random frequency band with that of another sample."""
    s1 = np.concatenate([np.asarray(x, dtype=float), [float(y)]])
    s2 = np.concatenate([np.asarray(x2, dtype=float), [float(y2)]])
    S1, S2 = np.fft.rfft(s1), np.fft.rfft(s2)
    a, b = _band(len(S1), rng, mu, protect_dc)
    S1[a:b] = S2[a:b]
    out = np.fft.irfft(S1, n=len(s1))
    return out[:-1], float(out[-1])
