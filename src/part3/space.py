"""Acceptability degree spaces of the weighted h-categorizer.

For a framework (A, R), the space is S = { HCat(w) : w in [0,1]^|A| }. We
sample it by drawing uniform weight vectors, then study its convex hull and
the density of the sampled points.

Inverse map. Since Deg(a) (1 + sum_{b->a} Deg(b)) = w(a), the weights are
recovered from the degrees by g(x) = x * (1 + x @ M). Because the fixed point
is unique, x belongs to S iff x >= 0 and g(x) <= 1, and the density of the
points x_i (w_i uniform) is |det Jg(x)| on S. This gives an exact reference
for the hull (true volume of S) and for the heat-maps.
"""

import itertools

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .hcategorizer import h_categorizer_batch


# ---------------------------------------------------------------- sampling

def sample_space(af, n_samples=100_000, seed=0, epsilon=1e-10):
    """Draw n_samples weight vectors in [0,1]^|A| and map them to degrees."""
    rng = np.random.default_rng(seed)
    W = rng.random((n_samples, len(af)))
    X, iterations = h_categorizer_batch(af, W, epsilon, return_iterations=True)
    return W, X, iterations


# ------------------------------------------------------- exact description

def inverse_weights(af, X):
    """g(x) = x * (1 + x @ M): the weights that produce the degrees x."""
    return X * (1.0 + X @ af.matrix)


def in_space(af, X, tol=1e-12):
    """Membership test for S (exact characterisation, see module docstring)."""
    return np.all(X >= -tol, axis=-1) & np.all(inverse_weights(af, X) <= 1 + tol, axis=-1)


def jacobian_det(af, X):
    """det of the Jacobian of g at each row of X.

    dg_a/dx_c = delta_ac (1 + s_a) + x_a M[c, a], with s = x @ M.
    """
    X = np.atleast_2d(X)
    M = af.matrix
    S = X @ M
    J = np.einsum("ka,ca->kac", X, M)
    idx = np.arange(len(af))
    J[:, idx, idx] += 1.0 + S
    return np.linalg.det(J)


def exact_density(af, X):
    """Density of x = HCat(w), w ~ U[0,1]^n, evaluated at X (0 outside S)."""
    return np.where(in_space(af, X), np.abs(jacobian_det(af, X)), 0.0)


def true_volume(af, X_samples):
    """Volume of S estimated without bias as E_w[1 / |det Jg(HCat(w))|]."""
    inv = 1.0 / np.abs(jacobian_det(af, X_samples))
    return inv.mean(), inv.std(ddof=1) / np.sqrt(len(inv))


# -------------------------------------------------------------- convex hull

def convex_hull(X, max_dim=6):
    """Convex hull of the points (None if too high-dimensional or degenerate)."""
    n = X.shape[1]
    if n == 1:
        return None
    if n > max_dim:
        return None
    try:
        return ConvexHull(X, qhull_options="Qx" if n > 4 else "")
    except QhullError:
        return None


def hull_volume(X, hull=None):
    if X.shape[1] == 1:
        return float(X.max() - X.min())
    hull = hull if hull is not None else convex_hull(X)
    return float(hull.volume) if hull is not None else float("nan")


def hull_gap(af, hull, n_probe=200_000, seed=123):
    """Fraction of the hull volume that lies outside S (empty space of the hull).

    Points are drawn uniformly in the hull's bounding box, kept if inside the
    hull, then tested for membership in S.
    """
    rng = np.random.default_rng(seed)
    lo, hi = hull.points[hull.vertices].min(0), hull.points[hull.vertices].max(0)
    P = lo + (hi - lo) * rng.random((n_probe, hull.points.shape[1]))
    A, b = hull.equations[:, :-1].T, hull.equations[:, -1]
    inside_hull = np.concatenate([np.all(C @ A + b <= 1e-12, axis=1)
                                  for C in np.array_split(P, max(1, len(P) // 2000))])
    P = P[inside_hull]
    return float(1.0 - in_space(af, P).mean()), len(P)


# ------------------------------------------------------------------ density

def density_2d(X, i, j, bins=40, rng_box=((0, 1), (0, 1))):
    """Normalised 2-D histogram (probability mass per cell) of columns i, j."""
    H, xe, ye = np.histogram2d(X[:, i], X[:, j], bins=bins, range=rng_box)
    return H / H.sum(), xe, ye


def exact_cell_mass_2d(af, bins=40, sub=16):
    """Exact probability mass per cell for a 2-argument framework (midpoint rule)."""
    assert len(af) == 2
    m = bins * sub
    c = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(c, c, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel()], axis=1)
    dens = exact_density(af, P).reshape(m, m) / m**2
    return dens.reshape(bins, sub, bins, sub).sum(axis=(1, 3))


def total_variation(P, Q):
    return 0.5 * np.abs(P - Q).sum()


def jensen_shannon(P, Q):
    P, Q = P.ravel(), Q.ravel()
    Mx = 0.5 * (P + Q)

    def kl(A, B):
        mask = A > 0
        return np.sum(A[mask] * np.log2(A[mask] / B[mask]))

    return np.sqrt(0.5 * kl(P, Mx) + 0.5 * kl(Q, Mx))


def high_density_region(H, quantile_mass=0.5):
    """Smallest set of cells holding `quantile_mass` of the probability."""
    flat = H.ravel()
    order = np.argsort(flat)[::-1]
    k = np.searchsorted(np.cumsum(flat[order]), quantile_mass) + 1
    mask = np.zeros(flat.shape, bool)
    mask[order[:k]] = True
    return mask.reshape(H.shape)


def jaccard(A, B):
    return (A & B).sum() / max((A | B).sum(), 1)


def pairs(n):
    return list(itertools.combinations(range(n), 2))


def resampling_study(af, n_samples=100_000, n_repeats=10, bins=40, epsilon=1e-10, base_seed=1000):
    """Stability of hull and heat-maps across independent re-samplings."""
    hists, volumes, hdr = [], [], []
    for r in range(n_repeats):
        _, X, _ = sample_space(af, n_samples, seed=base_seed + r, epsilon=epsilon)
        hs = [density_2d(X, i, j, bins)[0] for i, j in pairs(len(af))]
        hists.append(hs)
        hdr.append([high_density_region(h) for h in hs])
        volumes.append(hull_volume(X))
    tv, js, jac, maxrel = [], [], [], []
    for r, s in itertools.combinations(range(n_repeats), 2):
        for p in range(len(pairs(len(af)))):
            P, Q = hists[r][p], hists[s][p]
            tv.append(total_variation(P, Q))
            js.append(jensen_shannon(P, Q))
            jac.append(jaccard(hdr[r][p], hdr[s][p]))
            dense = (P + Q) / 2 > 1.0 / bins**2  # cells above the uniform level
            if dense.any():
                maxrel.append(np.max(np.abs(P - Q)[dense] / ((P + Q)[dense] / 2)))
    volumes = np.array(volumes)
    return {
        "tv_mean": float(np.mean(tv)), "tv_max": float(np.max(tv)),
        "js_mean": float(np.mean(js)),
        "hdr50_jaccard_mean": float(np.mean(jac)), "hdr50_jaccard_min": float(np.min(jac)),
        "dense_cell_max_rel_diff": float(np.max(maxrel)) if maxrel else float("nan"),
        "hull_volume_mean": float(volumes.mean()),
        "hull_volume_rel_std": float(volumes.std(ddof=1) / volumes.mean()),
        "hists_first_two": (hists[0], hists[1]),
    }


def tv_vs_sample_size(af, sizes, n_repeats=5, bins=40, reference_size=2_000_000, seed=7):
    """TV distance between a sample's heat-map and a reference heat-map, vs N.

    For |A| = 2 the reference is exact (integrated density); otherwise it is
    a very large independent sample. Only the (0, 1) projection is used.
    """
    if len(af) == 2:
        ref = exact_cell_mass_2d(af, bins)
    else:
        _, Xr, _ = sample_space(af, reference_size, seed=seed)
        ref = density_2d(Xr, 0, 1, bins)[0]
    out = {}
    for N in sizes:
        vals = []
        for r in range(n_repeats):
            _, X, _ = sample_space(af, N, seed=seed + 100 + r)
            vals.append(total_variation(density_2d(X, 0, 1, bins)[0], ref))
        out[N] = (float(np.mean(vals)), float(np.std(vals)))
    return out
