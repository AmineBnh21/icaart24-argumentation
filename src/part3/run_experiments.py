"""Part 3 experiments: acceptability degree spaces of the weighted h-categorizer.

Usage (from the repository root):
    python -m src.part3.run_experiments                      # full catalogue
    python -m src.part3.run_experiments --af "a,b,c : a>b, b>c, c>a"
    python -m src.part3.run_experiments --quick              # small N, for a smoke test

Figures and tables are written to results/part3/.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from . import frameworks as fw  # noqa: E402
from . import space as sp  # noqa: E402
from .hcategorizer import h_categorizer_batch  # noqa: E402

OUT = Path("results/part3")
CMAP = "magma"
BINS = 40


def _hull_outline(ax, P, **kw):
    """Draw the 2-D convex hull of points P (projection of the n-D hull)."""
    h = ConvexHull(P)
    v = np.append(h.vertices, h.vertices[0])
    ax.plot(P[v, 0], P[v, 1], **kw)


def _boundary_2d(ax, af, i=0, j=1):
    """True boundary of S for a 2-argument framework (level set of g)."""
    m = 400
    c = np.linspace(0, 1, m)
    gx, gy = np.meshgrid(c, c, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    F = (sp.inverse_weights(af, P).max(1) - 1).reshape(m, m)
    ax.contour(gx, gy, F, levels=[0], colors="deepskyblue", linewidths=1.5, linestyles="--")


def _heatmap(ax, X, i, j, af, vmax=None):
    H, xe, ye = sp.density_2d(X, i, j, BINS)
    dens = H * BINS**2  # probability per unit area
    im = ax.imshow(dens.T, origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    _hull_outline(ax, X[:, [i, j]], color="cyan", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return im


def plot_2d(af, X, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    a, b = af.arguments
    sub = X[np.random.default_rng(0).choice(len(X), min(8000, len(X)), replace=False)]
    ax = axes[0]
    ax.scatter(sub[:, 0], sub[:, 1], s=1, alpha=0.4, color="tab:gray")
    _hull_outline(ax, X, color="crimson", lw=2, label="convex hull")
    _boundary_2d(ax, af)
    ax.plot([], [], "--", color="deepskyblue", label="true boundary of S")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("points $x_i$ (8k shown) and convex hull")

    vmax = np.percentile(sp.density_2d(X, 0, 1, BINS)[0] * BINS**2, 99.5)
    im = _heatmap(axes[1], X, 0, 1, af, vmax=vmax)
    axes[1].set_title(f"empirical density, N={len(X):,}")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    m = 300
    c = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(c, c, indexing="ij")
    D = sp.exact_density(af, np.stack([gx.ravel(), gy.ravel()], 1)).reshape(m, m)
    D = np.ma.masked_equal(D, 0)
    im = axes[2].imshow(D.T, origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, vmax=vmax, aspect="equal")
    axes[2].set_title(r"exact density $|\det J_g(x)|$")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel(f"Deg({a})")
        ax.set_ylabel(f"Deg({b})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(repr(af))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_3d(af, X, hull, path):
    fig = plt.figure(figsize=(17, 5))
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    tri = Poly3DCollection([X[s] for s in hull.simplices], alpha=0.12, facecolor="crimson",
                           edgecolor="crimson", linewidth=0.15)
    ax.add_collection3d(tri)
    sub = X[np.random.default_rng(0).choice(len(X), min(6000, len(X)), replace=False)]
    d = sp.exact_density(af, sub)
    sc = ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=d, cmap=CMAP, s=2)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.1, label="density")
    ax.set_xlabel(af.arguments[0])
    ax.set_ylabel(af.arguments[1])
    ax.set_zlabel(af.arguments[2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=22, azim=-60)
    ax.set_title("convex hull + points coloured by density", fontsize=9)
    for k, (i, j) in enumerate(sp.pairs(3)):
        axk = fig.add_subplot(1, 4, k + 2)
        im = _heatmap(axk, X, i, j, af)
        axk.set_xlabel(f"Deg({af.arguments[i]})")
        axk.set_ylabel(f"Deg({af.arguments[j]})")
        fig.colorbar(im, ax=axk, fraction=0.046)
    fig.suptitle(repr(af))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pairs(af, X, path):
    n = len(af)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(X[:, i], bins=60, range=(0, 1), color="tab:purple", density=True)
                ax.set_xlim(0, 1)
            elif i > j:
                _heatmap(ax, X, j, i, af)
            else:
                ax.axis("off")
                r = np.corrcoef(X[:, i], X[:, j])[0, 1]
                attack = ("→" if af.matrix[i, j] else "") + ("←" if af.matrix[j, i] else "")
                ax.text(0.5, 0.5, f"{af.arguments[i]} {attack or '·'} {af.arguments[j]}\nρ = {r:+.2f}",
                        ha="center", va="center", fontsize=10)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(af.arguments[i])
            if i == n - 1:
                ax.set_xlabel(af.arguments[j])
    fig.suptitle(repr(af) + "\nlower: 2-D projected density + projected hull; diagonal: marginal", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_resampling(af, res, path):
    P, Q = res["hists_first_two"][0][0], res["hists_first_two"][1][0]
    i, j = sp.pairs(len(af))[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmax = max(P.max(), Q.max()) * BINS**2
    for ax, H, t in ((axes[0], P, "sample 1"), (axes[1], Q, "sample 2")):
        im = ax.imshow(H.T * BINS**2, origin="lower", extent=(0, 1, 0, 1), cmap=CMAP, vmax=vmax)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    D = (P - Q) * BINS**2
    lim = np.abs(D).max()
    im = axes[2].imshow(D.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title(f"difference (TV = {sp.total_variation(P, Q):.4f})")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel(f"Deg({af.arguments[i]})")
        ax.set_ylabel(f"Deg({af.arguments[j]})")
    fig.suptitle(f"{af.name}: two independent re-samplings of N = 100,000")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def convergence_study(afs, path, n_samples=20_000):
    """Error to the fixed point vs iteration, and iterations vs epsilon."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    table = {}
    eps_list = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    W = np.random.default_rng(0).random((n_samples, 6))
    for af in afs:
        Wn = W[:, : len(af)]
        ref = h_categorizer_batch(af, Wn, 1e-15)
        x, errs = Wn.copy(), []
        for _ in range(60):
            x = Wn / (1 + x @ af.matrix)
            errs.append(np.abs(x - ref).max())
        axes[0].semilogy(range(1, 61), np.maximum(errs, 1e-17), label=af.name)
        iters = []
        for eps in eps_list:
            X, k = h_categorizer_batch(af, Wn, eps, return_iterations=True)
            iters.append(k)
        err_final = {eps: float(np.abs(h_categorizer_batch(af, Wn, eps) - ref).max()) for eps in eps_list}
        table[af.name] = {"iterations": dict(zip(map(str, eps_list), iters)),
                          "max_error": {str(k): v for k, v in err_final.items()}}
        axes[1].semilogx(eps_list, iters, "o-", label=af.name)
    axes[0].set_xlabel("iteration k")
    axes[0].set_ylabel(r"$\max_{i,a}|f^k_i(a) - Deg_i(a)|$")
    axes[0].set_title(f"distance to the fixed point ({n_samples:,} weight vectors)")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("iterations until max step < epsilon")
    axes[1].set_title("iterations needed vs epsilon")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return table


def plot_tv_vs_n(curves, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, c in curves.items():
        Ns = np.array(sorted(c))
        m = np.array([c[N][0] for N in Ns])
        s = np.array([c[N][1] for N in Ns])
        ax.errorbar(Ns, m, yerr=s, marker="o", capsize=3, label=name)
    Ns = np.array(sorted(next(iter(curves.values()))))
    ax.plot(Ns, 0.5 * np.sqrt(2 * BINS**2 / (np.pi * Ns)), "k:", label=r"$\propto 1/\sqrt{N}$ (theory)")
    ax.axvline(100_000, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of samples N")
    ax.set_ylabel("TV distance to reference heat-map")
    ax.set_title(f"Heat-map error vs N ({BINS}x{BINS} cells)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyse(af, n_samples, n_repeats, epsilon):
    t0 = time.time()
    W, X, iters = sp.sample_space(af, n_samples, seed=0, epsilon=epsilon)
    t_sample = time.time() - t0
    hull = sp.convex_hull(X)
    hv = sp.hull_volume(X, hull)
    tv, tv_se = sp.true_volume(af, X)
    gap, _ = sp.hull_gap(af, hull) if hull is not None else (float("nan"), 0)
    dens = np.abs(sp.jacobian_det(af, X))
    n = len(af)
    if n == 2:
        plot_2d(af, X, OUT / f"space_{af.name}.png")
    elif n == 3:
        plot_3d(af, X, hull, OUT / f"space_{af.name}.png")
    else:
        plot_pairs(af, X, OUT / f"space_{af.name}.png")
    row = {
        "framework": af.name, "n_args": n, "n_attacks": len(af.attacks),
        "iterations": iters, "sampling_time_s": round(t_sample, 3),
        "hull_volume": hv, "hull_vertices": int(len(hull.vertices)) if hull is not None else 0,
        "true_volume": tv, "true_volume_se": tv_se,
        "hull_empty_fraction": gap,
        "density_min": float(dens.min()), "density_max": float(dens.max()),
        "mean_degree": X.mean(0).round(4).tolist(),
    }
    if n_repeats > 1:
        res = sp.resampling_study(af, n_samples, n_repeats, BINS, epsilon)
        if af.name in ("attack2", "mutual2", "cycle3"):
            plot_resampling(af, res, OUT / f"resampling_{af.name}.png")
        res.pop("hists_first_two")
        row.update({f"resample_{k}": v for k, v in res.items()})
    return row


def write_markdown_table(rows, path):
    cols = [("framework", "AF"), ("n_args", "|A|"), ("n_attacks", "|R|"), ("iterations", "iter"),
            ("hull_volume", "vol(hull)"), ("true_volume", "vol(S)"), ("hull_empty_fraction", "empty in hull"),
            ("density_max", "max dens."), ("resample_tv_mean", "TV resample"),
            ("resample_hdr50_jaccard_mean", "Jaccard HDR50"), ("resample_hull_volume_rel_std", "CV vol(hull)")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:.2e}" if k == "resample_hull_volume_rel_std" else f"{v:.4f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--af", help="custom framework, e.g. 'a,b,c : a>b, b>c'")
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--repeats", type=int, default=10, help="re-samplings for the resilience test")
    p.add_argument("--epsilon", type=float, default=1e-10)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_samples, args.repeats = 5_000, 3
    OUT.mkdir(parents=True, exist_ok=True)

    afs = [fw.parse(args.af)] if args.af else fw.experiment_frameworks()
    rows = []
    for af in afs:
        t = time.time()
        rows.append(analyse(af, args.n_samples, args.repeats, args.epsilon))
        print(f"{af!r}: done in {time.time() - t:.1f}s")
    suffix = "_custom" if args.af else ""
    (OUT / f"summary{suffix}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown_table(rows, OUT / f"summary{suffix}.md")

    if not args.af:
        conv_afs = [fw.small_frameworks()[k] for k in ("attack2", "mutual2", "self1")] + \
                   [fw.chain(5), fw.cycle(3), fw.cycle(6), fw.complete(4), fw.random_af(6, 0.3, 3)]
        conv = convergence_study(conv_afs, OUT / "convergence.png")
        (OUT / "convergence.json").write_text(json.dumps(conv, indent=2), encoding="utf-8")
        sizes = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
        if args.quick:
            sizes = sizes[:3]
        curves = {af.name: sp.tv_vs_sample_size(af, sizes)
                  for af in (afs[1], afs[2], fw.cycle(3), fw.complete(4))}
        plot_tv_vs_n(curves, OUT / "tv_vs_n.png")
        (OUT / "tv_vs_n.json").write_text(json.dumps({k: {str(n): v for n, v in c.items()}
                                                      for k, c in curves.items()}, indent=2), encoding="utf-8")
    print(f"results written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
