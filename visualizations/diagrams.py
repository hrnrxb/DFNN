"""Publication-grade static diagrams for DFNN.

Used by the notebook and by README figure generation.
Requires only matplotlib and numpy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, FancyArrow
from matplotlib.collections import LineCollection
import numpy as np

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

BG = "#0B0F19"
INK = "#E8ECF1"
MUTED = "#8B93A7"
GOLD = "#F6C945"
CYAN = "#5EC8F2"
CRIMSON = "#FF5D73"
LIME = "#7DDAA1"
VIOLET = "#B794F6"
PANEL = "#121826"
STROKE = "#2A3A52"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": STROKE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#1E2A3C",
            "grid.alpha": 0.9,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.facecolor": PANEL,
            "legend.edgecolor": STROKE,
            "savefig.facecolor": BG,
            "savefig.dpi": 160,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    return path


def _box(ax, xy, w, h, text, color, fontsize=10, textcolor=None):
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6,
        edgecolor=color,
        facecolor=PANEL,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=textcolor or color, fontsize=fontsize, fontweight="bold")
    return patch


def _arrow(ax, p0, p1, color=MUTED):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=12),
    )


def fig_computational_graph(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Forward cache, backward message  ·  one computational graph")

    nodes = [
        (1.1, 2.6, "X\n(N, Fin)", INK),
        (3.1, 2.6, "Z = XW+b\n(N, Fout)", CYAN),
        (5.3, 2.6, "A = f(Z)\n(N, Fout)", VIOLET),
        (7.6, 2.6, "logits ZL\n(N, C)", CYAN),
        (10.1, 2.6, "L = CCE(P,Y)\nscalar", CRIMSON),
    ]
    for x, y, t, c in nodes:
        _box(ax, (x, y), 1.7, 1.15, t, c, fontsize=9)
    for a, b in zip(nodes, nodes[1:]):
        _arrow(ax, (a[0] + 0.9, 2.6), (b[0] - 0.9, 2.6), MUTED)

    ax.text(6, 4.35, "FORWARD  (gold)  cache X and Z", color=GOLD, ha="center", fontsize=11)
    ax.text(6, 0.85, "BACKWARD  (crimson)  dZ, dW, db, dX", color=CRIMSON, ha="center", fontsize=11)
    _arrow(ax, (10.1, 1.95), (7.6, 1.95), CRIMSON)
    _arrow(ax, (7.6, 1.95), (5.3, 1.95), CRIMSON)
    _arrow(ax, (5.3, 1.95), (3.1, 1.95), CRIMSON)
    _arrow(ax, (3.1, 1.95), (1.1, 1.95), CRIMSON)
    ax.text(3.1, 3.45, "store X", color=GOLD, ha="center", fontsize=8)
    ax.text(5.3, 3.45, "store Z", color=GOLD, ha="center", fontsize=8)
    path = _save(fig, "static_01_computational_graph.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_shape_flow(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Dense layer shape contract  ·  Z = XW + b")

    specs = [
        (1.5, 2.1, 1.5, 2.4, "X\n(N, Fin)", CYAN),
        (4.0, 2.1, 1.7, 2.0, "W\n(Fin, Fout)", GOLD),
        (6.6, 2.1, 1.5, 2.0, "XW\n(N, Fout)", LIME),
        (9.0, 2.1, 1.1, 1.1, "b\n(1, Fout)", VIOLET),
        (10.8, 2.1, 1.5, 2.0, "Z\n(N, Fout)", CRIMSON),
    ]
    for x, y, w, h, t, c in specs:
        _box(ax, (x, y), w, h, t, c, fontsize=10)
    ax.text(2.75, 2.1, "·", color=INK, fontsize=22, ha="center", va="center")
    ax.text(5.35, 2.1, "=", color=INK, fontsize=18, ha="center")
    ax.text(7.85, 2.1, "+", color=INK, fontsize=18, ha="center")
    ax.text(9.95, 2.1, "=", color=INK, fontsize=18, ha="center")
    ax.text(
        6.0,
        0.45,
        "Bias (1, Fout) broadcasts across N.  Keep it 2D or you will invent a transpose bug.",
        color=MUTED,
        ha="center",
        fontsize=10,
    )
    path = _save(fig, "static_02_shape_flow.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_backprop_shapes(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Three contractions of backpropagation")

    cards = [
        (2.0, 2.7, "dW = Xᵀ · dZ", "(Fin, N)·(N, Fout)\n→ (Fin, Fout)", GOLD),
        (6.0, 2.7, "db = Σ dZ", "sum over axis 0\n→ (1, Fout)", CYAN),
        (10.0, 2.7, "dX = dZ · Wᵀ", "(N, Fout)·(Fout, Fin)\n→ (N, Fin)", CRIMSON),
    ]
    for x, y, title, body, c in cards:
        _box(ax, (x, y), 3.3, 2.6, f"{title}\n\n{body}", c, fontsize=11)
    ax.text(6, 0.7, "If a shape does not match, the math is wrong.  Do not reshape to silence NumPy.", color=MUTED, ha="center")
    path = _save(fig, "static_03_backprop_shapes.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_xor_geometry(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    pts = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    cols = [CRIMSON, CRIMSON, CYAN, CYAN]
    labs = ["(0,0)→0", "(1,1)→0", "(0,1)→1", "(1,0)→1"]
    for p, c, lab in zip(pts, cols, labs):
        ax.scatter(*p, s=220, c=c, zorder=3, edgecolors=INK, linewidths=0.6)
        ax.annotate(lab, p + np.array([0.06, 0.06]), color=c, fontsize=10)
    ax.plot([0, 1], [0, 1], color=CRIMSON, lw=1.2, ls="--", alpha=0.5)
    ax.plot([0, 1], [1, 0], color=CYAN, lw=1.2, ls="--", alpha=0.5)
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(-0.25, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("XOR is not linearly separable")
    ax.grid(True, alpha=0.35)
    path = _save(fig, "static_04_xor_geometry.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_idx_format(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("MNIST IDX layout  ·  big-endian integers, then raw bytes")

    blocks = [
        (1.4, 2.1, 2.2, 1.6, "magic\nuint32\n0x00000803", GOLD),
        (3.9, 2.1, 2.0, 1.6, "count\nuint32\n60000", CYAN),
        (6.2, 2.1, 2.0, 1.6, "rows\nuint32\n28", VIOLET),
        (8.5, 2.1, 2.0, 1.6, "cols\nuint32\n28", LIME),
        (10.6, 2.1, 2.0, 1.6, "pixels\nuint8[N·28·28]", CRIMSON),
    ]
    for x, y, w, h, t, c in blocks:
        _box(ax, (x, y), w, h, t, c, fontsize=9)
    ax.text(6, 0.55, "Labels file: magic 0x00000801, then count, then uint8 labels.  We scale pixels to [0, 1].", color=MUTED, ha="center")
    path = _save(fig, "static_05_idx_format.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_three_way_split(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title("The three-way split  ·  train / validation / test")
    _box(ax, (2.0, 1.7), 3.2, 1.4, "Train\nfit parameters", CYAN, 12)
    _box(ax, (5.3, 1.7), 2.4, 1.4, "Validation\nchoose hyperparameters", GOLD, 11)
    _box(ax, (8.3, 1.7), 2.4, 1.4, "Test\ntouch once", CRIMSON, 12)
    ax.text(5, 0.45, "Reporting test as 'validation' is not gradient leakage.  It is a naming lie that hides overfitting.", color=MUTED, ha="center", fontsize=9)
    path = _save(fig, "static_06_three_way_split.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_topology(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("MNIST MLP  ·  784 → 256 (He, ReLU) → 64 (He, ReLU) → 10 (Xavier)")
    layers = [
        (1.5, 2.1, "784\npixels", INK),
        (4.0, 2.1, "256\nReLU · He", CYAN),
        (6.6, 2.1, "64\nReLU · He", GOLD),
        (9.4, 2.1, "10\nlogits · Xavier", CRIMSON),
    ]
    for x, y, t, c in layers:
        _box(ax, (x, y), 2.0, 1.6, t, c, 11)
    for a, b in zip(layers, layers[1:]):
        _arrow(ax, (a[0] + 1.05, 2.1), (b[0] - 1.05, 2.1), MUTED)
    ax.text(6, 0.55, "Softmax is fused into the loss.  The network itself ends on raw logits.", color=MUTED, ha="center")
    path = _save(fig, "static_07_topology.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_momentum_ravine(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    xs = np.linspace(-2.2, 2.2, 220)
    ys = np.linspace(-2.2, 2.2, 220)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.08 * X**2 + 1.6 * Y**2
    cs = ax.contour(X, Y, Z, levels=14, colors=CYAN, linewidths=0.8, alpha=0.7)
    # naive SGD zig-zag
    sgd = np.array([[1.8, 1.6], [1.55, -1.1], [1.35, 0.85], [1.15, -0.65], [0.95, 0.5], [0.75, -0.38], [0.55, 0.28]])
    mom = np.array([[1.8, 1.6], [1.4, 1.05], [0.95, 0.55], [0.55, 0.22], [0.28, 0.08], [0.12, 0.02], [0.04, 0.0]])
    ax.plot(sgd[:, 0], sgd[:, 1], "-o", color=CRIMSON, lw=2, ms=5, label="vanilla SGD")
    ax.plot(mom[:, 0], mom[:, 1], "-o", color=GOLD, lw=2, ms=5, label="Polyak momentum")
    ax.scatter([0], [0], c=LIME, s=80, zorder=5, label="minimum")
    ax.set_title("The ravine problem")
    ax.set_xlabel("long, flat axis")
    ax.set_ylabel("steep walls")
    ax.legend(loc="upper right", fontsize=9)
    path = _save(fig, "static_08_momentum_ravine.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_linear_vs_nonlinear(show: bool = True) -> Path:
    apply_style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    # left: linear composition stays a line
    t = np.linspace(-1.4, 1.4, 80)
    axes[0].plot(t, 0.7 * t, color=GOLD, lw=2)
    axes[0].plot(t, 1.4 * (0.7 * t), color=CRIMSON, lw=2, ls="--")
    axes[0].set_title("Two linear maps = one linear map")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.35)
    # right: tanh warp
    xx, yy = np.meshgrid(np.linspace(-2, 2, 18), np.linspace(-2, 2, 18))
    u = np.tanh(0.9 * xx + 0.4 * yy)
    v = np.tanh(-0.3 * xx + 1.1 * yy)
    axes[1].plot(xx, yy, color="#243044", lw=0.6)
    axes[1].plot(xx.T, yy.T, color="#243044", lw=0.6)
    axes[1].plot(u, v, color=CYAN, lw=0.8)
    axes[1].plot(u.T, v.T, color=CYAN, lw=0.8)
    axes[1].set_title("Tanh folds the plane")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    path = _save(fig, "light_01_linear_vs_nonlinear.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_activations(show: bool = True) -> Path:
    apply_style()
    x = np.linspace(-4.5, 4.5, 400)
    tanh = np.tanh(x)
    tanh_p = 1 - tanh**2
    relu = np.maximum(0, x)
    relu_p = (x > 0).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(x, tanh, color=GOLD, lw=2.2, label="tanh")
    axes[0].plot(x, tanh_p, color=CYAN, lw=1.8, label="tanh'")
    axes[0].axhline(0, color=STROKE, lw=0.8)
    axes[0].axvline(0, color=STROKE, lw=0.8)
    axes[0].set_title("Tanh saturates  ·  derivative dies in the tails")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, relu, color=GOLD, lw=2.2, label="ReLU")
    axes[1].plot(x, relu_p, color=CYAN, lw=1.8, label="ReLU'")
    axes[1].axhline(0, color=STROKE, lw=0.8)
    axes[1].axvline(0, color=STROKE, lw=0.8)
    axes[1].set_title("ReLU  ·  constant gradient 1 on the positive half-line")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = _save(fig, "light_02_activations.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_init_variance(show: bool = True) -> Path:
    apply_style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    configs = [
        ("randn * 0.01  (too small)", 0.01, CRIMSON),
        ("Xavier  sqrt(2/(n_in+n_out))", np.sqrt(2 / (256 + 256)), GOLD),
        ("He  sqrt(2/n_in)", np.sqrt(2 / 256), CYAN),
    ]
    x = rng.normal(0, 1, size=(2000, 256))
    for ax, (title, std, color) in zip(axes, configs):
        W = rng.normal(0, std, size=(256, 256))
        h = x
        for _ in range(6):
            h = np.maximum(0, h @ W)
        ax.hist(h.ravel(), bins=40, color=color, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("activation")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("count")
    fig.suptitle("Six ReLU layers  ·  activation histogram after depth", color=INK, y=1.03)
    fig.tight_layout()
    path = _save(fig, "light_04_init_variance.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_softmax_jacobian(show: bool = True) -> Path:
    apply_style()
    logits = np.array([1.6, -0.4, 0.9, 0.1, 0.3])
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    J = np.diag(p) - np.outer(p, p)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(J, cmap="coolwarm", vmin=-0.25, vmax=0.25)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlabel("dZ_j")
    ax.set_ylabel("dP_i")
    ax.set_title("Softmax Jacobian  J = diag(P) − PPᵀ")
    fig.colorbar(im, ax=ax, fraction=0.046)
    path = _save(fig, "light_06_softmax_jacobian.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_broadcast(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("Bias broadcast  ·  why b is (1, Fout), not (Fout,)")
    _box(ax, (2.0, 1.7), 2.4, 1.6, "XW\n(N, Fout)", CYAN, 11)
    _box(ax, (5.0, 1.7), 2.2, 0.9, "b  (1, Fout)", GOLD, 11)
    _box(ax, (8.1, 1.7), 2.4, 1.6, "Z\n(N, Fout)", LIME, 11)
    ax.text(3.55, 1.7, "+", color=INK, fontsize=18, ha="center")
    ax.text(6.55, 1.7, "=", color=INK, fontsize=18, ha="center")
    ax.text(5, 0.45, "NumPy stretches b down the batch. A 1D (Fout,) vector is how silent transpose bugs start.", color=MUTED, ha="center", fontsize=9)
    path = _save(fig, "static_09_broadcast.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_onehot(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title("Integer label vs one-hot  ·  argmax on the wrong shape is silently 0")
    _box(ax, (2.2, 1.8), 2.6, 1.5, "label  3\nshape (N,)", GOLD, 12)
    _box(ax, (7.4, 1.8), 3.6, 1.5, "[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]\nshape (N, 10)", CYAN, 11)
    _arrow(ax, (3.6, 1.8), (5.5, 1.8), MUTED)
    ax.text(5, 0.5, "calculate_accuracy uses argmax on axis 1. Integers in a column will all look like class 0.", color=MUTED, ha="center", fontsize=9)
    path = _save(fig, "static_10_onehot.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_logsumexp(show: bool = True) -> Path:
    apply_style()
    z = np.linspace(-8, 40, 400)
    naive = np.exp(z)
    naive[~np.isfinite(naive)] = np.nan
    shift = np.exp(z - np.max(z))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(z, naive, color=CRIMSON, lw=2, label="exp(z)  overflow")
    ax.plot(z, shift, color=LIME, lw=2, label="exp(z - max z)")
    ax.set_title("Log-sum-exp  ·  softmax without inf")
    ax.set_xlabel("logit")
    ax.set_ylabel("exp")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = _save(fig, "light_09_logsumexp.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_shape_bug_y(show: bool = True) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    left = np.zeros((4, 4))
    np.fill_diagonal(left, 1)
    axes[0].imshow(left, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("(4,1) - (4,)  becomes (4,4)")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    right = np.array([[1], [0], [0], [1]], dtype=float)
    axes[1].imshow(right, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("(4,1) - (4,1)  stays (4,1)")
    axes[1].set_xticks([0])
    axes[1].set_yticks(range(4))
    fig.suptitle("XOR Y must be (4, 1). A 1D Y is the classic silent broadcast bug.", color=INK, y=1.03)
    fig.tight_layout()
    path = _save(fig, "light_10_y_broadcast.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_softmax_bars(show: bool = True) -> Path:
    apply_style()
    z = np.array([1.6, -0.4, 0.9, 0.1, 0.3])
    e = np.exp(z - z.max())
    p = e / e.sum()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].bar(range(5), z, color=GOLD)
    axes[0].set_title("logits Z")
    axes[0].set_xticks(range(5))
    axes[1].bar(range(5), p, color=CYAN)
    axes[1].set_title("softmax P  (sums to 1)")
    axes[1].set_xticks(range(5))
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = _save(fig, "light_11_softmax_bars.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_overfit_sketch(show: bool = True) -> Path:
    apply_style()
    ep = np.arange(1, 41)
    train = 0.45 * np.exp(-ep / 6) + 0.002
    val = 0.22 * np.exp(-ep / 7) + 0.065 + 0.00035 * np.maximum(0, ep - 15)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(ep, train, color=CYAN, lw=2, label="train loss")
    ax.plot(ep, val, color=GOLD, lw=2, label="test loss (called val in the loop)")
    ax.axvline(15, color=CRIMSON, ls="--", lw=1, label="useful learning is over")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("What 40 epochs actually did")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = _save(fig, "light_12_overfit_sketch.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_minibatch_noise(show: bool = True) -> Path:
    apply_style()
    rng = np.random.default_rng(1)
    t = np.linspace(0, 8, 200)
    full = np.exp(-t / 3)
    mini = full + 0.08 * rng.normal(size=t.shape) * np.exp(-t / 8)
    sgd1 = full + 0.28 * rng.normal(size=t.shape) * np.exp(-t / 10)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(t, full, color=GOLD, lw=2, label="full batch")
    ax.plot(t, mini, color=CYAN, lw=1.4, label="mini-batch 128")
    ax.plot(t, sgd1, color=CRIMSON, lw=0.9, alpha=0.85, label="batch size 1")
    ax.set_title("Why we do not use batch size 1 or N")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (sketch)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = _save(fig, "light_13_minibatch.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_update_step(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("One training step")
    items = [
        (1.2, "forward\ncache X,Z", CYAN),
        (3.5, "loss\nscalar L", GOLD),
        (5.8, "backward\ndW, db, dX", CRIMSON),
        (8.2, "optimizer\nW -= lr v", LIME),
        (10.2, "repeat", INK),
    ]
    for x, t, c in items:
        _box(ax, (x, 1.55), 1.9, 1.3, t, c, 10)
    for a, b in zip(items, items[1:]):
        _arrow(ax, (a[0] + 0.95, 1.55), (b[0] - 0.95, 1.55), MUTED)
    path = _save(fig, "static_11_train_step.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_relu_dead(show: bool = True) -> Path:
    apply_style()
    x = np.linspace(-3, 3, 300)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(x, np.maximum(0, x), color=GOLD, lw=2.2, label="ReLU")
    ax.axvspan(-3, 0, color=CRIMSON, alpha=0.12)
    ax.text(-1.5, 1.6, "gradient 0\n(dead)", color=CRIMSON, ha="center", fontsize=10)
    ax.text(1.4, 1.6, "gradient 1", color=LIME, ha="center", fontsize=10)
    ax.set_title("ReLU dead zone  ·  why He init exists")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = _save(fig, "light_14_relu_dead.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def fig_chain_telescope(show: bool = True) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Chain rule as a telescope  ·  dL/dW = (dL/dZ) (dZ/dW)")
    _box(ax, (1.6, 1.55), 2.2, 1.2, "dL/dZ", CRIMSON, 12)
    _box(ax, (4.6, 1.55), 2.2, 1.2, "dZ/dW = X", CYAN, 12)
    _box(ax, (8.8, 1.55), 3.2, 1.2, "dL/dW = X^T dZ", GOLD, 12)
    ax.text(3.15, 1.55, "x", color=INK, fontsize=16, ha="center")
    ax.text(6.4, 1.55, "=", color=INK, fontsize=16, ha="center")
    path = _save(fig, "static_12_chain.png")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def render_all_static(show: bool = False) -> list[Path]:
    fns = [
        fig_computational_graph,
        fig_shape_flow,
        fig_backprop_shapes,
        fig_xor_geometry,
        fig_idx_format,
        fig_three_way_split,
        fig_topology,
        fig_momentum_ravine,
        fig_linear_vs_nonlinear,
        fig_activations,
        fig_init_variance,
        fig_softmax_jacobian,
        fig_broadcast,
        fig_onehot,
        fig_logsumexp,
        fig_shape_bug_y,
        fig_softmax_bars,
        fig_overfit_sketch,
        fig_minibatch_noise,
        fig_update_step,
        fig_relu_dead,
        fig_chain_telescope,
    ]
    return [fn(show=show) for fn in fns]


if __name__ == "__main__":
    paths = render_all_static(show=False)
    for p in paths:
        print(p)
