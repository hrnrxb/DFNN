# Visualizations

## Matplotlib (`diagrams.py`)

```bash
python visualizations/diagrams.py
```

Writes PNGs into `figures/`.

## Manim cinematics (`manim_scenes.py`)

Manim Community v0.18+ and `ffmpeg` on `PATH`. Pango `Text` only. No TeX.

```bash
manim -qm visualizations/manim_scenes.py AffineLoom BlameTelegraph JacobianCollapse ChainRuleContract DeadSignal
```

Pre-rendered 720p30 files: `visualizations/rendered/`.

| Class | File | Point |
| --- | --- | --- |
| `AffineLoom` | `01_affine_loom.mp4` | Linear maps shear. Tanh folds. XOR separates. |
| `BlameTelegraph` | `02_blame_telegraph.mp4` | Reverse-mode autodiff as a protocol. |
| `JacobianCollapse` | `03_jacobian_collapse.mp4` | Softmax Jacobian collapses to `P - Y`. |
| `ChainRuleContract` | `04_chain_rule_contract.mp4` | `dW = X^T dZ` is the only legal product. |
| `DeadSignal` | `05_dead_signal.mp4` | Tiny init is not a saddle. |
