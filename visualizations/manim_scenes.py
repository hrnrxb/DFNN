"""DFNN Manim scenes. Pango Text only. No LaTeX.

    manim -qm visualizations/manim_scenes.py AffineLoom BlameTelegraph JacobianCollapse ChainRuleContract DeadSignal
"""

from manim import *
import numpy as np

BG = "#0B0F19"
GOLD = "#F6C945"
CYAN = "#5EC8F2"
CRIMSON = "#FF5D73"
LIME = "#7DDAA1"
IVORY = "#E8ECF1"
SLATE = "#8B93A7"
VIOLET = "#B794F6"
INK = "#121826"


def _label(text, size=28, color=IVORY, weight=NORMAL):
    return Text(text, font_size=size, color=color, weight=weight)


def _chip(text, color, width=1.45, height=0.62):
    box = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.1,
        color=color,
        stroke_width=2,
        fill_color=INK,
        fill_opacity=0.95,
    )
    lab = _label(text, 20, color)
    return VGroup(box, lab)


def _tensor(h, w, name, color, note):
    rect = RoundedRectangle(
        width=w,
        height=h,
        corner_radius=0.08,
        color=color,
        fill_color=INK,
        fill_opacity=0.95,
        stroke_width=2,
    )
    lab = _label(name, 22, color, BOLD).move_to(rect.get_center() + UP * 0.16)
    n = _label(note, 16, SLATE).move_to(rect.get_center() + DOWN * 0.22)
    return VGroup(rect, lab, n)


class AffineLoom(Scene):
    """XOR is two diagonals. A line cannot cut them. Tanh folds the cloth."""

    def construct(self):
        self.camera.background_color = BG
        title = _label("The Affine Loom", 40, GOLD, BOLD)
        subtitle = _label("A line cannot solve XOR. A fold can.", 22, SLATE)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.14).to_edge(UP, buff=0.22)

        plane = NumberPlane(
            x_range=(-2.8, 2.8, 1),
            y_range=(-1.7, 1.7, 1),
            x_length=9.0,
            y_length=4.6,
            background_line_style={"stroke_color": "#2A3A52", "stroke_width": 1.2},
            axis_config={"stroke_color": "#4A5C78", "stroke_width": 2},
        ).shift(DOWN * 0.22)

        pts = [
            (np.array([-1.05, -1.05, 0.0]), CRIMSON, "00"),
            (np.array([1.05, 1.05, 0.0]), CRIMSON, "11"),
            (np.array([-1.05, 1.05, 0.0]), CYAN, "01"),
            (np.array([1.05, -1.05, 0.0]), CYAN, "10"),
        ]
        dots = VGroup()
        for p, color, _name in pts:
            dots.add(Dot(p + DOWN * 0.22, color=color, radius=0.13))

        cap = _label("XOR: same-color corners. One straight cut always leaves a mix.", 20, IVORY)
        cap.to_edge(DOWN, buff=0.18)

        self.play(FadeIn(header), Create(plane), run_time=1.0)
        self.play(FadeIn(dots, scale=0.7), Write(cap), run_time=0.8)

        knife = Line([-2.4, -0.55, 0], [2.4, 0.7, 0], color=GOLD, stroke_width=4).shift(DOWN * 0.22)
        self.play(Create(knife), run_time=0.45)
        self.play(Rotate(knife, angle=PI / 3.4, about_point=np.array([0, -0.22, 0])), run_time=0.9)
        fail = _label("Linearly inseparable. This is the 1969 perceptron limit.", 20, CRIMSON)
        fail.to_edge(DOWN, buff=0.18)
        self.play(Transform(cap, fail), FadeOut(knife), run_time=0.55)

        W = np.array([[1.15, 0.38], [0.12, 1.05]])

        def linear_map(p):
            v = np.array([p[0], p[1]]) @ W
            return np.array([v[0], v[1], p[2]])

        shear = _label("Dense: Z = XW + b. Lines stay lines. XOR still mixed.", 20, CYAN)
        shear.to_edge(DOWN, buff=0.18)
        self.play(Transform(cap, shear), run_time=0.35)
        plane.prepare_for_nonlinear_transform()
        self.play(plane.animate.apply_function(linear_map), dots.animate.apply_function(linear_map), run_time=1.4)

        def fold_map(p):
            return np.array(
                [
                    0.62 * p[0] + 0.95 * np.tanh(0.55 * p[0]),
                    0.62 * p[1] + 0.80 * np.tanh(0.55 * p[1]),
                    p[2],
                ]
            )

        fold = _label("Tanh folds the cloth. A straight cut in hidden space now works.", 20, LIME)
        fold.to_edge(DOWN, buff=0.18)
        self.play(Transform(cap, fold), run_time=0.35)
        plane.prepare_for_nonlinear_transform()
        self.play(plane.animate.apply_function(fold_map), dots.animate.apply_function(fold_map), run_time=1.45)

        sep = Line([-2.6, -0.18, 0], [2.6, 0.18, 0], color=LIME, stroke_width=4)
        win = _label("Hidden coordinates made XOR linearly separable.", 20, LIME)
        win.to_edge(DOWN, buff=0.18)
        self.play(Create(sep), Transform(cap, win), run_time=0.7)
        self.wait(0.55)


class BlameTelegraph(Scene):
    """Backprop is a protocol, not a cartoon of neurons."""

    def construct(self):
        self.camera.background_color = BG
        title = _label("The Blame Telegraph", 40, GOLD, BOLD)
        subtitle = _label("One Jacobian-vector product per layer. Never the full Jacobian.", 21, SLATE)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.2)
        self.play(FadeIn(header), run_time=0.55)

        names = ["X", "Dense", "Z", "Tanh", "A", "Dense", "Z2", "Loss"]
        colors = [IVORY, CYAN, GOLD, VIOLET, GOLD, CYAN, GOLD, CRIMSON]
        chips = VGroup(*[_chip(n, c) for n, c in zip(names, colors)])
        chips.arrange(RIGHT, buff=0.18).shift(UP * 1.45)

        arrows = VGroup()
        for i in range(len(chips) - 1):
            arrows.add(
                Arrow(
                    chips[i].get_right() + RIGHT * 0.01,
                    chips[i + 1].get_left() + LEFT * 0.01,
                    buff=0.02,
                    stroke_width=2.5,
                    color=SLATE,
                    max_tip_length_to_length_ratio=0.2,
                )
            )
        self.play(LaggedStart(*[FadeIn(c) for c in chips], lag_ratio=0.07), run_time=0.9)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.07), run_time=0.7)

        fwd = _label("FORWARD  ·  cache X and Z. Backward will need them.", 20, CYAN)
        fwd.to_edge(DOWN, buff=0.22)
        self.play(Write(fwd), run_time=0.5)
        pulse = Dot(color=GOLD, radius=0.08).move_to(chips[0].get_center())
        self.add(pulse)
        for chip in chips[1:]:
            self.play(pulse.animate.move_to(chip.get_center()), run_time=0.16)
        self.play(FadeOut(pulse), run_time=0.15)

        bwd = _label("BACKWARD  ·  blame walks the same path, reversed.", 20, CRIMSON)
        bwd.to_edge(DOWN, buff=0.22)
        self.play(Transform(fwd, bwd), run_time=0.45)
        blame = Dot(color=CRIMSON, radius=0.09).move_to(chips[-1].get_center())
        self.add(blame)
        for chip in reversed(list(chips)[:-1]):
            self.play(blame.animate.move_to(chip.get_center()), run_time=0.16)
        self.play(FadeOut(blame), run_time=0.15)

        laws = VGroup(
            _label("Dense local laws", 24, GOLD, BOLD),
            _label("dW = X^T · dZ     only product whose shape is W", 21, IVORY),
            _label("db = sum(dZ, axis=0)     bias was broadcast over the batch", 21, IVORY),
            _label("dX = dZ · W^T     this tensor is the message to the previous layer", 21, IVORY),
            _label("Activation: dX = dZ  ⊙  f'(Z)     Hadamard, not a matmul", 21, VIOLET),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        laws.next_to(chips, DOWN, buff=0.45)
        panel = RoundedRectangle(
            width=min(laws.width + 0.6, 13.2),
            height=laws.height + 0.42,
            corner_radius=0.14,
            color="#243044",
            stroke_width=1.4,
            fill_color="#101826",
            fill_opacity=0.96,
        ).move_to(laws.get_center())
        self.play(FadeIn(panel), FadeIn(laws), run_time=1.1)
        self.wait(0.8)


class JacobianCollapse(Scene):
    """Naive softmax Jacobian is C x C. Fused with CCE it is P - Y."""

    def construct(self):
        self.camera.background_color = BG
        title = _label("The Jacobian Collapse", 40, GOLD, BOLD)
        subtitle = _label("Softmax + cross-entropy  ·  the C×C tensor is never allocated", 21, SLATE)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.2)
        self.play(FadeIn(header), run_time=0.55)

        logits = np.array([1.6, -0.4, 0.9, 0.1])
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        y = np.array([0.0, 0.0, 1.0, 0.0])
        J = np.diag(p) - np.outer(p, p)

        def heat(v):
            t = float(np.clip((v + 0.25) / 0.5, 0, 1))
            return rgb_to_color((t, 0.18, 1 - t))

        cells = VGroup()
        cell = 0.46
        for i in range(4):
            for j in range(4):
                sq = Square(cell, stroke_color="#1E2A3C", stroke_width=1, fill_opacity=0.95)
                sq.set_fill(heat(J[i, j]))
                sq.move_to([(j - 1.5) * cell, (1.5 - i) * cell, 0])
                cells.add(sq)
        jac = VGroup(SurroundingRectangle(cells, color=SLATE, buff=0.07, stroke_width=1.5), cells)
        jac_lab = _label("J = diag(P) - P P^T", 18, CYAN)
        jac_g = VGroup(jac, jac_lab).arrange(DOWN, buff=0.14)

        mid = VGroup(
            _label("times", 22, GOLD, BOLD),
            _label("dL/dP = -Y/P", 20, CRIMSON),
        ).arrange(DOWN, buff=0.12)

        bars = VGroup()
        for i, val in enumerate(p - y):
            h = 0.35 + 2.0 * abs(val)
            col = GOLD if val >= 0 else CRIMSON
            rect = RoundedRectangle(
                width=0.62,
                height=h,
                corner_radius=0.07,
                color=col,
                fill_color=col,
                fill_opacity=0.88,
                stroke_width=1,
            )
            num = _label(f"{val:+.2f}", 16, col)
            name = _label(f"P{i}-Y{i}", 14, IVORY)
            colg = VGroup(num, rect, name).arrange(DOWN, buff=0.08)
            bars.add(colg)
        bars.arrange(RIGHT, buff=0.22, aligned_edge=DOWN)
        bar_lab = _label("dL/dZ = P - Y", 18, LIME)
        bar_g = VGroup(bars, bar_lab).arrange(DOWN, buff=0.16)

        eq = _label("=", 34, GOLD, BOLD)
        row = VGroup(jac_g, mid, eq, bar_g).arrange(RIGHT, buff=0.38).shift(DOWN * 0.15)
        if row.width > 13.4:
            row.scale(13.4 / row.width)

        self.play(FadeIn(jac_g), run_time=0.7)
        self.play(FadeIn(mid), FadeIn(eq), run_time=0.4)
        cap = _label("Naive path builds a C×C matrix per example. O(C^2) memory.", 20, SLATE)
        cap.to_edge(DOWN, buff=0.22)
        self.play(Write(cap), run_time=0.45)
        self.play(FadeIn(bar_g, shift=UP * 0.15), run_time=0.8)

        fused = _label("Fused path: dL/dZ = (P - Y) / N. One subtraction.", 21, LIME)
        fused.to_edge(DOWN, buff=0.22)
        self.play(Transform(cap, fused), Circumscribe(bar_g, color=LIME, fade_out=True), run_time=1.0)
        self.wait(0.7)


class ChainRuleContract(Scene):
    """Why dW = X^T dZ is the only legal product."""

    def construct(self):
        self.camera.background_color = BG
        title = _label("The only legal contraction", 36, GOLD, BOLD)
        sub = _label("dW must have the same shape as W. Guess-and-check is not a method.", 20, SLATE)
        header = VGroup(title, sub).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.22)
        self.play(FadeIn(header), run_time=0.5)

        X = _tensor(1.55, 2.4, "X", CYAN, "(N, Fin)")
        dZ = _tensor(1.55, 2.4, "dZ", CRIMSON, "(N, Fout)")
        intro = VGroup(X, _label("and", 18, SLATE), dZ).arrange(RIGHT, buff=0.5)
        intro.shift(DOWN * 0.05)
        self.play(FadeIn(intro), run_time=0.6)
        cap = _label("Forward used X @ W. Backward needs a product that rebuilds W's shape.", 19, IVORY)
        cap.to_edge(DOWN, buff=0.2)
        self.play(Write(cap), run_time=0.45)
        self.play(FadeOut(intro), run_time=0.4)

        XT = _tensor(2.2, 1.7, "X^T", CYAN, "(Fin, N)")
        dZ2 = _tensor(1.7, 2.2, "dZ", CRIMSON, "(N, Fout)")
        W = _tensor(2.0, 2.2, "dW", GOLD, "(Fin, Fout)")
        at = _label("@", 32, GOLD, BOLD)
        eq = _label("=", 32, GOLD, BOLD)
        row = VGroup(XT, at, dZ2, eq, W).arrange(RIGHT, buff=0.32)
        row.shift(DOWN * 0.05)
        self.play(FadeIn(row), run_time=0.8)
        win = _label("(Fin, N) @ (N, Fout) = (Fin, Fout). That is dW = X^T · dZ.", 20, LIME)
        win.to_edge(DOWN, buff=0.2)
        self.play(Transform(cap, win), run_time=0.55)
        self.wait(0.7)


class DeadSignal(Scene):
    """Tiny init sits at 0.5. Xavier actually moves."""

    def construct(self):
        self.camera.background_color = BG
        title = _label("A dead net is not a saddle", 36, GOLD, BOLD)
        sub = _label("XOR, 2-3-1, tanh, 1000 steps, lr = 0.1. Only the init changes.", 20, SLATE)
        header = VGroup(title, sub).arrange(DOWN, buff=0.12).to_edge(UP, buff=0.22)
        self.play(FadeIn(header), run_time=0.5)

        def column(title_txt, color, values, tag):
            head = _label(title_txt, 22, color, BOLD)
            rows = VGroup()
            names = ["00", "01", "10", "11"]
            targets = ["0", "1", "1", "0"]
            for n, t, v in zip(names, targets, values):
                s = _label(f"{n}  target {t}  pred {v:.2f}", 20, IVORY)
                rows.add(s)
            tag_l = _label(tag, 18, color)
            return VGroup(head, rows.arrange(DOWN, aligned_edge=LEFT, buff=0.12), tag_l).arrange(DOWN, buff=0.22)

        left = column("W ~ 0.01 N(0,1)", CRIMSON, [0.50, 0.50, 0.50, 0.50], "MSE = 0.25  ·  nothing moved")
        right = column("Xavier", LIME, [0.00, 0.94, 0.94, 0.01], "MSE ~ 2e-3  ·  XOR is solved")
        VGroup(left, right).arrange(RIGHT, buff=1.3).shift(DOWN * 0.15)
        self.play(FadeIn(left), run_time=0.7)
        cap = _label("tanh of a near-zero pre-activation is ~0. After output tanh you sit at 0.5.", 19, SLATE)
        cap.to_edge(DOWN, buff=0.2)
        self.play(Write(cap), run_time=0.5)
        self.play(FadeIn(right), run_time=0.7)
        win = _label("Same topology. Same loop. Different sigma. That is initialization.", 20, GOLD)
        win.to_edge(DOWN, buff=0.2)
        self.play(Transform(cap, win), run_time=0.5)
        self.wait(0.7)
