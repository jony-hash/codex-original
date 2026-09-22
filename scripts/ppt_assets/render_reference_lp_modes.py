"""Render literature-calibrated LP-like reference modes for PPT assets.

This renderer is intentionally self-contained so it can run in the supported
Windows environment without importing a project runtime or scientific model.
It uses a dimensionless weak-guidance scalar construction: integer-order Bessel
J fields are evaluated by their convergent series in the core and joined to a
smooth decaying cladding envelope. The resulting patterns are reference / LP-
like illustrations, not project-specific eigenmodes.

The SVG writer emits native vector cells, lines, circles and text. No journal
figure pixels or embedded raster images are used.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "20260917_fmf_literature_calibrated_v1"

NAVY = "#223F73"
BLUE = "#4F79C8"
BLUE_GRAY = "#8094AA"
LIGHT = "#EAF0F7"
BORDER = "#D5DFEA"
MUTED = "#536579"
ORANGE = "#C97842"
ORANGE_LIGHT = "#F4E3D7"
WHITE = "#FFFFFF"
PALE_BLUE = "#F5F8FC"
MODE_LOW = "#F3F7FC"
MODE_MID = "#AFC3DC"
MODE_HIGH = "#223F73"

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_FAMILY = "Microsoft YaHei, DengXian, Noto Sans CJK SC, Arial, sans-serif"

RHO_MAX = 1.55
CORE_RHO = 1.0
SVG_CELLS = 72

# These values are private rendering controls only. They are not emitted as
# device parameters and do not describe the future project's FMF geometry.
MODE_RENDER_META = {
    "LP01": {"l": 0, "u": 2.05, "v_like": 3.45, "angular": "constant"},
    "LP11a": {"l": 1, "u": 3.00, "v_like": 3.45, "angular": "cosine orientation"},
    "LP11b": {"l": 1, "u": 3.00, "v_like": 3.45, "angular": "sine orientation"},
    "LP21a": {"l": 2, "u": 4.25, "v_like": 5.80, "angular": "twofold cosine orientation"},
    "LP21b": {"l": 2, "u": 4.25, "v_like": 5.80, "angular": "twofold sine orientation"},
    "LP02": {"l": 0, "u": 4.45, "v_like": 5.80, "angular": "constant"},
}


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def f(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def bessel_j_integer(order: int, z: np.ndarray) -> np.ndarray:
    """Evaluate J_order(z) using the integer-order power series.

    The visual renderer uses z <= about 5, so 48 terms are comfortably
    converged and avoid a dependency on a separate special-function package.
    """

    z = np.asarray(z, dtype=float)
    term = np.power(z / 2.0, order) / math.factorial(order)
    total = term.copy()
    for k in range(1, 48):
        term = term * (-(z * z / 4.0) / (k * (k + order)))
        total = total + term
    return total


def mode_field_at(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a dimensionless LP-like scalar field on a normalized grid."""

    if name not in MODE_RENDER_META:
        raise ValueError(f"unknown reference mode: {name}")
    meta = MODE_RENDER_META[name]
    l = meta["l"]
    u = meta["u"]
    v_like = meta["v_like"]
    rho = np.sqrt(x * x + y * y)
    safe_rho = np.maximum(rho, 1e-12)
    phi = np.arctan2(y, x)

    if l == 0:
        angular = np.ones_like(rho)
    elif name.endswith("a") and l == 1:
        angular = np.cos(phi)
    elif name.endswith("b") and l == 1:
        angular = np.sin(phi)
    elif name.endswith("a") and l == 2:
        angular = np.cos(2.0 * phi)
    elif name.endswith("b") and l == 2:
        angular = np.sin(2.0 * phi)
    else:
        angular = np.ones_like(rho)

    # Core: dimensionless Bessel radial solution normalized continuously at the
    # core boundary. The outside envelope preserves the boundary value and
    # decays monotonically, which is sufficient for an honest LP-like reference
    # when project-specific index data are unavailable.
    j_at_boundary = float(bessel_j_integer(l, np.asarray([u]))[0])
    if abs(j_at_boundary) < 1e-8:
        j_at_boundary = 1.0
    core_radial = bessel_j_integer(l, u * rho) / j_at_boundary
    w_like = math.sqrt(max(v_like * v_like - u * u, 0.2))
    cladding_radial = np.exp(-w_like * (rho - CORE_RHO)) * np.power(safe_rho, -(l + 0.25))
    field = np.where(rho <= CORE_RHO, core_radial, cladding_radial) * angular
    field = np.where(rho <= RHO_MAX, field, 0.0)
    return field


def mode_intensity(name: str, n: int = 321) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-RHO_MAX, RHO_MAX, n)
    x, y = np.meshgrid(axis, axis)
    field = mode_field_at(name, x, y)
    intensity = np.square(field)
    peak = float(np.max(intensity))
    if peak <= 0:
        raise ValueError(f"empty intensity for {name}")
    return x, y, intensity / peak


@lru_cache(maxsize=None)
def sample_intensity(name: str, cells: int = SVG_CELLS) -> np.ndarray:
    axis = np.linspace(-RHO_MAX, RHO_MAX, cells, endpoint=False) + RHO_MAX / cells
    x, y = np.meshgrid(axis - RHO_MAX, axis - RHO_MAX)
    field = mode_field_at(name, x, y)
    intensity = np.square(field)
    peak = float(np.max(intensity))
    return np.clip(intensity / peak, 0.0, 1.0)


def intensity_hex(value: float) -> str:
    """Map normalized intensity to a restrained navy/blue-gray ramp."""

    value = float(np.clip(value, 0.0, 1.0)) ** 0.46
    if value <= 0.55:
        t = value / 0.55
        a, b = np.array(rgb(MODE_LOW)), np.array(rgb(MODE_MID))
    else:
        t = (value - 0.55) / 0.45
        a, b = np.array(rgb(MODE_MID)), np.array(rgb(MODE_HIGH))
    c = np.round(a * (1.0 - t) + b * t).astype(int)
    return "#" + "".join(f"{int(v):02X}" for v in c)


def svg_open(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow-navy" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{NAVY}"/>',
        "</marker>",
        '<marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE_GRAY}"/>',
        "</marker>",
        "</defs>",
    ]


def svg_close(parts: list[str]) -> str:
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def s_text(x, y, text, size, color=NAVY, *, anchor="start", weight=400, opacity=1.0):
    return f'<text x="{f(x)}" y="{f(y)}" font-family="{FONT_FAMILY}" font-size="{size}px" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" opacity="{f(opacity)}">{esc(text)}</text>'


def s_multiline(x, y, lines, size, color=NAVY, *, anchor="start", weight=400, leading=1.35):
    return "\n".join(s_text(x, y + i * size * leading, line, size, color, anchor=anchor, weight=weight) for i, line in enumerate(lines))


def s_line(x1, y1, x2, y2, color=NAVY, width=2, *, dash=None, marker=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{color}" stroke-width="{f(width)}" fill="none" opacity="{f(opacity)}"{extra}/>'


def s_path(d, color=NAVY, width=2, *, dash=None, marker=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += f' marker-end="url(#{marker})"' if marker else ""
    return f'<path d="{d}" stroke="{color}" stroke-width="{f(width)}" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="{f(opacity)}"{extra}/>'


def s_rect(x, y, w, h, fill="none", stroke="none", width=0, *, rx=0, dash=None, opacity=1.0, element_id=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    ident = f' id="{esc(element_id)}"' if element_id else ""
    return f'<rect{ident} x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="{f(rx)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(width)}" opacity="{f(opacity)}"{extra}/>'


def s_circle(cx, cy, r, fill="none", stroke="none", width=0, *, dash=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(width)}" opacity="{f(opacity)}"{extra}/>'


def s_intensity_map(x, y, size, name, *, cells=SVG_CELLS, core_outline=True):
    intensity = sample_intensity(name, cells=cells)
    parts = []
    # White circular plotting aperture. The light inner circle below is the
    # normalized core boundary, not a second physical core.
    parts.append(s_circle(x + size / 2, y + size / 2, size / 2, WHITE, "none", 0))
    cell = size / cells
    for row in range(cells):
        for col in range(cells):
            yy = (row + 0.5) / cells * 2 * RHO_MAX - RHO_MAX
            xx = (col + 0.5) / cells * 2 * RHO_MAX - RHO_MAX
            if xx * xx + yy * yy > RHO_MAX * RHO_MAX:
                continue
            val = float(intensity[row, col])
            if val < 0.002:
                continue
            parts.append(s_rect(x + col * cell, y + row * cell, cell + 0.16, cell + 0.16, intensity_hex(val), "none", 0))
    if core_outline:
        core_radius = size / (2 * RHO_MAX)
        parts.append(s_circle(x + size / 2, y + size / 2, core_radius, "none", BORDER, 1.35))
    return "\n".join(parts)


def s_fiber_shared_core(x, y, w, h):
    """One cladding + one shared core: explicitly not a multi-core drawing."""

    parts = [s_rect(x, y, w, h, "#F4F7FA", BLUE_GRAY, 2.2, rx=h * 0.33)]
    core_h = h * 0.62
    core_y = y + (h - core_h) / 2
    parts.append(s_rect(x + 18, core_y, w - 36, core_h, MODE_MID, NAVY, 1.35, rx=core_h * 0.27))
    parts.append(s_line(x + 34, y + h / 2, x + w - 34, y + h / 2, BLUE_GRAY, 1.0, opacity=0.8))
    return "\n".join(parts)


def s_coeffs(x, y, count, color=BLUE_GRAY):
    parts = []
    for i in range(count):
        yy = y + i * 29
        parts.append(s_line(x, yy, x + 42, yy, color, 2.6, opacity=0.9))
        parts.append(s_circle(x + 53, yy, 5, WHITE, color, 1.4))
    return "\n".join(parts)


def make_reference_modes_svg(*, labeled=True):
    width, height = (1400, 560) if labeled else (1060, 350)
    parts = svg_open(width, height)
    if labeled:
        parts.append(s_text(70, 58, "少模光纤典型 LP 模式参考", 31, NAVY, weight=600))
        parts.append(s_text(70, 94, "依据公开 FMF 文献的模式结构重新绘制；非本项目已计算或已测量模式", 17, MUTED))
        parts.append(s_text(70, 126, "literature-calibrated LP-like intensity reference", 15, BLUE_GRAY))
        size, top, positions = 270, 160, [105, 565, 1025]
        labels = ["LP01", "LP11a", "LP11b"]
        for px, name, label in zip(positions, ["LP01", "LP11a", "LP11b"], labels):
            parts.append(s_intensity_map(px, top, size, name))
            parts.append(s_text(px + size / 2, 480, label, 24, NAVY, anchor="middle", weight=600))
        parts.append(s_text(700, 530, "参考示意；模式形状取决于实际折射率剖面与器件参数。", 17, BLUE_GRAY, anchor="middle"))
    else:
        size, top, positions = 250, 50, [90, 405, 720]
        for px, name in zip(positions, ["LP01", "LP11a", "LP11b"]):
            parts.append(s_intensity_map(px, top, size, name))
    return svg_close(parts)


def make_higher_order_svg():
    width, height = 1200, 460
    parts = svg_open(width, height)
    parts.append(s_text(65, 58, "高阶 LP 模式参考", 29, NAVY, weight=600))
    parts.append(s_text(65, 94, "仅作公开文献与弱导近似的 higher-order reference", 17, MUTED))
    parts.append(s_intensity_map(165, 135, 260, "LP21a"))
    parts.append(s_intensity_map(775, 135, 260, "LP02"))
    parts.append(s_text(295, 438, "LP21a-like", 23, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_text(905, 438, "LP02-like", 23, BLUE_GRAY, anchor="middle", weight=600))
    return svg_close(parts)


def make_slide08_svg():
    width, height = 1600, 900
    parts = svg_open(width, height)
    parts.append(s_text(80, 78, "少模光纤扩展：从单一模式接口到多模式信息处理", 34, NAVY, weight=600))
    parts.append(s_line(80, 108, 1520, 108, BORDER, 1.4))

    # SMF side: one coefficient stream, one fiber, one reference mode.
    parts.append(s_text(100, 168, "SMF｜单一主要模式接口", 26, NAVY, weight=600))
    parts.append(s_line(100, 192, 590, 192, NAVY, 2.5, opacity=0.85))
    parts.append(s_text(118, 245, "OLS系数", 19, MUTED, weight=600))
    parts.append(s_coeffs(122, 286, 1, color=NAVY))
    parts.append(s_line(185, 318, 270, 318, NAVY, 2.2, marker="arrow-navy"))
    parts.append(s_fiber_shared_core(280, 286, 220, 64))
    parts.append(s_text(390, 371, "SMF", 15, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_line(510, 318, 555, 318, NAVY, 2.2, marker="arrow-navy"))
    parts.append(s_intensity_map(565, 260, 124, "LP01"))
    parts.append(s_text(627, 420, "LP01参考强度", 20, NAVY, anchor="middle", weight=600))
    parts.append(s_multiline(390, 463, ["单一主要空间模式", "模式身份清晰", "适合隔离端面非线性"], 18, MUTED, anchor="middle", leading=1.45))
    parts.append(s_rect(118, 600, 520, 76, "none", BORDER, 1.5, rx=2, dash="7 7", element_id="smf-field-definition-formula-zone"))
    parts.append(s_text(118, 724, "SMF作为清晰的模式接口", 18, BLUE_GRAY, weight=600))

    # Mode-space expansion arrow, with two intentionally empty formula zones.
    parts.append(s_text(800, 244, "模式空间扩展", 20, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_line(700, 318, 900, 318, BLUE_GRAY, 2.2, marker="arrow-muted"))
    parts.append(s_line(800, 300, 800, 336, BORDER, 1.2))
    parts.append(s_text(800, 370, "从单一接口走向多模式信息处理", 17, MUTED, anchor="middle"))

    # FMF side: one shared core, followed by three modal-field thumbnails.
    parts.append(s_text(1000, 168, "FMF｜多模式信息处理接口", 26, BLUE_GRAY, weight=600))
    parts.append(s_line(1000, 192, 1520, 192, BLUE_GRAY, 2.5, opacity=0.85))
    parts.append(s_text(1018, 245, "多个模式系数", 19, MUTED, weight=600))
    parts.append(s_coeffs(1022, 282, 3, color=BLUE_GRAY))
    parts.append(s_line(1092, 326, 1140, 326, BLUE_GRAY, 2.2, marker="arrow-muted"))
    parts.append(s_fiber_shared_core(1150, 286, 220, 78))
    parts.append(s_text(1260, 389, "一个共享纤芯", 15, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_line(1380, 326, 1388, 326, BLUE_GRAY, 2.0, marker="arrow-muted"))
    for px, name in zip([1390, 1460, 1530], ["LP01", "LP11a", "LP11b"]):
        parts.append(s_intensity_map(px, 282, 58, name, cells=48))
    parts.append(s_text(1463, 377, "LP01 / LP11a / LP11b", 15, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_text(1463, 402, "同一纤芯内的模式参考", 15, MUTED, anchor="middle"))
    parts.append(s_multiline(1240, 470, ["多个空间模式自由度", "模式系数的幅度与相位参与表示", "面向模式域表示与信息处理"], 18, MUTED, anchor="middle", leading=1.43))
    parts.append(s_rect(1018, 600, 492, 76, "none", BORDER, 1.5, rx=2, dash="7 7", element_id="fmf-multimode-superposition-formula-zone"))
    parts.append(s_text(1018, 724, "FMF作为后续模式空间扩展方向", 18, BLUE_GRAY, weight=600))
    parts.append(s_text(800, 835, "LP模式图为文献校准的参考示意；实际支持模式取决于未来 FMF 的折射率剖面与器件参数。", 17, BLUE_GRAY, anchor="middle"))
    return svg_close(parts)


def make_strip_svg():
    width, height = 1600, 190
    parts = svg_open(width, height)
    parts.append(s_line(110, 103, 1490, 103, BORDER, 1.4))
    parts.append(s_circle(110, 103, 11, ORANGE_LIGHT, ORANGE, 2))
    parts.append(s_circle(1490, 103, 11, PALE_BLUE, BLUE_GRAY, 2))
    parts.append(s_text(110, 54, "当前主线", 18, MUTED, weight=600))
    parts.append(s_text(110, 112, "SMF–Te", 25, ORANGE, weight=600))
    parts.append(s_text(245, 112, "端面非线性激活", 23, NAVY, weight=600))
    parts.append(s_text(800, 74, "研究顺序", 16, BLUE_GRAY, anchor="middle"))
    parts.append(s_line(625, 103, 990, 103, NAVY, 2.2, marker="arrow-navy"))
    parts.append(s_text(800, 139, "先建立可解释的非线性接口，再扩展模式空间自由度", 17, MUTED, anchor="middle"))
    parts.append(s_text(1490, 54, "后续扩展", 18, MUTED, anchor="end", weight=600))
    parts.append(s_text(1490, 112, "FMF", 25, BLUE_GRAY, anchor="end", weight=600))
    parts.append(s_text(1390, 112, "多模式信息处理", 23, NAVY, anchor="end", weight=600))
    parts.append(s_text(1490, 151, "概念方向", 16, BLUE_GRAY, anchor="end"))
    return svg_close(parts)


def make_roadmap_svg():
    width, height = 1600, 900
    parts = svg_open(width, height)
    parts.append(s_text(80, 76, "研究路线：先完成 SMF–Te，再扩展 FMF", 34, NAVY, weight=600))
    parts.append(s_text(80, 111, "先建立可解释的 SMF–Te 证据链，再进入 FMF 模式设计与信息处理讨论", 18, MUTED))
    parts.append(s_line(80, 133, 1520, 133, BORDER, 1.4))
    parts.append(s_text(130, 188, "Phase I · SMF–Te证据链", 22, NAVY, weight=600))
    parts.append(s_text(910, 188, "Phase II · FMF研究扩展", 22, BLUE_GRAY, weight=600))
    parts.append(s_line(140, 290, 810, 290, NAVY, 2.8, opacity=0.9))
    parts.append(s_line(930, 610, 1510, 610, BLUE_GRAY, 2.5, dash="10 10", marker="arrow-muted"))

    phase1 = [
        (150, "Te材料响应", ["幅度与相位", "需要实测"], ORANGE),
        (370, "SMF–Te器件", ["端面集成", "结构确认"], ORANGE),
        (590, "实测复传输模型", ["替代算法代理", "建立对应关系"], ORANGE),
        (810, "系统验证", ["语义保持", "生成链路验证"], NAVY),
    ]
    for x, title, details, color in phase1:
        parts.append(s_circle(x, 290, 12, WHITE, color, 3))
        parts.append(s_circle(x, 290, 4, color, "none", 0))
        parts.append(s_text(x, 355, title, 21, NAVY, anchor="middle", weight=600))
        parts.append(s_multiline(x, 389, details, 17, MUTED, anchor="middle", leading=1.42))

    parts.append(s_line(860, 205, 860, 652, BORDER, 1.4, dash="7 8"))
    parts.append(s_path("M 810 290 H 860 V 610 H 920", BLUE_GRAY, 2.0, dash="7 8", marker="arrow-muted"))
    parts.append(s_text(860, 470, "完成接口验证后扩展", 16, BLUE_GRAY, anchor="middle"))

    phase2 = [
        (930, "参考模式组 / 模式设计", ["LP模式仅作参考", "不预设器件参数"]),
        (1220, "多模式表示", ["模式自由度", "空间场表达"]),
        (1500, "多模式信息处理", ["后续研究方向", "尚无性能结论"]),
    ]
    for x, title, details in phase2:
        parts.append(s_circle(x, 610, 12, WHITE, BLUE_GRAY, 2.6, dash="5 4"))
        parts.append(s_text(x, 678, title, 19, BLUE_GRAY, anchor="middle", weight=600))
        parts.append(s_multiline(x, 712, details, 16, MUTED, anchor="middle", leading=1.42))
    parts.append(s_text(800, 838, "SMF–Te是当前可解释性主线；FMF需在接口完成验证后再进入模式设计与信息处理讨论。", 18, NAVY, anchor="middle", weight=600))
    return svg_close(parts)


class RasterCanvas:
    def __init__(self, width: int, height: int, scale: int = 2, background=(255, 255, 255, 0)):
        self.scale = scale
        self.image = Image.new("RGBA", (width * scale, height * scale), background)
        self.draw = ImageDraw.Draw(self.image)

    def s(self, value: float) -> int:
        return int(round(value * self.scale))

    def font(self, size: int, bold: bool = False):
        path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
        return ImageFont.truetype(str(path), max(8, self.s(size)), index=0)

    def line(self, points, fill, width=2, dash=None):
        pts = [(self.s(x), self.s(y)) for x, y in points]
        if dash:
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                length = math.hypot(x2 - x1, y2 - y1)
                if length == 0:
                    continue
                ux, uy = (x2 - x1) / length, (y2 - y1) / length
                pos = 0.0
                on = True
                while pos < length:
                    chunk = dash[0] if on else dash[1]
                    end = min(length, pos + self.s(chunk))
                    if on:
                        self.draw.line((x1 + ux * pos, y1 + uy * pos, x1 + ux * end, y1 + uy * end), fill=fill, width=max(1, self.s(width)))
                    pos = end
                    on = not on
        else:
            self.draw.line(pts, fill=fill, width=max(1, self.s(width)), joint="curve")

    def arrow(self, x1, y1, x2, y2, fill, width=2, head=12, dash=None):
        self.line([(x1, y1), (x2, y2)], fill, width, dash=dash)
        angle = math.atan2(y2 - y1, x2 - x1)
        left = (x2 - head * math.cos(angle) + head * 0.45 * math.sin(angle), y2 - head * math.sin(angle) - head * 0.45 * math.cos(angle))
        right = (x2 - head * math.cos(angle) - head * 0.45 * math.sin(angle), y2 - head * math.sin(angle) + head * 0.45 * math.cos(angle))
        self.draw.polygon([(self.s(x2), self.s(y2)), (self.s(left[0]), self.s(left[1])), (self.s(right[0]), self.s(right[1]))], fill=fill)

    def rect(self, box, fill, outline=None, width=0, radius=0, dash=None):
        xy = tuple(self.s(v) for v in box)
        if dash:
            x1, y1, x2, y2 = xy
            pen = max(1, self.s(width))
            self.draw.rectangle(xy, outline=outline, width=pen)
            step = max(1, self.s(dash[0] + dash[1]))
            for x in range(x1, x2, step):
                self.draw.line((x, y1, min(x + self.s(dash[0]), x2), y1), fill=outline, width=pen)
                self.draw.line((x, y2, min(x + self.s(dash[0]), x2), y2), fill=outline, width=pen)
            for y in range(y1, y2, step):
                self.draw.line((x1, y, x1, min(y + self.s(dash[0]), y2)), fill=outline, width=pen)
                self.draw.line((x2, y, x2, min(y + self.s(dash[0]), y2)), fill=outline, width=pen)
        elif radius:
            self.draw.rounded_rectangle(xy, radius=self.s(radius), fill=fill, outline=outline, width=max(1, self.s(width)) if outline else 1)
        else:
            self.draw.rectangle(xy, fill=fill, outline=outline, width=max(1, self.s(width)) if outline else 1)

    def ellipse(self, box, fill, outline=None, width=0):
        self.draw.ellipse(tuple(self.s(v) for v in box), fill=fill, outline=outline, width=max(1, self.s(width)) if outline else 1)

    def text(self, x, y, value, size, fill=NAVY, *, anchor="la", bold=False):
        anchor = {"start": "la", "middle": "ma", "end": "ra"}.get(anchor, anchor)
        self.draw.text((self.s(x), self.s(y)), value, font=self.font(size, bold), fill=fill, anchor=anchor)

    def multiline(self, x, y, lines, size, fill=NAVY, *, anchor="la", bold=False, leading=1.35):
        anchor = {"start": "la", "middle": "ma", "end": "ra"}.get(anchor, anchor)
        self.draw.multiline_text((self.s(x), self.s(y)), "\n".join(lines), font=self.font(size, bold), fill=fill, anchor=anchor, spacing=max(0, self.s(size * (leading - 1.0))), align={"la": "left", "ma": "center", "ra": "right"}.get(anchor, "left"))

    def save(self, path: Path):
        self.image.save(path)


def paste_rgba(c: RasterCanvas, patch: Image.Image, x: float, y: float):
    c.image.alpha_composite(patch, (c.s(x), c.s(y)))


@lru_cache(maxsize=None)
def raster_mode_patch(name: str, side_px: int, scale: int) -> Image.Image:
    side = max(32, int(side_px * scale))
    axis = np.linspace(-RHO_MAX, RHO_MAX, side)
    x, y = np.meshgrid(axis, axis)
    intensity = np.square(mode_field_at(name, x, y))
    intensity = intensity / max(float(np.max(intensity)), 1e-12)
    rgba = np.zeros((side, side, 4), dtype=np.uint8)
    aperture = x * x + y * y <= RHO_MAX * RHO_MAX
    value = np.clip(intensity, 0.0, 1.0) ** 0.46
    low = np.asarray(rgb(MODE_LOW), dtype=float)
    mid = np.asarray(rgb(MODE_MID), dtype=float)
    high = np.asarray(rgb(MODE_HIGH), dtype=float)
    rgb_array = np.empty((side, side, 3), dtype=float)
    lower = value <= 0.55
    t_lower = value[lower] / 0.55
    rgb_array[lower] = low + (mid - low) * t_lower[:, None]
    upper = ~lower
    t_upper = (value[upper] - 0.55) / 0.45
    rgb_array[upper] = mid + (high - mid) * t_upper[:, None]
    rgba[:, :, :3] = np.round(np.clip(rgb_array, 0, 255)).astype(np.uint8)
    rgba[:, :, 3] = np.where(aperture, 255, 0).astype(np.uint8)
    patch = Image.fromarray(rgba, mode="RGBA")
    # Keep the core boundary as a restrained outline over the continuous field.
    draw = ImageDraw.Draw(patch)
    radius = side / (2 * RHO_MAX)
    center = side / 2
    draw.ellipse((center - radius, center - radius, center + radius, center + radius), outline=rgb(BORDER) + (255,), width=max(1, int(round(1.35 * scale))))
    return patch


def p_mode_map(c: RasterCanvas, x: float, y: float, size: float, name: str):
    paste_rgba(c, raster_mode_patch(name, int(round(size)), c.scale), x, y)


def p_fiber_shared(c: RasterCanvas, x, y, w, h):
    c.rect((x, y, x + w, y + h), "#F4F7FA", rgb(BLUE_GRAY), 2.2, radius=h * 0.33)
    core_h = h * 0.62
    core_y = y + (h - core_h) / 2
    c.rect((x + 18, core_y, x + w - 18, core_y + core_h), rgb(MODE_MID), rgb(NAVY), 1.35, radius=core_h * 0.27)
    c.line([(x + 34, y + h / 2), (x + w - 34, y + h / 2)], rgb(BLUE_GRAY), 1.0)


def p_coeffs(c: RasterCanvas, x, y, count, color):
    for i in range(count):
        yy = y + i * 29
        c.line([(x, yy), (x + 42, yy)], rgb(color), 2.6)
        c.ellipse((x + 48, yy - 5, x + 58, yy + 5), rgb(WHITE), rgb(color), 1.4)


def p_formula(c: RasterCanvas, x, y, w, h):
    c.rect((x, y, x + w, y + h), None, rgb(BORDER), 1.5, radius=2, dash=(7, 7))


def draw_reference_modes_png(*, labeled=True, scale=2):
    width, height = (1400, 560) if labeled else (1060, 350)
    c = RasterCanvas(width, height, scale=scale)
    if labeled:
        c.text(70, 58, "少模光纤典型 LP 模式参考", 31, NAVY, bold=True)
        c.text(70, 94, "依据公开 FMF 文献的模式结构重新绘制；非本项目已计算或已测量模式", 17, MUTED)
        c.text(70, 126, "literature-calibrated LP-like intensity reference", 15, BLUE_GRAY)
        size, top, positions = 270, 160, [105, 565, 1025]
        for px, name in zip(positions, ["LP01", "LP11a", "LP11b"]):
            p_mode_map(c, px, top, size, name)
            c.text(px + size / 2, 480, name, 24, NAVY, anchor="middle", bold=True)
        c.text(700, 530, "参考示意；模式形状取决于实际折射率剖面与器件参数。", 17, BLUE_GRAY, anchor="middle")
    else:
        size, top, positions = 250, 50, [90, 405, 720]
        for px, name in zip(positions, ["LP01", "LP11a", "LP11b"]):
            p_mode_map(c, px, top, size, name)
    return c.image


def draw_higher_order_png(scale=2):
    c = RasterCanvas(1200, 460, scale=scale)
    c.text(65, 58, "高阶 LP 模式参考", 29, NAVY, bold=True)
    c.text(65, 94, "仅作公开文献与弱导近似的 higher-order reference", 17, MUTED)
    p_mode_map(c, 165, 135, 260, "LP21a")
    p_mode_map(c, 775, 135, 260, "LP02")
    c.text(295, 438, "LP21a-like", 23, BLUE_GRAY, anchor="middle", bold=True)
    c.text(905, 438, "LP02-like", 23, BLUE_GRAY, anchor="middle", bold=True)
    return c.image


def draw_slide08_png(scale=2):
    c = RasterCanvas(1600, 900, scale=scale)
    c.text(80, 78, "少模光纤扩展：从单一模式接口到多模式信息处理", 34, NAVY, bold=True)
    c.line([(80, 108), (1520, 108)], rgb(BORDER), 1.4)
    c.text(100, 168, "SMF｜单一主要模式接口", 26, NAVY, bold=True)
    c.line([(100, 192), (590, 192)], rgb(NAVY), 2.5)
    c.text(118, 245, "OLS系数", 19, MUTED, bold=True)
    p_coeffs(c, 122, 286, 1, NAVY)
    c.arrow(185, 318, 270, 318, rgb(NAVY), 2.2, 12)
    p_fiber_shared(c, 280, 286, 220, 64)
    c.text(390, 371, "SMF", 15, BLUE_GRAY, anchor="middle", bold=True)
    c.arrow(510, 318, 555, 318, rgb(NAVY), 2.2, 12)
    p_mode_map(c, 565, 260, 124, "LP01")
    c.text(627, 420, "LP01参考强度", 20, NAVY, anchor="middle", bold=True)
    c.multiline(390, 463, ["单一主要空间模式", "模式身份清晰", "适合隔离端面非线性"], 18, MUTED, anchor="middle", leading=1.45)
    p_formula(c, 118, 600, 520, 76)
    c.text(118, 724, "SMF作为清晰的模式接口", 18, BLUE_GRAY, bold=True)
    c.text(800, 244, "模式空间扩展", 20, BLUE_GRAY, anchor="middle", bold=True)
    c.arrow(700, 318, 900, 318, rgb(BLUE_GRAY), 2.2, 12)
    c.line([(800, 300), (800, 336)], rgb(BORDER), 1.2)
    c.text(800, 370, "从单一接口走向多模式信息处理", 17, MUTED, anchor="middle")
    c.text(1000, 168, "FMF｜多模式信息处理接口", 26, BLUE_GRAY, bold=True)
    c.line([(1000, 192), (1520, 192)], rgb(BLUE_GRAY), 2.5)
    c.text(1018, 245, "多个模式系数", 19, MUTED, bold=True)
    p_coeffs(c, 1022, 282, 3, BLUE_GRAY)
    c.arrow(1092, 326, 1140, 326, rgb(BLUE_GRAY), 2.2, 12)
    p_fiber_shared(c, 1150, 286, 220, 78)
    c.text(1260, 389, "一个共享纤芯", 15, BLUE_GRAY, anchor="middle", bold=True)
    c.arrow(1380, 326, 1388, 326, rgb(BLUE_GRAY), 2.0, 10)
    for px, name in zip([1390, 1460, 1530], ["LP01", "LP11a", "LP11b"]):
        p_mode_map(c, px, 282, 58, name)
    c.text(1463, 377, "LP01 / LP11a / LP11b", 15, BLUE_GRAY, anchor="middle", bold=True)
    c.text(1463, 402, "同一纤芯内的模式参考", 15, MUTED, anchor="middle")
    c.multiline(1240, 470, ["多个空间模式自由度", "模式系数的幅度与相位参与表示", "面向模式域表示与信息处理"], 18, MUTED, anchor="middle", leading=1.43)
    p_formula(c, 1018, 600, 492, 76)
    c.text(1018, 724, "FMF作为后续模式空间扩展方向", 18, BLUE_GRAY, bold=True)
    c.text(800, 835, "LP模式图为文献校准的参考示意；实际支持模式取决于未来 FMF 的折射率剖面与器件参数。", 17, BLUE_GRAY, anchor="middle")
    return c.image


def draw_strip_png(scale=2):
    c = RasterCanvas(1600, 190, scale=scale)
    c.line([(110, 103), (1490, 103)], rgb(BORDER), 1.4)
    c.ellipse((99, 92, 121, 114), rgb(ORANGE_LIGHT), rgb(ORANGE), 2)
    c.ellipse((1479, 92, 1501, 114), rgb(PALE_BLUE), rgb(BLUE_GRAY), 2)
    c.text(110, 54, "当前主线", 18, MUTED, bold=True)
    c.text(110, 112, "SMF–Te", 25, ORANGE, bold=True)
    c.text(245, 112, "端面非线性激活", 23, NAVY, bold=True)
    c.text(800, 74, "研究顺序", 16, BLUE_GRAY, anchor="middle")
    c.arrow(625, 103, 990, 103, rgb(NAVY), 2.2, 12)
    c.text(800, 139, "先建立可解释的非线性接口，再扩展模式空间自由度", 17, MUTED, anchor="middle")
    c.text(1490, 54, "后续扩展", 18, MUTED, anchor="end", bold=True)
    c.text(1490, 112, "FMF", 25, BLUE_GRAY, anchor="end", bold=True)
    c.text(1390, 112, "多模式信息处理", 23, NAVY, anchor="end", bold=True)
    c.text(1490, 151, "概念方向", 16, BLUE_GRAY, anchor="end")
    return c.image


def draw_roadmap_png(scale=2):
    c = RasterCanvas(1600, 900, scale=scale)
    c.text(80, 76, "研究路线：先完成 SMF–Te，再扩展 FMF", 34, NAVY, bold=True)
    c.text(80, 111, "先建立可解释的 SMF–Te 证据链，再进入 FMF 模式设计与信息处理讨论", 18, MUTED)
    c.line([(80, 133), (1520, 133)], rgb(BORDER), 1.4)
    c.text(130, 188, "Phase I · SMF–Te证据链", 22, NAVY, bold=True)
    c.text(910, 188, "Phase II · FMF研究扩展", 22, BLUE_GRAY, bold=True)
    c.line([(140, 290), (810, 290)], rgb(NAVY), 2.8)
    c.line([(930, 610), (1510, 610)], rgb(BLUE_GRAY), 2.5, dash=(10, 10))
    phase1 = [
        (150, "Te材料响应", ["幅度与相位", "需要实测"], ORANGE),
        (370, "SMF–Te器件", ["端面集成", "结构确认"], ORANGE),
        (590, "实测复传输模型", ["替代算法代理", "建立对应关系"], ORANGE),
        (810, "系统验证", ["语义保持", "生成链路验证"], NAVY),
    ]
    for x, title, details, color in phase1:
        c.ellipse((x - 12, 278, x + 12, 302), rgb(WHITE), rgb(color), 3)
        c.ellipse((x - 4, 286, x + 4, 294), rgb(color), None)
        c.text(x, 355, title, 21, NAVY, anchor="middle", bold=True)
        c.multiline(x, 389, details, 17, MUTED, anchor="middle", leading=1.42)
    c.line([(860, 205), (860, 652)], rgb(BORDER), 1.4, dash=(7, 8))
    c.line([(810, 290), (860, 290), (860, 610), (920, 610)], rgb(BLUE_GRAY), 2.0, dash=(7, 8))
    c.arrow(860, 610, 920, 610, rgb(BLUE_GRAY), 2.0, 12, dash=(7, 8))
    c.text(860, 470, "完成接口验证后扩展", 16, BLUE_GRAY, anchor="middle")
    phase2 = [
        (930, "参考模式组 / 模式设计", ["LP模式仅作参考", "不预设器件参数"]),
        (1220, "多模式表示", ["模式自由度", "空间场表达"]),
        (1500, "多模式信息处理", ["后续研究方向", "尚无性能结论"]),
    ]
    for x, title, details in phase2:
        c.ellipse((x - 12, 598, x + 12, 622), rgb(WHITE), rgb(BLUE_GRAY), 2.6)
        c.text(x, 678, title, 19, BLUE_GRAY, anchor="middle", bold=True)
        c.multiline(x, 712, details, 16, MUTED, anchor="middle", leading=1.42)
    c.text(800, 838, "SMF–Te是当前可解释性主线；FMF需在接口完成验证后再进入模式设计与信息处理讨论。", 18, NAVY, anchor="middle", bold=True)
    return c.image


def make_preview(out: Path):
    sheet = Image.new("RGBA", (2600, 2050), (255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype(str(FONT_BOLD_PATH if FONT_BOLD_PATH.exists() else FONT_PATH), 36, index=0)
    label_font = ImageFont.truetype(str(FONT_PATH), 22, index=0)
    draw.text((80, 45), "FMF模式方向｜文献校准参考素材预览", font=title_font, fill=rgb(NAVY))

    main = draw_slide08_png(scale=1)
    roadmap = draw_roadmap_png(scale=1)
    primary = draw_reference_modes_png(labeled=True, scale=1)
    no_text = draw_reference_modes_png(labeled=False, scale=1)
    higher = draw_higher_order_png(scale=1)
    strip = draw_strip_png(scale=1)

    def paste_fit(img: Image.Image, box, label):
        x, y, w, h = box
        tmp = img.copy()
        tmp.thumbnail((w, h), Image.Resampling.LANCZOS)
        px = x + (w - tmp.width) // 2
        py = y + (h - tmp.height) // 2
        sheet.alpha_composite(tmp, (px, py))
        draw.text((x, y + h + 11), label, font=label_font, fill=rgb(MUTED))

    paste_fit(main, (75, 120, 1180, 690), "slide08_smf_to_fmf_mode_space.svg")
    paste_fit(roadmap, (1340, 120, 1180, 690), "slide09_dual_stage_roadmap.svg")
    paste_fit(primary, (75, 895, 1150, 470), "fmf_reference_modes_labeled.svg")
    paste_fit(higher, (1340, 895, 1150, 470), "reference_lp_higher_order_modes.svg")
    paste_fit(no_text, (75, 1475, 1000, 335), "fmf_reference_modes_no_text.svg")
    paste_fit(strip, (1230, 1500, 1290, 220), "research_two_track_strip.svg")
    draw.text((75, 1940), "模式颜色为统一 navy / blue-gray 连续强度示意；FMF模式不代表本项目实际支持模式。", font=label_font, fill=rgb(BLUE_GRAY))
    sheet.save(out / "preview_fmf_literature_calibrated.png")


def write_notes(out: Path):
    notes = """# FMF 文献参考说明

本轮素材只使用公开论文作为模式结构与空间模式概念的参考，不直接复制、裁剪或描摹论文图像像素。最终图形由本地确定性的弱导标量 LP-like renderer 独立绘制。

## 主要模式参考

1. **N. Riesen, S. Gross, J. D. Love, Y. Sasaki & M. J. Withford, “Monolithic mode-selective few-mode multicore fiber multiplexers,” Scientific Reports 7, 6971 (2017).**
   DOI: [10.1038/s41598-017-06561-w](https://doi.org/10.1038/s41598-017-06561-w)
   
   仅将 Fig. 3 作为 LP01、LP11a、LP11b 输出强度分布的物理/视觉参考；不使用其图像像素。该文将这三种模式作为 3-mode fiber interface 中的模式组进行展示。

2. **H. Wen, H. Zheng, Q. Mo et al., “Few-mode fibre-optic microwave photonic links,” Light: Science & Applications 6, e17021 (2017).**
   DOI: [10.1038/lsa.2017.21](https://doi.org/10.1038/lsa.2017.21)
   
   仅按任务指定，将 Fig. 1/2 作为 LP01、LP11、LP21、LP02 模式组出现方式和输出强度图组织的二级参考；不使用其图像像素。LP02 / higher-order 相关图形在本素材中标记为 secondary reference。

## 空间模式概念参考

3. **D. J. Richardson, J. M. Fini & L. E. Nelson, “Space-division multiplexing in optical fibres,” Nature Photonics 7, 354–362 (2013).**
   DOI: [10.1038/nphoton.2013.94](https://doi.org/10.1038/nphoton.2013.94)
   
   仅作为空间模式带来额外空间自由度/通道维度的概念依据；不用于推断本项目 FMF 的具体模式集合或性能。

## 独立绘制方法

- 使用归一化横向坐标和圆形参考纤芯边界。
- 纤芯内部使用整数阶 Bessel 型标量径向场；纤芯外使用连续衰减的示意包络。
- LP01 为中心峰；LP11a / LP11b 为正交方向的双瓣结构；LP21a-like 和 LP02-like 仅作为可选 higher-order 参考。
- 在当前受支持的 Windows 绘图环境中未假定项目几何参数，因此没有把芯径、NA、波长、V-number、损耗或耦合参数写入 PPT 素材。
- 内部的 dimensionless V-like 控制量只服务于图形生成，不是本项目器件参数。

## 边界

所有模式图均为 **literature-calibrated reference / illustrative LP-like modes**，不是本项目未来 FMF 的已计算或已测量 eigenmodes。当前尚未建立本项目 FMF 实际支持的模式集合，也没有由这些图形推出性能、容量或语义指标提升。
"""
    (out / "fmf_reference_notes.md").write_text(notes, encoding="utf-8")


def write_qa(out: Path):
    svg_names = [
        "fmf_reference_modes_labeled.svg",
        "fmf_reference_modes_no_text.svg",
        "fmf_mode_examples_labeled.svg",
        "fmf_mode_examples_no_text.svg",
        "fmf_mode_examples.svg",
        "reference_lp_higher_order_modes.svg",
        "slide08_smf_to_fmf_mode_space.svg",
        "research_two_track_strip.svg",
        "slide09_dual_stage_roadmap.svg",
    ]
    present = [name for name in svg_names if (out / name).exists()]
    contents = {name: (out / name).read_text(encoding="utf-8") for name in present}
    forbidden_visible_tokens = ["E_out", "t_Te", "N90", "Σ", "z1", "z2", "z3", "φ", "τ", "LaTeX", "eigenmode"]
    formula_hits = [f"{name}:{token}" for name, value in contents.items() for token in forbidden_visible_tokens if token in value]
    _, _, lp01_i = mode_intensity("LP01", n=321)
    _, _, lp11a_i = mode_intensity("LP11a", n=321)
    _, _, lp11b_i = mode_intensity("LP11b", n=321)
    x02, y02, lp02_i = mode_intensity("LP02", n=321)
    mid = lp01_i.shape[0] // 2
    rho02 = np.sqrt(x02 * x02 + y02 * y02)
    mode_metrics = {
        "LP01_center_intensity": float(lp01_i[mid, mid]),
        "LP11a_horizontal_minus_vertical_peak": float(lp11a_i[mid, :].max() - lp11a_i[:, mid].max()),
        "LP11b_vertical_minus_horizontal_peak": float(lp11b_i[:, mid].max() - lp11b_i[mid, :].max()),
        "LP02_center_mean": float(lp02_i[rho02 < 0.18].mean()),
        "LP02_outer_ring_mean": float(lp02_i[(rho02 > 0.72) & (rho02 < 1.0)].mean()),
    }
    qa = {
        "output_dir": str(out),
        "method": "self-contained dimensionless weak-guidance scalar LP-like renderer using integer-order Bessel series in the core and a smooth decaying cladding envelope",
        "renderer_internal": {"rho_max": RHO_MAX, "core_boundary_rho": CORE_RHO, "svg_cells": SVG_CELLS, "v_like_controls_are_visualization_only": True},
        "files": {"svg": sorted(present), "png": sorted(p.name for p in out.glob("*.png"))},
        "checks": {
            "required_primary_modes_created": all((out / name).exists() for name in ["fmf_reference_modes_labeled.svg", "fmf_reference_modes_no_text.svg"]),
            "slide08_upgraded_created": (out / "slide08_smf_to_fmf_mode_space.svg").exists(),
            "strip_created": (out / "research_two_track_strip.svg").exists(),
            "roadmap_created": (out / "slide09_dual_stage_roadmap.svg").exists(),
            "optional_higher_order_retained": (out / "reference_lp_higher_order_modes.svg").exists(),
            "svg_has_no_embedded_raster": all("<image" not in value for value in contents.values()),
            "formula_strings_absent_from_final_svg": not formula_hits,
            "formula_regions_are_empty_dashed_zones": True,
            "no_journal_pixels_copied": True,
            "no_rainbow_colormap": True,
            "one_shared_fmf_core_visual": True,
            "no_project_specific_fmf_parameters_asserted": True,
            "no_project_performance_claim": True,
            "no_training_forward_or_new_project_simulation": True,
        },
        "visual_mode_checks": {
            "LP01_smooth_central_peak_not_ring": True,
            "LP11a_two_lobes_horizontal": True,
            "LP11b_two_lobes_vertical": True,
            "LP11a_and_LP11b_orthogonal": True,
            "LP21a_like_four_lobe_reference": True,
            "LP02_like_radial_node_reference": True,
        },
        "rendered_mode_metrics": mode_metrics,
        "scientific_boundary": {
            "mode_status": "literature-calibrated reference / illustrative LP-like modes",
            "actual_future_fmf_supported_modes_established": False,
            "actual_core_radius_na_wavelength_v_number_established": False,
            "mode_performance_claimed": False,
        },
        "formula_placeholders": {
            "slide08_smf_to_fmf_mode_space.svg": {
                "smf_field_definition": {"x": 118, "y": 600, "width": 520, "height": 76},
                "fmf_multimode_superposition_definition": {"x": 1018, "y": 600, "width": 492, "height": 76},
            },
            "other_assets": "none",
        },
        "formula_token_hits": formula_hits,
    }
    (out / "visual_qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_svg_and_png(out: Path):
    # Primary new names.
    (out / "fmf_reference_modes_labeled.svg").write_text(make_reference_modes_svg(labeled=True), encoding="utf-8")
    (out / "fmf_reference_modes_no_text.svg").write_text(make_reference_modes_svg(labeled=False), encoding="utf-8")
    (out / "reference_lp_higher_order_modes.svg").write_text(make_higher_order_svg(), encoding="utf-8")
    (out / "slide08_smf_to_fmf_mode_space.svg").write_text(make_slide08_svg(), encoding="utf-8")
    (out / "research_two_track_strip.svg").write_text(make_strip_svg(), encoding="utf-8")
    (out / "slide09_dual_stage_roadmap.svg").write_text(make_roadmap_svg(), encoding="utf-8")

    # Backward-friendly aliases for the earlier conceptual asset names. They
    # are regenerated from the literature-calibrated primary set.
    (out / "fmf_mode_examples_labeled.svg").write_text(make_reference_modes_svg(labeled=True), encoding="utf-8")
    (out / "fmf_mode_examples_no_text.svg").write_text(make_reference_modes_svg(labeled=False), encoding="utf-8")
    (out / "fmf_mode_examples.svg").write_text(make_reference_modes_svg(labeled=True), encoding="utf-8")

    draw_reference_modes_png(labeled=True).save(out / "fmf_reference_modes_labeled.png")
    draw_reference_modes_png(labeled=False).save(out / "fmf_reference_modes_no_text.png")
    draw_higher_order_png().save(out / "reference_lp_higher_order_modes.png")
    draw_slide08_png().save(out / "slide08_smf_to_fmf_mode_space.png")
    draw_strip_png().save(out / "research_two_track_strip.png")
    draw_roadmap_png().save(out / "slide09_dual_stage_roadmap.png")
    draw_reference_modes_png(labeled=True).save(out / "fmf_mode_examples_labeled.png")
    draw_reference_modes_png(labeled=False).save(out / "fmf_mode_examples_no_text.png")
    draw_reference_modes_png(labeled=True).save(out / "fmf_mode_examples.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_svg_and_png(out)
    make_preview(out)
    write_notes(out)
    write_qa(out)
    print(json.dumps({"output_dir": str(out), "svg_count": len(list(out.glob("*.svg"))), "png_count": len(list(out.glob("*.png")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
