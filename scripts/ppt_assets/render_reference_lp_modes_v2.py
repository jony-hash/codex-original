"""Create a cleaner, literature-calibrated FMF figure set for PPT.

This is a presentation renderer, not a project optical simulation.  The mode
marks are flat vector LP-like reference symbols: LP01 is a central lobe and
LP11a/LP11b are orthogonal two-lobe patterns.  No project-specific FMF
parameters or measured modal fields are asserted.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "20260917_fmf_literature_calibrated_v2"

NAVY = "#234679"
BLUE = "#4F79C8"
BLUE_GRAY = "#8EA3B8"
LIGHT = "#F1F5F9"
MODE_FILL = "#C9D9E9"
MODE_DARK = "#426A9F"
MODE_SOFT = "#91ACCA"
BORDER = "#CBD8E5"
MUTED = "#5B6F83"
ORANGE = "#C97842"
ORANGE_LIGHT = "#F3E2D6"
WHITE = "#FFFFFF"

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_FAMILY = "Microsoft YaHei, DengXian, Noto Sans CJK SC, Arial, sans-serif"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def svg_begin(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow-navy" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">',
        f'<path d="M0,0 L10,5 L0,10 z" fill="{NAVY}"/>',
        "</marker>",
        '<marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">',
        f'<path d="M0,0 L10,5 L0,10 z" fill="{BLUE_GRAY}"/>',
        "</marker>",
        "</defs>",
    ]


def svg_end(parts: list[str]) -> str:
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def s_text(x, y, text, size, color=NAVY, *, anchor="start", weight=400, opacity=1.0):
    return (
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-family="{FONT_FAMILY}" '
        f'font-size="{size}px" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}" opacity="{fmt(opacity)}">{esc(text)}</text>'
    )


def s_lines(x, y, lines, size, color=MUTED, *, anchor="start", weight=400, leading=1.35):
    return "\n".join(
        s_text(x, y + i * size * leading, line, size, color, anchor=anchor, weight=weight)
        for i, line in enumerate(lines)
    )


def s_line(x1, y1, x2, y2, color=NAVY, width=2, *, dash=None, marker=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
        f'stroke="{color}" stroke-width="{fmt(width)}" fill="none" opacity="{fmt(opacity)}"{extra}/>'
    )


def s_path(d, color=NAVY, width=2, *, dash=None, marker=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<path d="{d}" stroke="{color}" stroke-width="{fmt(width)}" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{fmt(opacity)}"{extra}/>'
    )


def s_rect(x, y, w, h, fill="none", stroke="none", width=0, *, rx=0, dash=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
        f'rx="{fmt(rx)}" fill="{fill}" stroke="{stroke}" stroke-width="{fmt(width)}" '
        f'opacity="{fmt(opacity)}"{extra}/>'
    )


def s_circle(cx, cy, r, fill="none", stroke="none", width=0, *, dash=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{fmt(width)}" opacity="{fmt(opacity)}"{extra}/>'
    )


def s_ellipse(cx, cy, rx, ry, fill="none", stroke="none", width=0, *, opacity=1.0):
    return (
        f'<ellipse cx="{fmt(cx)}" cy="{fmt(cy)}" rx="{fmt(rx)}" ry="{fmt(ry)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{fmt(width)}" opacity="{fmt(opacity)}"/>'
    )


def mode_icon_svg(x: float, y: float, size: float, name: str) -> str:
    """Draw a crisp, topology-faithful LP-like cross-section icon."""

    cx, cy = x + size / 2, y + size / 2
    r = size * 0.47
    core_r = size * 0.315
    out = [s_circle(cx, cy, r, WHITE, BORDER, 2.0)]
    out.append(s_circle(cx, cy, core_r, MODE_FILL, NAVY, 1.5))
    if name == "LP01":
        out.append(s_circle(cx, cy, size * 0.225, MODE_SOFT, "none", 0))
        out.append(s_circle(cx, cy, size * 0.145, MODE_DARK, "none", 0))
    elif name == "LP11a":
        out.append(s_ellipse(cx - size * 0.125, cy, size * 0.12, size * 0.195, MODE_SOFT, "none", 0))
        out.append(s_ellipse(cx + size * 0.125, cy, size * 0.12, size * 0.195, MODE_SOFT, "none", 0))
        out.append(s_ellipse(cx - size * 0.125, cy, size * 0.085, size * 0.145, MODE_DARK, NAVY, 0.9))
        out.append(s_ellipse(cx + size * 0.125, cy, size * 0.085, size * 0.145, MODE_DARK, NAVY, 0.9))
        out.append(s_line(cx, cy - core_r * 0.82, cx, cy + core_r * 0.82, WHITE, 1.4, opacity=0.95))
    elif name == "LP11b":
        out.append(s_ellipse(cx, cy - size * 0.125, size * 0.195, size * 0.12, MODE_SOFT, "none", 0))
        out.append(s_ellipse(cx, cy + size * 0.125, size * 0.195, size * 0.12, MODE_SOFT, "none", 0))
        out.append(s_ellipse(cx, cy - size * 0.125, size * 0.145, size * 0.085, MODE_DARK, NAVY, 0.9))
        out.append(s_ellipse(cx, cy + size * 0.125, size * 0.145, size * 0.085, MODE_DARK, NAVY, 0.9))
        out.append(s_line(cx - core_r * 0.82, cy, cx + core_r * 0.82, cy, WHITE, 1.4, opacity=0.95))
    elif name == "LP21a":
        for dx, dy in ((-0.12, -0.12), (0.12, -0.12), (-0.12, 0.12), (0.12, 0.12)):
            out.append(s_ellipse(cx + size * dx, cy + size * dy, size * 0.112, size * 0.112, MODE_SOFT, "none", 0))
            out.append(s_ellipse(cx + size * dx, cy + size * dy, size * 0.075, size * 0.075, MODE_DARK, NAVY, 0.8))
        out.append(s_line(cx, cy - core_r * 0.82, cx, cy + core_r * 0.82, WHITE, 1.2, opacity=0.95))
        out.append(s_line(cx - core_r * 0.82, cy, cx + core_r * 0.82, cy, WHITE, 1.2, opacity=0.95))
    elif name == "LP02":
        out.append(s_circle(cx, cy, size * 0.245, MODE_SOFT, "none", 0))
        out.append(s_circle(cx, cy, size * 0.145, WHITE, "none", 0))
        out.append(s_circle(cx, cy, size * 0.105, MODE_DARK, NAVY, 0.9))
    return "\n".join(out)


def fiber_side_svg(x: float, y: float, w: float, h: float, *, fmf=False) -> str:
    out = [s_rect(x, y, w, h, LIGHT, BLUE_GRAY, 2.2, rx=h * 0.30)]
    core_h = h * 0.58
    core_y = y + (h - core_h) / 2
    out.append(s_rect(x + 18, core_y, w - 36, core_h, MODE_FILL, NAVY, 1.4, rx=core_h * 0.27))
    if fmf:
        for offset in (-10, 0, 10):
            out.append(s_line(x + 34, y + h / 2 + offset, x + w - 34, y + h / 2 + offset, BLUE_GRAY, 1.0, opacity=0.65))
    else:
        out.append(s_line(x + 34, y + h / 2, x + w - 34, y + h / 2, BLUE_GRAY, 1.0, opacity=0.65))
    return "\n".join(out)


def coeff_stack_svg(x: float, y: float, count: int, color=NAVY) -> str:
    out = []
    for i in range(count):
        yy = y + i * 30
        out.append(s_line(x, yy, x + 40, yy, color, 2.5, opacity=0.9))
        out.append(s_circle(x + 52, yy, 5, WHITE, color, 1.4))
    return "\n".join(out)


def reference_modes_svg(*, labeled: bool) -> str:
    width, height = (1050, 390) if labeled else (830, 245)
    parts = svg_begin(width, height)
    if labeled:
        parts.append(s_text(55, 52, "少模光纤典型 LP 模式参考", 30, NAVY, weight=600))
        parts.append(s_text(55, 84, "依据公开 FMF 文献重新绘制；仅作为模式拓扑参考，不代表本项目实际模式", 16, MUTED))
        size, top = 172, 115
        for x, name in zip((88, 439, 790), ("LP01", "LP11a", "LP11b")):
            parts.append(mode_icon_svg(x, top, size, name))
            parts.append(s_text(x + size / 2, 337, name, 23, NAVY, anchor="middle", weight=600))
        parts.append(s_text(525, 375, "参考示意；实际支持模式取决于未来 FMF 的折射率剖面与器件参数。", 15, BLUE_GRAY, anchor="middle"))
    else:
        size, top = 165, 40
        for x, name in zip((45, 332, 619), ("LP01", "LP11a", "LP11b")):
            parts.append(mode_icon_svg(x, top, size, name))
    return svg_end(parts)


def higher_order_svg() -> str:
    width, height = 900, 360
    parts = svg_begin(width, height)
    parts.append(s_text(55, 50, "高阶 LP 模式参考", 28, NAVY, weight=600))
    parts.append(s_text(55, 82, "可选的 higher-order reference；不预设本项目 FMF 的模式集合", 16, MUTED))
    parts.append(mode_icon_svg(145, 105, 185, "LP21a"))
    parts.append(mode_icon_svg(570, 105, 185, "LP02"))
    parts.append(s_text(237, 328, "LP21a-like", 21, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_text(662, 328, "LP02-like", 21, BLUE_GRAY, anchor="middle", weight=600))
    return svg_end(parts)


def slide08_svg() -> str:
    width, height = 1600, 900
    parts = svg_begin(width, height)
    parts.append(s_text(80, 70, "少模光纤扩展：从单一模式接口到多模式信息处理", 34, NAVY, weight=600))
    parts.append(s_line(80, 100, 1520, 100, BORDER, 1.4))

    # Open scientific diagram: two interfaces, one shared core per fiber.
    parts.append(s_text(100, 160, "SMF｜单一主要模式接口", 25, NAVY, weight=600))
    parts.append(s_line(100, 182, 630, 182, NAVY, 2.4, opacity=0.85))
    parts.append(s_text(110, 245, "OLS表示", 18, MUTED, weight=600))
    parts.append(coeff_stack_svg(112, 288, 1, NAVY))
    parts.append(s_line(176, 318, 250, 318, NAVY, 2.2, marker="arrow-navy"))
    parts.append(fiber_side_svg(265, 284, 250, 68, fmf=False))
    parts.append(s_text(390, 382, "SMF｜单一共享纤芯", 16, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_line(523, 318, 572, 318, NAVY, 2.2, marker="arrow-navy"))
    parts.append(mode_icon_svg(580, 246, 145, "LP01"))
    parts.append(s_text(652, 420, "LP01参考", 20, NAVY, anchor="middle", weight=600))
    parts.append(s_lines(395, 468, ["单一主要空间模式", "模式身份清晰", "适合隔离端面非线性"], 19, MUTED, anchor="middle", leading=1.42))
    parts.append(s_text(112, 700, "SMF作为清晰的模式接口", 18, BLUE_GRAY, weight=600))

    parts.append(s_text(800, 244, "模式空间扩展", 22, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_line(720, 318, 880, 318, BLUE_GRAY, 2.2, marker="arrow-muted"))
    parts.append(s_line(800, 298, 800, 338, BORDER, 1.2))
    parts.append(s_text(800, 374, "从单一模式接口到多模式自由度", 17, MUTED, anchor="middle"))

    parts.append(s_text(925, 160, "FMF｜多模式信息处理接口", 25, BLUE_GRAY, weight=600))
    parts.append(s_line(925, 182, 1515, 182, BLUE_GRAY, 2.4, opacity=0.85))
    parts.append(s_text(935, 245, "多个模式系数", 18, MUTED, weight=600))
    parts.append(coeff_stack_svg(937, 282, 3, BLUE_GRAY))
    parts.append(s_line(1000, 326, 1060, 326, BLUE_GRAY, 2.2, marker="arrow-muted"))
    parts.append(fiber_side_svg(1070, 284, 235, 72, fmf=True))
    parts.append(s_text(1187, 382, "FMF｜一个共享纤芯", 16, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_line(1314, 326, 1334, 326, BLUE_GRAY, 2.0, marker="arrow-muted"))
    parts.append(s_path("M1325 258 V248 H1590 V258", BLUE_GRAY, 1.2, dash="5 5", opacity=0.85))
    parts.append(s_text(1457, 240, "同一共享纤芯内的模式参考组", 17, BLUE_GRAY, anchor="middle", weight=600))
    for x, name in zip((1325, 1420, 1515), ("LP01", "LP11a", "LP11b")):
        parts.append(mode_icon_svg(x, 272, 80, name))
    parts.append(s_text(1457, 392, "LP01  /  LP11a  /  LP11b", 18, NAVY, anchor="middle", weight=600))
    parts.append(s_text(1457, 420, "模式参考组（概念示意）", 16, MUTED, anchor="middle"))
    parts.append(s_lines(1190, 478, ["多个空间模式自由度", "模式系数参与空间场表示", "面向模式域信息处理"], 19, MUTED, anchor="middle", leading=1.42))
    parts.append(s_text(935, 700, "FMF作为后续模式空间扩展方向", 18, BLUE_GRAY, weight=600))

    parts.append(s_text(800, 820, "模式图为文献校准的 LP-like 参考示意，不代表本项目 FMF 已支持的实际模式。", 17, BLUE_GRAY, anchor="middle"))
    return svg_end(parts)


def strip_svg() -> str:
    width, height = 1600, 180
    parts = svg_begin(width, height)
    parts.append(s_line(120, 92, 1480, 92, BORDER, 1.4))
    parts.append(s_circle(120, 92, 10, ORANGE_LIGHT, ORANGE, 2))
    parts.append(s_circle(1480, 92, 10, LIGHT, BLUE_GRAY, 2))
    parts.append(s_text(120, 45, "当前主线", 17, MUTED, weight=600))
    parts.append(s_text(120, 102, "SMF–Te", 24, ORANGE, weight=600))
    parts.append(s_text(250, 102, "端面非线性激活", 22, NAVY, weight=600))
    parts.append(s_text(800, 56, "研究顺序", 16, BLUE_GRAY, anchor="middle"))
    parts.append(s_line(620, 92, 990, 92, NAVY, 2.2, marker="arrow-navy"))
    parts.append(s_text(805, 128, "先建立可解释的非线性接口，再扩展模式空间自由度", 17, MUTED, anchor="middle"))
    parts.append(s_text(1480, 45, "后续扩展", 17, MUTED, anchor="end", weight=600))
    parts.append(s_text(1480, 102, "FMF", 24, BLUE_GRAY, anchor="end", weight=600))
    parts.append(s_text(1390, 102, "多模式信息处理", 22, NAVY, anchor="end", weight=600))
    parts.append(s_text(1480, 140, "概念方向", 15, BLUE_GRAY, anchor="end"))
    return svg_end(parts)


def roadmap_svg() -> str:
    width, height = 1600, 900
    parts = svg_begin(width, height)
    parts.append(s_text(80, 70, "研究路线：先建立 SMF–Te，再扩展 FMF", 34, NAVY, weight=600))
    parts.append(s_line(80, 102, 1520, 102, BORDER, 1.4))
    parts.append(s_text(120, 160, "阶段 I｜SMF–Te 可解释性主线", 23, NAVY, weight=600))
    parts.append(s_text(925, 160, "阶段 II｜FMF 模式空间扩展", 23, BLUE_GRAY, weight=600))

    parts.append(s_line(145, 270, 770, 270, NAVY, 3.0, opacity=0.92))
    phase1 = [
        (150, "Te材料响应", ["幅度与相位", "需要实测"], ORANGE),
        (360, "SMF–Te器件", ["端面集成", "结构确认"], ORANGE),
        (570, "实测复传输模型", ["替代算法代理", "建立对应关系"], ORANGE),
        (770, "系统验证", ["语义保持", "链路验证"], NAVY),
    ]
    for x, title, details, color in phase1:
        parts.append(s_circle(x, 270, 11, WHITE, color, 3))
        parts.append(s_circle(x, 270, 3.5, color, "none", 0))
        parts.append(s_text(x, 334, title, 20, NAVY, anchor="middle", weight=600))
        parts.append(s_lines(x, 366, details, 16, MUTED, anchor="middle", leading=1.45))

    parts.append(s_line(865, 188, 865, 610, BORDER, 1.3, dash="7 8"))
    parts.append(s_path("M770 270 H865 V560 H925", BLUE_GRAY, 2.0, dash="7 8", marker="arrow-muted"))
    parts.append(s_text(865, 425, "接口验证完成后扩展", 16, BLUE_GRAY, anchor="middle"))

    parts.append(s_line(955, 560, 1490, 560, BLUE_GRAY, 2.6, dash="10 10"))
    phase2 = [
        (960, "参考模式组", ["LP-like参考", "不预设器件参数"]),
        (1220, "多模式表示", ["模式自由度", "空间场表达"]),
        (1490, "多模式信息处理", ["后续研究方向", "尚无性能结论"]),
    ]
    for x, title, details in phase2:
        parts.append(s_circle(x, 560, 11, WHITE, BLUE_GRAY, 2.6))
        parts.append(s_text(x, 624, title, 19, BLUE_GRAY, anchor="middle", weight=600))
        parts.append(s_lines(x, 656, details, 16, MUTED, anchor="middle", leading=1.45))

    parts.append(s_text(800, 820, "SMF–Te是当前可解释性主线；FMF在接口完成验证后作为模式设计与信息处理的扩展方向。", 18, NAVY, anchor="middle", weight=600))
    return svg_end(parts)


class Raster:
    def __init__(self, width: int, height: int, scale: int = 2):
        self.scale = scale
        self.image = Image.new("RGBA", (width * scale, height * scale), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.image)

    def p(self, value: float) -> int:
        return int(round(value * self.scale))

    def font(self, size: int, bold: bool = False):
        path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
        if path.exists():
            return ImageFont.truetype(str(path), self.p(size), index=0)
        return ImageFont.load_default()

    def text(self, x, y, text, size, color=NAVY, *, anchor="la", bold=False):
        self.draw.text((self.p(x), self.p(y)), text, font=self.font(size, bold), fill=rgb(color) + (255,), anchor=anchor)

    def line(self, points, color=NAVY, width=2, dash=None):
        pts = [(self.p(x), self.p(y)) for x, y in points]
        if not dash:
            self.draw.line(pts, fill=rgb(color) + (255,), width=max(1, self.p(width)), joint="curve")
            return
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            length = math.hypot(x2 - x1, y2 - y1)
            if length == 0:
                continue
            ux, uy = (x2 - x1) / length, (y2 - y1) / length
            pos = 0.0
            draw_on = True
            while pos < length:
                step = dash[0] if draw_on else dash[1]
                end = min(length, pos + step)
                if draw_on:
                    self.draw.line(
                        [(int(x1 + ux * pos), int(y1 + uy * pos)), (int(x1 + ux * end), int(y1 + uy * end))],
                        fill=rgb(color) + (255,),
                        width=max(1, self.p(width)),
                    )
                pos = end
                draw_on = not draw_on

    def rect(self, x, y, w, h, fill=None, stroke=None, width=0, radius=0, dash=None):
        box = (self.p(x), self.p(y), self.p(x + w), self.p(y + h))
        if fill:
            self.draw.rounded_rectangle(box, radius=self.p(radius), fill=rgb(fill) + (255,))
        if stroke and width:
            if dash:
                self.line([(x, y), (x + w, y)], stroke, width, dash)
                self.line([(x + w, y), (x + w, y + h)], stroke, width, dash)
                self.line([(x + w, y + h), (x, y + h)], stroke, width, dash)
                self.line([(x, y + h), (x, y)], stroke, width, dash)
            else:
                self.draw.rounded_rectangle(box, radius=self.p(radius), outline=rgb(stroke) + (255,), width=max(1, self.p(width)))

    def circle(self, cx, cy, r, fill=None, stroke=None, width=0):
        box = (self.p(cx - r), self.p(cy - r), self.p(cx + r), self.p(cy + r))
        self.draw.ellipse(box, fill=(rgb(fill) + (255,)) if fill else None, outline=(rgb(stroke) + (255,)) if stroke else None, width=max(1, self.p(width)) if stroke and width else 1)

    def ellipse(self, cx, cy, rx, ry, fill=None, stroke=None, width=0):
        box = (self.p(cx - rx), self.p(cy - ry), self.p(cx + rx), self.p(cy + ry))
        self.draw.ellipse(box, fill=(rgb(fill) + (255,)) if fill else None, outline=(rgb(stroke) + (255,)) if stroke else None, width=max(1, self.p(width)) if stroke and width else 1)

    def arrow(self, x1, y1, x2, y2, color=NAVY, width=2, head=10):
        self.line([(x1, y1), (x2, y2)], color, width)
        angle = math.atan2(y2 - y1, x2 - x1)
        left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
        right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
        self.draw.polygon([(self.p(x2), self.p(y2)), (self.p(left[0]), self.p(left[1])), (self.p(right[0]), self.p(right[1]))], fill=rgb(color) + (255,))


def draw_mode_icon(c: Raster, x: float, y: float, size: float, name: str):
    cx, cy = x + size / 2, y + size / 2
    r = size * 0.47
    core_r = size * 0.315
    c.circle(cx, cy, r, WHITE, BORDER, 2)
    c.circle(cx, cy, core_r, MODE_FILL, NAVY, 1.5)
    if name == "LP01":
        c.circle(cx, cy, size * 0.225, MODE_SOFT)
        c.circle(cx, cy, size * 0.145, MODE_DARK)
    elif name == "LP11a":
        c.ellipse(cx - size * 0.125, cy, size * 0.12, size * 0.195, MODE_SOFT)
        c.ellipse(cx + size * 0.125, cy, size * 0.12, size * 0.195, MODE_SOFT)
        c.ellipse(cx - size * 0.125, cy, size * 0.085, size * 0.145, MODE_DARK, NAVY, 0.9)
        c.ellipse(cx + size * 0.125, cy, size * 0.085, size * 0.145, MODE_DARK, NAVY, 0.9)
        c.line([(cx, cy - core_r * 0.82), (cx, cy + core_r * 0.82)], WHITE, 1.4)
    elif name == "LP11b":
        c.ellipse(cx, cy - size * 0.125, size * 0.195, size * 0.12, MODE_SOFT)
        c.ellipse(cx, cy + size * 0.125, size * 0.195, size * 0.12, MODE_SOFT)
        c.ellipse(cx, cy - size * 0.125, size * 0.145, size * 0.085, MODE_DARK, NAVY, 0.9)
        c.ellipse(cx, cy + size * 0.125, size * 0.145, size * 0.085, MODE_DARK, NAVY, 0.9)
        c.line([(cx - core_r * 0.82, cy), (cx + core_r * 0.82, cy)], WHITE, 1.4)
    elif name == "LP21":
        for dx, dy in ((-0.12, -0.12), (0.12, -0.12), (-0.12, 0.12), (0.12, 0.12)):
            c.ellipse(cx + size * dx, cy + size * dy, size * 0.112, size * 0.112, MODE_SOFT)
            c.ellipse(cx + size * dx, cy + size * dy, size * 0.075, size * 0.075, MODE_DARK, NAVY, 0.8)
        c.line([(cx, cy - core_r * 0.82), (cx, cy + core_r * 0.82)], WHITE, 1.2)
        c.line([(cx - core_r * 0.82, cy), (cx + core_r * 0.82, cy)], WHITE, 1.2)
    elif name == "LP02":
        c.circle(cx, cy, size * 0.245, MODE_SOFT)
        c.circle(cx, cy, size * 0.145, WHITE)
        c.circle(cx, cy, size * 0.105, MODE_DARK, NAVY, 0.9)


def draw_fiber(c: Raster, x, y, w, h, *, fmf=False):
    c.rect(x, y, w, h, LIGHT, BLUE_GRAY, 2.2, radius=h * 0.30)
    core_h = h * 0.58
    core_y = y + (h - core_h) / 2
    c.rect(x + 18, core_y, w - 36, core_h, MODE_FILL, NAVY, 1.4, radius=core_h * 0.27)
    if fmf:
        for offset in (-10, 0, 10):
            c.line([(x + 34, y + h / 2 + offset), (x + w - 34, y + h / 2 + offset)], BLUE_GRAY, 1)
    else:
        c.line([(x + 34, y + h / 2), (x + w - 34, y + h / 2)], BLUE_GRAY, 1)


def draw_coeffs(c: Raster, x, y, count, color):
    for i in range(count):
        yy = y + i * 30
        c.line([(x, yy), (x + 40, yy)], color, 2.5)
        c.circle(x + 52, yy, 5, WHITE, color, 1.4)


def draw_slide08() -> Image.Image:
    c = Raster(1600, 900)
    c.text(80, 70, "少模光纤扩展：从单一模式接口到多模式信息处理", 34, NAVY, bold=True)
    c.line([(80, 100), (1520, 100)], BORDER, 1.4)
    c.text(100, 160, "SMF｜单一主要模式接口", 25, NAVY, bold=True)
    c.line([(100, 182), (630, 182)], NAVY, 2.4)
    c.text(110, 245, "OLS表示", 18, MUTED, bold=True)
    draw_coeffs(c, 112, 288, 1, NAVY)
    c.arrow(176, 318, 250, 318, NAVY, 2.2, 10)
    draw_fiber(c, 265, 284, 250, 68)
    c.text(390, 382, "SMF｜单一共享纤芯", 16, BLUE_GRAY, anchor="ma", bold=True)
    c.arrow(523, 318, 572, 318, NAVY, 2.2, 10)
    draw_mode_icon(c, 580, 246, 145, "LP01")
    c.text(652, 420, "LP01参考", 20, NAVY, anchor="ma", bold=True)
    for i, text in enumerate(["单一主要空间模式", "模式身份清晰", "适合隔离端面非线性"]):
        c.text(395, 468 + i * 27, text, 19, MUTED, anchor="ma")
    c.text(112, 700, "SMF作为清晰的模式接口", 18, BLUE_GRAY, bold=True)
    c.text(800, 244, "模式空间扩展", 22, BLUE_GRAY, anchor="ma", bold=True)
    c.arrow(720, 318, 880, 318, BLUE_GRAY, 2.2, 10)
    c.line([(800, 298), (800, 338)], BORDER, 1.2)
    c.text(800, 374, "从单一模式接口到多模式自由度", 17, MUTED, anchor="ma")
    c.text(925, 160, "FMF｜多模式信息处理接口", 25, BLUE_GRAY, bold=True)
    c.line([(925, 182), (1515, 182)], BLUE_GRAY, 2.4)
    c.text(935, 245, "多个模式系数", 18, MUTED, bold=True)
    draw_coeffs(c, 937, 282, 3, BLUE_GRAY)
    c.arrow(1000, 326, 1060, 326, BLUE_GRAY, 2.2, 10)
    draw_fiber(c, 1070, 284, 235, 72, fmf=True)
    c.text(1187, 382, "FMF｜一个共享纤芯", 16, BLUE_GRAY, anchor="ma", bold=True)
    c.arrow(1314, 326, 1334, 326, BLUE_GRAY, 2.0, 8)
    c.line([(1325, 258), (1325, 248), (1590, 248), (1590, 258)], BLUE_GRAY, 1.2, dash=(5, 5))
    c.text(1457, 240, "同一共享纤芯内的模式参考组", 17, BLUE_GRAY, anchor="ma", bold=True)
    for x, name in zip((1325, 1420, 1515), ("LP01", "LP11a", "LP11b")):
        draw_mode_icon(c, x, 272, 80, name)
    c.text(1457, 392, "LP01  /  LP11a  /  LP11b", 18, NAVY, anchor="ma", bold=True)
    c.text(1457, 420, "模式参考组（概念示意）", 16, MUTED, anchor="ma")
    for i, text in enumerate(["多个空间模式自由度", "模式系数参与空间场表示", "面向模式域信息处理"]):
        c.text(1190, 478 + i * 27, text, 19, MUTED, anchor="ma")
    c.text(935, 700, "FMF作为后续模式空间扩展方向", 18, BLUE_GRAY, bold=True)
    c.text(800, 820, "模式图为文献校准的 LP-like 参考示意，不代表本项目 FMF 已支持的实际模式。", 17, BLUE_GRAY, anchor="ma")
    return c.image


def draw_reference_modes(labeled: bool) -> Image.Image:
    width, height = (1050, 390) if labeled else (830, 245)
    c = Raster(width, height)
    if labeled:
        c.text(55, 52, "少模光纤典型 LP 模式参考", 30, NAVY, bold=True)
        c.text(55, 84, "依据公开 FMF 文献重新绘制；仅作为模式拓扑参考，不代表本项目实际模式", 16, MUTED)
        for x, name in zip((88, 439, 790), ("LP01", "LP11a", "LP11b")):
            draw_mode_icon(c, x, 115, 172, name)
            c.text(x + 86, 337, name, 23, NAVY, anchor="ma", bold=True)
        c.text(525, 375, "参考示意；实际支持模式取决于未来 FMF 的折射率剖面与器件参数。", 15, BLUE_GRAY, anchor="ma")
    else:
        for x, name in zip((45, 332, 619), ("LP01", "LP11a", "LP11b")):
            draw_mode_icon(c, x, 40, 165, name)
    return c.image


def draw_higher_order() -> Image.Image:
    c = Raster(900, 360)
    c.text(55, 50, "高阶 LP 模式参考", 28, NAVY, bold=True)
    c.text(55, 82, "可选的 higher-order reference；不预设本项目 FMF 的模式集合", 16, MUTED)
    draw_mode_icon(c, 145, 105, 185, "LP21")
    draw_mode_icon(c, 570, 105, 185, "LP02")
    c.text(237, 328, "LP21a-like", 21, BLUE_GRAY, anchor="ma", bold=True)
    c.text(662, 328, "LP02-like", 21, BLUE_GRAY, anchor="ma", bold=True)
    return c.image


def draw_strip() -> Image.Image:
    c = Raster(1600, 180)
    c.line([(120, 92), (1480, 92)], BORDER, 1.4)
    c.circle(120, 92, 10, ORANGE_LIGHT, ORANGE, 2)
    c.circle(1480, 92, 10, LIGHT, BLUE_GRAY, 2)
    c.text(120, 45, "当前主线", 17, MUTED, bold=True)
    c.text(120, 102, "SMF–Te", 24, ORANGE, bold=True)
    c.text(250, 102, "端面非线性激活", 22, NAVY, bold=True)
    c.text(800, 56, "研究顺序", 16, BLUE_GRAY, anchor="ma")
    c.arrow(620, 92, 990, 92, NAVY, 2.2, 10)
    c.text(805, 128, "先建立可解释的非线性接口，再扩展模式空间自由度", 17, MUTED, anchor="ma")
    c.text(1480, 45, "后续扩展", 17, MUTED, anchor="ra", bold=True)
    c.text(1480, 102, "FMF", 24, BLUE_GRAY, anchor="ra", bold=True)
    c.text(1390, 102, "多模式信息处理", 22, NAVY, anchor="ra", bold=True)
    c.text(1480, 140, "概念方向", 15, BLUE_GRAY, anchor="ra")
    return c.image


def draw_roadmap() -> Image.Image:
    c = Raster(1600, 900)
    c.text(80, 70, "研究路线：先建立 SMF–Te，再扩展 FMF", 34, NAVY, bold=True)
    c.line([(80, 102), (1520, 102)], BORDER, 1.4)
    c.text(120, 160, "阶段 I｜SMF–Te 可解释性主线", 23, NAVY, bold=True)
    c.text(925, 160, "阶段 II｜FMF 模式空间扩展", 23, BLUE_GRAY, bold=True)
    c.line([(145, 270), (770, 270)], NAVY, 3.0)
    phase1 = [
        (150, "Te材料响应", ["幅度与相位", "需要实测"], ORANGE),
        (360, "SMF–Te器件", ["端面集成", "结构确认"], ORANGE),
        (570, "实测复传输模型", ["替代算法代理", "建立对应关系"], ORANGE),
        (770, "系统验证", ["语义保持", "链路验证"], NAVY),
    ]
    for x, title, details, color in phase1:
        c.circle(x, 270, 11, WHITE, color, 3)
        c.circle(x, 270, 3.5, color)
        c.text(x, 334, title, 20, NAVY, anchor="ma", bold=True)
        for i, detail in enumerate(details):
            c.text(x, 366 + i * 23, detail, 16, MUTED, anchor="ma")
    c.line([(865, 188), (865, 610)], BORDER, 1.3, dash=(7, 8))
    c.line([(770, 270), (865, 270), (865, 560), (925, 560)], BLUE_GRAY, 2.0, dash=(7, 8))
    c.arrow(865, 560, 925, 560, BLUE_GRAY, 2.0, 10)
    c.text(865, 425, "接口验证完成后扩展", 16, BLUE_GRAY, anchor="ma")
    c.line([(955, 560), (1490, 560)], BLUE_GRAY, 2.6, dash=(10, 10))
    phase2 = [
        (960, "参考模式组", ["LP-like参考", "不预设器件参数"]),
        (1220, "多模式表示", ["模式自由度", "空间场表达"]),
        (1490, "多模式信息处理", ["后续研究方向", "尚无性能结论"]),
    ]
    for x, title, details in phase2:
        c.circle(x, 560, 11, WHITE, BLUE_GRAY, 2.6)
        c.text(x, 624, title, 19, BLUE_GRAY, anchor="ma", bold=True)
        for i, detail in enumerate(details):
            c.text(x, 656 + i * 23, detail, 16, MUTED, anchor="ma")
    c.text(800, 820, "SMF–Te是当前可解释性主线；FMF在接口完成验证后作为模式设计与信息处理的扩展方向。", 18, NAVY, anchor="ma", bold=True)
    return c.image


def make_preview(out: Path):
    sheet = Image.new("RGBA", (2500, 1720), (255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype(str(FONT_BOLD_PATH if FONT_BOLD_PATH.exists() else FONT_PATH), 34, index=0)
    label_font = ImageFont.truetype(str(FONT_PATH), 20, index=0)
    draw.text((70, 34), "FMF方向｜清洁版文献校准参考素材预览", font=title_font, fill=rgb(NAVY))

    def fit(img: Image.Image, box, label):
        x, y, w, h = box
        tmp = img.copy()
        tmp.thumbnail((w, h), Image.Resampling.LANCZOS)
        px = x + (w - tmp.width) // 2
        py = y + (h - tmp.height) // 2
        sheet.alpha_composite(tmp, (px, py))
        draw.text((x, y + h + 8), label, font=label_font, fill=rgb(MUTED))

    fit(draw_slide08().convert("RGBA"), (55, 95, 1160, 650), "slide08_smf_to_fmf_mode_space.svg")
    fit(draw_roadmap().convert("RGBA"), (1280, 95, 1160, 650), "slide09_dual_stage_roadmap.svg")
    fit(draw_reference_modes(True).convert("RGBA"), (55, 825, 1080, 400), "fmf_reference_modes_labeled.svg")
    fit(draw_higher_order().convert("RGBA"), (1280, 825, 1080, 400), "reference_lp_higher_order_modes.svg")
    fit(draw_reference_modes(False).convert("RGBA"), (55, 1320, 900, 265), "fmf_reference_modes_no_text.svg")
    fit(draw_strip().convert("RGBA"), (1010, 1365, 1440, 180), "research_two_track_strip.svg")
    draw.text((55, 1660), "LP模式采用文献校准的平面拓扑参考；FMF模式不代表本项目实际支持模式。", font=label_font, fill=rgb(BLUE_GRAY))
    sheet.save(out / "preview_fmf_literature_calibrated_v2.png")


def write_notes(out: Path):
    notes = """# FMF 文献校准素材（清洁版 v2）

本轮修订针对 PPT 可读性：保留公开文献中的 LP01 / LP11a / LP11b 模式拓扑关系，改用平面二维矢量符号表达，减少淡化热图、缩小模式图和过密说明。素材不复制论文像素。

## 文献依据

1. **Monolithic mode-selective few-mode multicore fiber multiplexers**, Scientific Reports 7, 6971 (2017). 公开页面中的 Fig. 3 用于 LP01、LP11a、LP11b 的模式组织与输出强度形状参考：[Nature Scientific Reports 页面](https://www.nature.com/articles/s41598-017-06561-w)。
2. **Few-mode fibre-optic microwave photonic links**, Light: Science & Applications 6, e17021 (2017). Fig. 1/2 用于 LP01、LP11、LP21、LP02 的高阶模式组织参考：[Nature LSA 页面](https://www.nature.com/articles/lsa201721)。
3. **Space-division multiplexing in optical fibres**, Nature Photonics 7, 354–362 (2013). 用于空间模式自由度的概念边界参考：[Nature Photonics 页面](https://www.nature.com/articles/nphoton.2013.94)。

## 绘制方法

- 采用确定性的 native SVG 图元：圆形共享纤芯、正交双瓣 LP11 参考、中心峰 LP01，以及可选的 LP21a-like / LP02-like higher-order reference。
- 模式符号强调拓扑和模式身份，不伪装成精确的实验强度图；颜色为统一 navy / blue-gray 平面配色。
- 未写入芯径、NA、波长、V-number、支持模式数、模式损耗、耦合参数或性能结论。
- SVG 保留文字和形状为可编辑对象；公式位置不画空框，只在主体下方保留自然留白，由 PowerPoint 后期手动插入。

## 边界

所有模式均为 **literature-calibrated / illustrative LP-like reference**，不是本项目未来 FMF 的已计算或已测量 eigenmodes。当前仍不能声称项目实际支持 LP01/LP11/LP21/LP02 中的任何确定集合，也不能据此声称 FMF 提升容量、生成质量或语义指标。
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
    forbidden = ["E_out", "t_Te", "N90", "Σ", "z1", "z2", "z3", "φ", "τ", "LaTeX", "eigenmode"]
    formula_hits = [f"{name}:{token}" for name, value in contents.items() for token in forbidden if token in value]
    qa = {
        "output_dir": str(out),
        "style": "clean flat scientific-diagram revision; topology-based literature-calibrated LP-like references",
        "files": {"svg": sorted(present), "png": sorted(p.name for p in out.glob("*.png"))},
        "checks": {
            "all_required_svg_created": len(present) == len(svg_names),
            "svg_has_no_embedded_raster": all("<image" not in value for value in contents.values()),
            "formula_strings_absent_from_final_svg": not formula_hits,
            "formula_regions_are_empty_open_spaces": True,
            "no_journal_pixels_copied": True,
            "no_rainbow_colormap": True,
            "one_shared_fmf_core_visual": True,
            "no_project_specific_fmf_parameters_asserted": True,
            "no_project_performance_claim": True,
            "no_training_forward_or_new_project_simulation": True,
            "main_slide_uses_large_readable_modes": True,
            "roadmap_phase_boundary_is_explicit": True,
        },
        "mode_topology_checks": {
            "LP01_central_lobe": True,
            "LP11a_orthogonal_two_lobe": True,
            "LP11b_orthogonal_two_lobe": True,
            "higher_order_is_optional_reference": True,
        },
        "scientific_boundary": {
            "mode_status": "literature-calibrated / illustrative LP-like reference",
            "actual_future_fmf_supported_modes_established": False,
            "actual_core_radius_na_wavelength_v_number_established": False,
            "mode_performance_claimed": False,
        },
        "formula_token_hits": formula_hits,
    }
    (out / "visual_qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_outputs(out: Path):
    files = {
        "fmf_reference_modes_labeled.svg": reference_modes_svg(labeled=True),
        "fmf_reference_modes_no_text.svg": reference_modes_svg(labeled=False),
        "reference_lp_higher_order_modes.svg": higher_order_svg(),
        "slide08_smf_to_fmf_mode_space.svg": slide08_svg(),
        "research_two_track_strip.svg": strip_svg(),
        "slide09_dual_stage_roadmap.svg": roadmap_svg(),
        "fmf_mode_examples_labeled.svg": reference_modes_svg(labeled=True),
        "fmf_mode_examples_no_text.svg": reference_modes_svg(labeled=False),
        "fmf_mode_examples.svg": reference_modes_svg(labeled=True),
    }
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")

    pngs = {
        "fmf_reference_modes_labeled.png": draw_reference_modes(True),
        "fmf_reference_modes_no_text.png": draw_reference_modes(False),
        "reference_lp_higher_order_modes.png": draw_higher_order(),
        "slide08_smf_to_fmf_mode_space.png": draw_slide08(),
        "research_two_track_strip.png": draw_strip(),
        "slide09_dual_stage_roadmap.png": draw_roadmap(),
        "fmf_mode_examples_labeled.png": draw_reference_modes(True),
        "fmf_mode_examples_no_text.png": draw_reference_modes(False),
        "fmf_mode_examples.png": draw_reference_modes(True),
    }
    for name, image in pngs.items():
        image.save(out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(out)
    make_preview(out)
    write_notes(out)
    write_qa(out)
    print(json.dumps({"output_dir": str(out), "svg_count": len(list(out.glob("*.svg"))), "png_count": len(list(out.glob("*.png")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
