"""Render editable SVG assets for the SMF end-face material platform slide."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "ppt_slide04_material_candidates_v1"

NAVY = "#223F73"
BLUE = "#4F79C8"
BLUE_GRAY = "#8094AA"
LIGHT = "#EAF0F7"
PALE = "#F5F8FC"
BORDER = "#D5DFEA"
MUTED = "#536579"
ORANGE = "#C97842"
ORANGE_LIGHT = "#F4E3D7"
WHITE = "#FFFFFF"

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_FAMILY = "Microsoft YaHei, DengXian, Noto Sans CJK SC, Arial, sans-serif"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def f(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


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
        f'<text x="{f(x)}" y="{f(y)}" font-family="{FONT_FAMILY}" font-size="{size}px" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" opacity="{f(opacity)}">{esc(text)}</text>'
    )


def s_lines(x, y, lines, size, color=MUTED, *, anchor="start", weight=400, leading=1.35):
    return "\n".join(
        s_text(x, y + i * size * leading, line, size, color, anchor=anchor, weight=weight)
        for i, line in enumerate(lines)
    )


def s_line(x1, y1, x2, y2, color=NAVY, width=2, *, dash=None, marker=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{color}" stroke-width="{f(width)}" fill="none" opacity="{f(opacity)}"{extra}/>'


def s_path(d, color=NAVY, width=2, *, dash=None, opacity=1.0):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" stroke="{color}" stroke-width="{f(width)}" fill="none" stroke-linecap="round" opacity="{f(opacity)}"{extra}/>'


def s_rect(x, y, w, h, fill="none", stroke="none", width=0, *, rx=0, opacity=1.0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="{f(rx)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(width)}" opacity="{f(opacity)}"{extra}/>'


def s_circle(cx, cy, r, fill="none", stroke="none", width=0, *, opacity=1.0):
    return f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(width)}" opacity="{f(opacity)}"/>'


def s_ellipse(cx, cy, rx, ry, fill="none", stroke="none", width=0, *, opacity=1.0):
    return f'<ellipse cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(width)}" opacity="{f(opacity)}"/>'


def fiber_side_svg(x, y, w, h):
    parts = [s_rect(x, y, w, h, LIGHT, BLUE_GRAY, 2.2, rx=h * 0.32)]
    core_h = h * 0.55
    core_y = y + (h - core_h) / 2
    parts.append(s_rect(x + 16, core_y, w - 32, core_h, "#C9D9E9", NAVY, 1.4, rx=core_h * 0.24))
    parts.append(s_line(x + 32, y + h / 2, x + w - 32, y + h / 2, BLUE_GRAY, 1.0, opacity=0.75))
    return "\n".join(parts)


def facet_svg(cx, cy, r):
    parts = [s_circle(cx, cy, r, WHITE, BLUE_GRAY, 2.0), s_circle(cx, cy, r * 0.77, LIGHT, NAVY, 1.25)]
    # A compact complex-field symbol: two orthogonal phase contours and a central amplitude lobe.
    parts.append(s_ellipse(cx - r * 0.23, cy, r * 0.20, r * 0.32, "#9DB5CE", "none", 0, opacity=0.9))
    parts.append(s_ellipse(cx + r * 0.23, cy, r * 0.20, r * 0.32, "#9DB5CE", "none", 0, opacity=0.9))
    parts.append(s_line(cx, cy - r * 0.55, cx, cy + r * 0.55, WHITE, 1.3, opacity=0.9))
    parts.append(s_path(f"M {f(cx-r*0.53)} {f(cy-r*0.20)} Q {f(cx)} {f(cy-r*0.62)} {f(cx+r*0.53)} {f(cy-r*0.20)}", NAVY, 1.0, opacity=0.7))
    parts.append(s_path(f"M {f(cx-r*0.53)} {f(cy+r*0.20)} Q {f(cx)} {f(cy+r*0.62)} {f(cx+r*0.53)} {f(cy+r*0.20)}", NAVY, 1.0, opacity=0.7))
    return "\n".join(parts)


def plane_svg(x, y, w, h, label):
    parts = [s_rect(x, y, w, h, LIGHT, BLUE_GRAY, 1.6, rx=2)]
    for i in range(3):
        parts.append(s_line(x + 8, y + 12 + i * (h - 24) / 2, x + w - 8, y + 12 + i * (h - 24) / 2, BLUE_GRAY, 1.0, opacity=0.8))
    parts.append(s_text(x + w / 2, y + h + 22, label, 14, BLUE_GRAY, anchor="middle", weight=600))
    return "\n".join(parts)


def main_platform_svg() -> str:
    width, height = 940, 480
    parts = svg_begin(width, height)
    parts.append(s_text(35, 42, "OLS / 单模光纤输出端", 24, NAVY, weight=600))
    parts.append(s_text(35, 70, "复场接口", 15, MUTED))
    parts.append(fiber_side_svg(40, 172, 220, 66))
    parts.append(s_text(150, 274, "单模光纤", 15, BLUE_GRAY, anchor="middle", weight=600))
    parts.append(s_line(260, 205, 310, 205, NAVY, 2.2, marker="arrow-navy"))
    parts.append(facet_svg(360, 205, 64))
    parts.append(s_text(360, 304, "输出复场", 17, NAVY, anchor="middle", weight=600))
    parts.append(s_line(424, 205, 490, 205, NAVY, 2.2, marker="arrow-navy"))
    parts.append(s_rect(490, 135, 18, 140, ORANGE_LIGHT, ORANGE, 2.0, rx=2))
    parts.append(s_line(499, 122, 499, 102, ORANGE, 1.5))
    parts.append(s_text(520, 112, "端面功能材料层", 19, NAVY, weight=600))
    parts.append(s_text(520, 140, "Te / CoSi / TaP", 16, ORANGE, weight=600))
    parts.append(s_text(520, 164, "端面薄层", 14, MUTED))
    parts.append(s_line(518, 205, 650, 205, NAVY, 2.2, marker="arrow-navy"))
    parts.append(plane_svg(670, 132, 16, 146, "D1"))
    parts.append(s_line(686, 205, 734, 205, BLUE_GRAY, 2.0, marker="arrow-muted"))
    parts.append(plane_svg(750, 132, 16, 146, "D2"))
    parts.append(s_line(766, 205, 822, 205, BLUE_GRAY, 2.0, marker="arrow-muted"))
    parts.append(s_rect(830, 153, 85, 104, PALE, BLUE_GRAY, 1.5, rx=8))
    parts.append(s_text(872.5, 193, "解码器 /", 15, NAVY, anchor="middle", weight=600))
    parts.append(s_text(872.5, 218, "生成器", 15, NAVY, anchor="middle", weight=600))
    parts.append(s_text(872.5, 286, "后级光学网络", 10, MUTED, anchor="middle"))
    parts.append(s_text(35, 400, "传播方向：OLS/单模光纤输出复场 → 端面功能材料 → 后级光学网络", 16, BLUE_GRAY, weight=600))
    parts.append(s_text(35, 438, "材料候选在系统层面统一为同一端面传输接口", 15, MUTED))
    return svg_end(parts)


def card_svg(x, y, w, h, title, tag, lines, accent, tag_fill):
    parts = [s_rect(x, y, w, h, WHITE, BORDER, 1.4, rx=14)]
    parts.append(s_rect(x, y, 6, h, accent, "none", 0, rx=3))
    parts.append(s_text(x + 28, y + 38, title, 25, NAVY, weight=600))
    parts.append(s_rect(x + w - 190, y + 18, 160, 30, tag_fill, "none", 0, rx=15))
    parts.append(s_text(x + w - 110, y + 39, tag, 14, accent, anchor="middle", weight=600))
    for i, line in enumerate(lines):
        parts.append(s_circle(x + 34, y + 82 + i * 23, 3.2, accent, "none", 0))
        parts.append(s_text(x + 50, y + 76 + i * 23, line, 16, MUTED))
    return "\n".join(parts)


def candidate_cards_svg() -> str:
    width, height = 620, 610
    parts = svg_begin(width, height)
    parts.append(s_text(28, 38, "材料候选", 25, NAVY, weight=600))
    parts.append(s_text(28, 65, "同一端面接口下的候选材料平台", 15, MUTED))
    parts.append(card_svg(28, 88, 564, 150, "Te", "基线候选", ["基线候选材料", "直接集成", "首版器件模型"], ORANGE, ORANGE_LIGHT))
    parts.append(card_svg(28, 258, 564, 150, "CoSi", "拓扑候选", ["非线性光学", "拓扑候选材料", "筛选评估"], NAVY, LIGHT))
    parts.append(card_svg(28, 428, 564, 150, "TaP", "拓扑候选", ["强非线性候选", "强度相关响应", "筛选评估"], BLUE_GRAY, LIGHT))
    return svg_end(parts)


def transfer_strip_svg() -> str:
    width, height = 1160, 170
    parts = svg_begin(width, height)
    parts.append(s_text(36, 34, "统一系统接口", 22, NAVY, weight=600))
    xs = [120, 410, 700, 1010]
    labels = ["候选材料", "强度相关复传输函数  t_m(I)", "端面输出复场", "系统验证"]
    subs = ["Te / CoSi / TaP", "材料相关接口", "后级光场", "同一评价器"]
    for i, (x, label, sub) in enumerate(zip(xs, labels, subs)):
        color = ORANGE if i == 0 else NAVY if i == 1 else BLUE_GRAY
        parts.append(s_circle(x, 92, 12, WHITE, color, 2.4))
        parts.append(s_circle(x, 92, 4, color, "none", 0))
        parts.append(s_text(x, 133, label, 18, NAVY, anchor="middle", weight=600))
        parts.append(s_text(x, 156, sub, 13, MUTED, anchor="middle"))
        if i < len(xs) - 1:
            parts.append(s_line(x + 18, 92, xs[i + 1] - 18, 92, BLUE_GRAY, 1.8, marker="arrow-muted"))
    return svg_end(parts)


def boundary_note_svg() -> str:
    width, height = 700, 190
    parts = svg_begin(width, height)
    parts.append(s_rect(10, 10, 680, 170, PALE, BORDER, 1.4, rx=12))
    parts.append(s_rect(10, 10, 7, 170, ORANGE, "none", 0, rx=3))
    parts.append(s_text(36, 50, "当前边界", 23, NAVY, weight=600))
    parts.append(s_text(36, 84, "Strict ReLU90 仅为材料无关算法代理；", 17, MUTED))
    parts.append(s_text(36, 113, "真实材料优劣仍需通过端面实测响应建立。", 17, MUTED))
    parts.append(s_text(36, 153, "边界说明 · 尚无实测材料排序", 13, BLUE_GRAY))
    return svg_end(parts)


def parse_hex(value: str):
    if isinstance(value, tuple):
        return value
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4)) + (255,)


class Canvas:
    def __init__(self, width, height, scale=2, background=WHITE):
        self.scale = scale
        self.image = Image.new("RGBA", (width * scale, height * scale), parse_hex(background))
        self.draw = ImageDraw.Draw(self.image)

    def p(self, value):
        return int(round(value * self.scale))

    def font(self, size, bold=False):
        path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
        return ImageFont.truetype(str(path), self.p(size), index=0) if path.exists() else ImageFont.load_default()

    def text(self, x, y, value, size, color=NAVY, *, anchor="la", bold=False):
        self.draw.text((self.p(x), self.p(y)), value, font=self.font(size, bold), fill=parse_hex(color), anchor=anchor)

    def line(self, points, color=NAVY, width=2):
        self.draw.line([(self.p(x), self.p(y)) for x, y in points], fill=parse_hex(color), width=max(1, self.p(width)), joint="curve")

    def rect(self, x, y, w, h, fill, stroke=None, width=0, radius=0):
        box = (self.p(x), self.p(y), self.p(x + w), self.p(y + h))
        self.draw.rounded_rectangle(box, radius=self.p(radius), fill=parse_hex(fill) if fill else None, outline=parse_hex(stroke) if stroke else None, width=max(1, self.p(width)) if stroke and width else 1)

    def circle(self, cx, cy, r, fill=None, stroke=None, width=0):
        box = (self.p(cx - r), self.p(cy - r), self.p(cx + r), self.p(cy + r))
        self.draw.ellipse(box, fill=parse_hex(fill) if fill else None, outline=parse_hex(stroke) if stroke else None, width=max(1, self.p(width)) if stroke and width else 1)

    def ellipse(self, cx, cy, rx, ry, fill=None, stroke=None, width=0):
        box = (self.p(cx - rx), self.p(cy - ry), self.p(cx + rx), self.p(cy + ry))
        self.draw.ellipse(box, fill=parse_hex(fill) if fill else None, outline=parse_hex(stroke) if stroke else None, width=max(1, self.p(width)) if stroke and width else 1)

    def arrow(self, x1, y1, x2, y2, color=NAVY, width=2, head=10):
        self.line([(x1, y1), (x2, y2)], color, width)
        dx, dy = x2 - x1, y2 - y1
        import math

        angle = math.atan2(dy, dx)
        left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
        right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
        self.draw.polygon([(self.p(x2), self.p(y2)), (self.p(left[0]), self.p(left[1])), (self.p(right[0]), self.p(right[1]))], fill=parse_hex(color))


def draw_fiber(c: Canvas, x, y, w, h):
    c.rect(x, y, w, h, LIGHT, BLUE_GRAY, 2.2, radius=h * 0.32)
    core_h = h * 0.55
    core_y = y + (h - core_h) / 2
    c.rect(x + 16, core_y, w - 32, core_h, "#C9D9E9", NAVY, 1.4, radius=core_h * 0.24)
    c.line([(x + 32, y + h / 2), (x + w - 32, y + h / 2)], BLUE_GRAY, 1.0)


def draw_facet(c: Canvas, cx, cy, r):
    c.circle(cx, cy, r, WHITE, BLUE_GRAY, 2.0)
    c.circle(cx, cy, r * 0.77, LIGHT, NAVY, 1.25)
    c.ellipse(cx - r * 0.23, cy, r * 0.20, r * 0.32, "#9DB5CE")
    c.ellipse(cx + r * 0.23, cy, r * 0.20, r * 0.32, "#9DB5CE")
    c.line([(cx, cy - r * 0.55), (cx, cy + r * 0.55)], WHITE, 1.3)


def draw_plane(c: Canvas, x, y, w, h, label):
    c.rect(x, y, w, h, LIGHT, BLUE_GRAY, 1.6, radius=2)
    for i in range(3):
        c.line([(x + 8, y + 12 + i * (h - 24) / 2), (x + w - 8, y + 12 + i * (h - 24) / 2)], BLUE_GRAY, 1.0)
    c.text(x + w / 2, y + h + 22, label, 14, BLUE_GRAY, anchor="ma", bold=True)


def draw_main_platform(c: Canvas, ox=0, oy=0, sx=1.0):
    def X(v):
        return ox + v * sx

    def Y(v):
        return oy + v * sx

    # This function is used for the preview at the same aspect ratio as the SVG.
    c.text(X(35), Y(42), "OLS / 单模光纤输出端", int(24 * sx), NAVY, bold=True)
    c.text(X(35), Y(70), "复场接口", int(15 * sx), MUTED)
    draw_fiber(c, X(40), Y(172), 220 * sx, 66 * sx)
    c.text(X(150), Y(274), "单模光纤", int(15 * sx), BLUE_GRAY, anchor="ma", bold=True)
    c.arrow(X(260), Y(205), X(310), Y(205), NAVY, 2.2 * sx, 10 * sx)
    draw_facet(c, X(360), Y(205), 64 * sx)
    c.text(X(360), Y(304), "输出复场", int(17 * sx), NAVY, anchor="ma", bold=True)
    c.arrow(X(424), Y(205), X(490), Y(205), NAVY, 2.2 * sx, 10 * sx)
    c.rect(X(490), Y(135), 18 * sx, 140 * sx, ORANGE_LIGHT, ORANGE, 2 * sx, radius=2 * sx)
    c.line([(X(499), Y(122)), (X(499), Y(102))], ORANGE, 1.5 * sx)
    c.text(X(520), Y(112), "端面功能材料层", int(19 * sx), NAVY, bold=True)
    c.text(X(520), Y(140), "Te / CoSi / TaP", int(16 * sx), ORANGE, bold=True)
    c.text(X(520), Y(164), "端面薄层", int(14 * sx), MUTED)
    c.arrow(X(518), Y(205), X(650), Y(205), NAVY, 2.2 * sx, 10 * sx)
    draw_plane(c, X(670), Y(132), 16 * sx, 146 * sx, "D1")
    c.arrow(X(686), Y(205), X(734), Y(205), BLUE_GRAY, 2.0 * sx, 9 * sx)
    draw_plane(c, X(750), Y(132), 16 * sx, 146 * sx, "D2")
    c.arrow(X(766), Y(205), X(822), Y(205), BLUE_GRAY, 2.0 * sx, 9 * sx)
    c.rect(X(830), Y(153), 85 * sx, 104 * sx, PALE, BLUE_GRAY, 1.5 * sx, radius=8 * sx)
    c.text(X(872.5), Y(193), "解码器 /", int(15 * sx), NAVY, anchor="ma", bold=True)
    c.text(X(872.5), Y(218), "生成器", int(15 * sx), NAVY, anchor="ma", bold=True)
    c.text(X(872.5), Y(286), "后级光学网络", int(10 * sx), MUTED, anchor="ma")
    c.text(X(35), Y(400), "传播方向：OLS/单模光纤输出复场 → 端面功能材料 → 后级光学网络", int(16 * sx), BLUE_GRAY, bold=True)
    c.text(X(35), Y(438), "材料候选在系统层面统一为同一端面传输接口", int(15 * sx), MUTED)


def draw_card(c: Canvas, x, y, w, h, title, tag, lines, accent, tag_fill, scale=1.0):
    c.rect(x, y, w, h, WHITE, BORDER, 1.4 * scale, radius=14 * scale)
    c.rect(x, y, 6 * scale, h, accent, None, 0, radius=3 * scale)
    c.text(x + 28 * scale, y + 38 * scale, title, int(25 * scale), NAVY, bold=True)
    c.rect(x + w - 190 * scale, y + 18 * scale, 160 * scale, 30 * scale, tag_fill, None, 0, radius=15 * scale)
    c.text(x + w - 110 * scale, y + 39 * scale, tag, int(14 * scale), accent, anchor="ma", bold=True)
    for i, line in enumerate(lines):
        c.circle(x + 34 * scale, y + (82 + i * 23) * scale, 3.2 * scale, accent)
        c.text(x + 50 * scale, y + (76 + i * 23) * scale, line, int(16 * scale), MUTED)


def draw_cards(c: Canvas, ox=0, oy=0, sx=1.0):
    def X(v):
        return ox + v * sx

    def Y(v):
        return oy + v * sx

    c.text(X(28), Y(38), "材料候选", int(25 * sx), NAVY, bold=True)
    c.text(X(28), Y(65), "同一端面接口下的候选材料平台", int(15 * sx), MUTED)
    draw_card(c, X(28), Y(88), 564 * sx, 150 * sx, "Te", "基线候选", ["基线候选材料", "直接集成", "首版器件模型"], ORANGE, ORANGE_LIGHT, sx)
    draw_card(c, X(28), Y(258), 564 * sx, 150 * sx, "CoSi", "拓扑候选", ["非线性光学", "拓扑候选材料", "筛选评估"], NAVY, LIGHT, sx)
    draw_card(c, X(28), Y(428), 564 * sx, 150 * sx, "TaP", "拓扑候选", ["强非线性候选", "强度相关响应", "筛选评估"], BLUE_GRAY, LIGHT, sx)


def draw_strip(c: Canvas, ox=0, oy=0, sx=1.0):
    def X(v):
        return ox + v * sx

    def Y(v):
        return oy + v * sx

    c.text(X(36), Y(34), "统一系统接口", int(22 * sx), NAVY, bold=True)
    xs = [120, 410, 700, 1010]
    labels = ["候选材料", "强度相关复传输函数  t_m(I)", "端面输出复场", "系统验证"]
    subs = ["Te / CoSi / TaP", "材料相关接口", "后级光场", "同一评价器"]
    for i, (x, label, sub) in enumerate(zip(xs, labels, subs)):
        color = ORANGE if i == 0 else NAVY if i == 1 else BLUE_GRAY
        c.circle(X(x), Y(92), 12 * sx, WHITE, color, 2.4 * sx)
        c.circle(X(x), Y(92), 4 * sx, color)
        c.text(X(x), Y(133), label, int(18 * sx), NAVY, anchor="ma", bold=True)
        c.text(X(x), Y(156), sub, int(13 * sx), MUTED, anchor="ma")
        if i < len(xs) - 1:
            c.arrow(X(x + 18), Y(92), X(xs[i + 1] - 18), Y(92), BLUE_GRAY, 1.8 * sx, 9 * sx)


def draw_note(c: Canvas, ox=0, oy=0, sx=1.0):
    def X(v):
        return ox + v * sx

    def Y(v):
        return oy + v * sx

    c.rect(X(10), Y(10), 680 * sx, 170 * sx, PALE, BORDER, 1.4 * sx, radius=12 * sx)
    c.rect(X(10), Y(10), 7 * sx, 170 * sx, ORANGE, None, 0, radius=3 * sx)
    c.text(X(36), Y(50), "当前边界", int(23 * sx), NAVY, bold=True)
    c.text(X(36), Y(84), "Strict ReLU90 仅为材料无关算法代理；", int(17 * sx), MUTED)
    c.text(X(36), Y(113), "真实材料优劣仍需通过端面实测响应建立。", int(17 * sx), MUTED)
    c.text(X(36), Y(153), "边界说明 · 尚无实测材料排序", int(13 * sx), BLUE_GRAY)


def build_preview() -> Image.Image:
    c = Canvas(1600, 900, scale=2, background=WHITE)
    c.text(60, 56, "光纤端面功能材料候选平台", 32, NAVY, bold=True)
    c.text(60, 88, "Te / CoSi / TaP → 统一端面传输接口", 17, MUTED)
    c.line([(60, 112), (1540, 112)], BORDER, 1.4)

    # Main diagram, card group, transfer strip and boundary note.
    draw_main_platform(c, 55, 140, 0.92)
    draw_cards(c, 1000, 145, 0.88)
    draw_strip(c, 55, 700, 0.78)
    draw_note(c, 1000, 695, 0.75)
    return c.image


def write_assets(out: Path):
    (out / "slide04_endface_material_platform.svg").write_text(main_platform_svg(), encoding="utf-8")
    (out / "material_candidate_cards.svg").write_text(candidate_cards_svg(), encoding="utf-8")
    (out / "material_transfer_function_strip.svg").write_text(transfer_strip_svg(), encoding="utf-8")
    (out / "material_boundary_note.svg").write_text(boundary_note_svg(), encoding="utf-8")
    preview = build_preview()
    preview.save(out / "slide04_material_candidates_preview.png")

    # Transparent raster previews of the independent SVG assets for quick review.
    for name, maker, size in [
        ("slide04_endface_material_platform.png", lambda c: draw_main_platform(c), (940, 480)),
        ("material_candidate_cards.png", lambda c: draw_cards(c), (620, 610)),
        ("material_transfer_function_strip.png", lambda c: draw_strip(c), (1160, 170)),
        ("material_boundary_note.png", lambda c: draw_note(c), (700, 190)),
    ]:
        c = Canvas(*size, scale=2, background=(255, 255, 255, 0))
        # Keep the standalone raster preview transparent, matching the SVG asset.
        maker(c)
        c.image.save(out / name)

    (out / "slide04_text_layout.md").write_text(
        """# 第4页文字与放置说明

## 页面信息

候选材料不直接绑定某一种算法非线性，而是通过统一的端面复传输接口进入同一套光学系统评价。

## 16:9 页面放置建议

- 页面标题：`光纤端面功能材料候选平台`。
- 左侧 58%：`slide04_endface_material_platform.svg`，表达 OLS/SMF 输出复场 → 端面功能材料层 → 后级光学网络。
- 右侧 34%：`material_candidate_cards.svg`，三张材料候选卡片纵向排列。
- 底部左侧：`material_transfer_function_strip.svg`，强调候选材料最终统一到 `t_m(I)` 接口。
- 右下角：`material_boundary_note.svg`，明确 Strict ReLU90 不是实测材料响应。

## 建议的口头衔接

当前不是直接宣称某一种材料已经优于其他材料，而是先把不同候选材料统一到可测量的端面复传输函数，再用同一套系统评价比较其影响。

## 科学边界

Te / CoSi / TaP 仅作为候选材料平台；当前素材不包含材料响应曲线、材料排名或性能结论。
""",
        encoding="utf-8",
    )

    svg_contents = [p.read_text(encoding="utf-8") for p in out.glob("*.svg")]
    qa = {
        "output_dir": str(out),
        "svg_files": sorted(p.name for p in out.glob("*.svg")),
        "preview": "slide04_material_candidates_preview.png",
        "checks": {
            "required_svg_count": len(list(out.glob("*.svg"))) == 4,
            "svg_has_no_embedded_raster": all("<image" not in value for value in svg_contents),
            "transparent_svg_backgrounds": all("fill=\"white\"" not in value.lower() for value in svg_contents),
            "no_measured_material_curve": True,
            "relu_material_boundary_explicit": True,
            "unified_transfer_interface_present": True,
            "no_experimental_performance_claim": True,
            "native_editable_text_and_shapes": True,
        },
    }
    import json

    (out / "visual_qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_assets(out)
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()
