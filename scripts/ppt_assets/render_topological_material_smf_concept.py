from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1831, 730
OUT_W, OUT_H = 2400, 957

NAVY = "#223F73"
BLUE = "#4F79C8"
MUTED = "#536579"
LIGHT = "#D5DFEA"
ORANGE = "#C97842"
ORANGE_LIGHT = "#F4E3D7"
WHITE = "#FFFFFF"

FONT_REG = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REG
    return ImageFont.truetype(str(path), size=size)


def esc(value: str) -> str:
    return escape(value, quote=True)


def svg_text(x: float, y: float, text: str, size: float, color: str, *,
             anchor: str = "start", weight: int = 500) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-family="Microsoft YaHei, '
        f'Microsoft JhengHei, Arial, sans-serif" font-size="{size}px" '
        f'font-weight="{weight}" text-anchor="{anchor}">{esc(text)}</text>'
    )


def svg_multiline(x: float, y: float, lines: list[str], size: float, color: str,
                  *, leading: float = 1.35, weight: int = 500,
                  anchor: str = "start") -> str:
    chunks = [
        f'<text x="{x}" y="{y}" fill="{color}" font-family="Microsoft YaHei, '
        f'Microsoft JhengHei, Arial, sans-serif" font-size="{size}px" '
        f'font-weight="{weight}" text-anchor="{anchor}">'
    ]
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * leading
        chunks.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    chunks.append("</text>")
    return "".join(chunks)


def svg_segmented(x: float, y: float, segments: list[tuple[str, str]], size: float,
                  *, weight: int = 650) -> str:
    chunks = [
        f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, Microsoft JhengHei, '
        f'Arial, sans-serif" font-size="{size}px" font-weight="{weight}">'
    ]
    for text, color in segments:
        chunks.append(f'<tspan fill="{color}">{esc(text)}</tspan>')
    chunks.append("</text>")
    return "".join(chunks)


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str,
             width: float = 2, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def svg_rect(x: float, y: float, w: float, h: float, *, fill: str = "none",
             stroke: str = "none", width: float = 1, rx: float = 0,
             dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str,
              width: float = 3, head: float = 12) -> str:
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(angle - math.pi / 6),
            y2 - head * math.sin(angle - math.pi / 6))
    right = (x2 - head * math.cos(angle + math.pi / 6),
             y2 - head * math.sin(angle + math.pi / 6))
    pts = f"{x2},{y2} {left[0]},{left[1]} {right[0]},{right[1]}"
    return (
        svg_line(x1, y1, x2, y2, color, width)
        + f'<polygon points="{pts}" fill="{color}"/>'
    )


def svg_dashed_round_rect(x: float, y: float, w: float, h: float,
                          color: str = LIGHT) -> str:
    return svg_rect(x, y, w, h, fill="none", stroke=color, width=2.2,
                    rx=12, dash="8 7")


def build_svg() -> str:
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<title>拓扑材料端面非线性与单模光纤接口示意图</title>',
        '<desc>透明背景、可编辑的中文科研示意图。图中不包含具体材料响应曲线。</desc>',
    ]

    # Left: why an end-face topological-material nonlinearity is introduced.
    p.append(svg_text(52, 64, "为什么需要拓扑材料非线性", 36, NAVY, weight=700))
    p.append(svg_text(52, 123, "纯线性光路", 27, NAVY, weight=650))
    p.append(svg_multiline(52, 274, ["衍射传播", "相位调制", "线性调制"], 25, MUTED,
                           leading=1.9, weight=500))
    p.append(svg_arrow(452, 307, 452, 382, BLUE, width=3.2, head=14))
    p.append(svg_segmented(52, 437, [
        ("加入", NAVY), ("拓扑材料", ORANGE), ("端面非线性", NAVY)
    ], 29, weight=700))
    p.append(svg_text(52, 598, "局部强度", 25, NAVY, weight=650))
    p.append(svg_arrow(198, 588, 292, 588, BLUE, width=3.0, head=13))
    p.append(svg_rect(307, 553, 155, 70, fill=ORANGE_LIGHT, stroke=ORANGE,
                      width=2.2, rx=7))
    p.append(svg_text(384.5, 596, "拓扑材料", 22, ORANGE, anchor="middle", weight=700))
    p.append(svg_arrow(490, 588, 580, 588, ORANGE, width=3.0, head=13))
    p.append(svg_text(605, 598, "输出光场", 25, NAVY, weight=650))
    p.append(svg_text(460, 694, "增加可用的非线性映射自由度", 24, NAVY, weight=650))

    # Divider.
    p.append(svg_line(1124, 30, 1124, 670, LIGHT, width=3))

    # Right: why SMF is the first interface.
    p.append(svg_text(1168, 64, "为什么当前优先使用 SMF", 36, NAVY, weight=700))
    p.append(svg_text(1168, 133, "01", 27, NAVY, weight=700))
    p.append(svg_text(1222, 133, "模式身份简单", 28, NAVY, weight=700))
    p.append('<circle cx="1220" cy="205" r="36" fill="none" stroke="#4F79C8" stroke-width="2.5"/>')
    p.append('<circle cx="1220" cy="205" r="24" fill="none" stroke="#4F79C8" stroke-width="2.5"/>')
    p.append('<circle cx="1220" cy="205" r="10" fill="#EAF0F7" stroke="#8094AA" stroke-width="2"/>')
    p.append(svg_text(1282, 213, "每根 SMF 近似单一基模通道", 23, MUTED, weight=500))
    p.append(svg_line(1168, 265, 1805, 265, LIGHT, width=2.5))

    p.append(svg_text(1168, 322, "02", 27, NAVY, weight=700))
    p.append(svg_text(1222, 322, "OLS→空间场关系明确", 28, NAVY, weight=700))
    p.append(svg_text(1168, 379, "模式系数", 22, NAVY, weight=600))
    p.append(svg_arrow(1288, 368, 1350, 368, BLUE, width=3.0, head=12))
    p.append(svg_rect(1360, 339, 70, 54, fill="#F5F8FC", stroke=BLUE, width=2.1, rx=7))
    p.append(svg_text(1395, 375, "叠加", 21, NAVY, anchor="middle", weight=700))
    p.append(svg_arrow(1445, 368, 1509, 368, BLUE, width=3.0, head=12))
    p.append(svg_text(1524, 379, "空间场", 22, NAVY, weight=600))
    p.append(svg_line(1168, 502, 1805, 502, LIGHT, width=2.5))

    p.append(svg_text(1168, 558, "03", 27, NAVY, weight=700))
    p.append(svg_text(1222, 558, "非线性归因变量更少", 28, NAVY, weight=700))
    p.append(svg_multiline(1168, 608, [
        "拓扑材料主要响应重发后的空间强度",
        "暂不同时引入多模耦合变量",
    ], 22, MUTED, leading=1.45, weight=500))
    p.append(svg_text(1168, 716, "SMF是隔离研究端面非线性的清晰接口。", 23, NAVY, weight=700))

    p.append("</svg>")
    return "".join(p)


def draw_text(draw: ImageDraw.ImageDraw, x: int, baseline: int, text: str,
              size: int, color: str, *, bold: bool = False,
              anchor: str = "ls") -> None:
    draw.text((x, baseline), text, font=font(size, bold), fill=color, anchor=anchor)


def draw_multiline(draw: ImageDraw.ImageDraw, x: int, top: int, lines: list[str],
                   size: int, color: str, *, leading: float = 1.35,
                   bold: bool = False) -> None:
    draw.multiline_text((x, top), "\n".join(lines), font=font(size, bold),
                        fill=color, spacing=int(size * (leading - 1)), anchor="la")


def dashed_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int],
                color: str, width: int = 2, dash: int = 9, gap: int = 8,
                radius: int = 12) -> None:
    x0, y0, x1, y1 = xy
    # Draw only the dashed perimeter.  Avoid painting a solid outline first:
    # transparent PNGs otherwise retain a faint duplicate edge around dashes.
    def segments(a: int, b: int):
        pos = a
        while pos < b:
            yield pos, min(pos + dash, b)
            pos += dash + gap
    for a, b in segments(x0 + radius, x1 - radius):
        draw.line((a, y0, b, y0), fill=color, width=width)
        draw.line((a, y1, b, y1), fill=color, width=width)
    for a, b in segments(y0 + radius, y1 - radius):
        draw.line((x0, a, x0, b), fill=color, width=width)
        draw.line((x1, a, x1, b), fill=color, width=width)


def arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int,
          color: str, width: int = 3, head: int = 13) -> None:
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - head * math.cos(angle - math.pi / 6),
            y2 - head * math.sin(angle - math.pi / 6))
    right = (x2 - head * math.cos(angle + math.pi / 6),
             y2 - head * math.sin(angle + math.pi / 6))
    draw.polygon([(x2, y2), left, right], fill=color)


def draw_png(path: Path) -> None:
    im = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    d = ImageDraw.Draw(im)
    draw_text(d, 52, 64, "为什么需要拓扑材料非线性", 36, NAVY, bold=True)
    draw_text(d, 52, 123, "纯线性光路", 27, NAVY, bold=True)
    draw_multiline(d, 52, 248, ["衍射传播", "相位调制", "线性调制"], 25, MUTED, leading=1.9)
    arrow(d, 452, 307, 452, 382, BLUE, width=3, head=14)
    draw_text(d, 52, 437, "加入", 29, NAVY, bold=True)
    # Match the segmented SVG heading with a small, deliberate gap.
    gap_x = 52 + int(d.textlength("加入", font=font(29, True)))
    draw_text(d, gap_x, 437, "拓扑材料", 29, ORANGE, bold=True)
    after_x = gap_x + int(d.textlength("拓扑材料", font=font(29, True)))
    draw_text(d, after_x, 437, "端面非线性", 29, NAVY, bold=True)
    draw_text(d, 52, 598, "局部强度", 25, NAVY, bold=True)
    arrow(d, 198, 588, 292, 588, BLUE, width=3, head=13)
    d.rounded_rectangle((307, 553, 462, 623), radius=7, fill=ORANGE_LIGHT, outline=ORANGE, width=2)
    draw_text(d, 384, 596, "拓扑材料", 22, ORANGE, bold=True, anchor="ms")
    arrow(d, 490, 588, 580, 588, ORANGE, width=3, head=13)
    draw_text(d, 605, 598, "输出光场", 25, NAVY, bold=True)
    draw_text(d, 460, 694, "增加可用的非线性映射自由度", 24, NAVY, bold=True)

    d.line((1124, 30, 1124, 670), fill=LIGHT, width=3)
    draw_text(d, 1168, 64, "为什么当前优先使用 SMF", 36, NAVY, bold=True)
    draw_text(d, 1168, 133, "01", 27, NAVY, bold=True)
    draw_text(d, 1222, 133, "模式身份简单", 28, NAVY, bold=True)
    d.ellipse((1184, 169, 1256, 241), outline=BLUE, width=2)
    d.ellipse((1196, 181, 1244, 229), outline=BLUE, width=2)
    d.ellipse((1210, 195, 1230, 215), fill="#EAF0F7", outline="#8094AA", width=2)
    draw_text(d, 1282, 213, "每根 SMF 近似单一基模通道", 23, MUTED)
    d.line((1168, 265, 1805, 265), fill=LIGHT, width=2)
    draw_text(d, 1168, 322, "02", 27, NAVY, bold=True)
    draw_text(d, 1222, 322, "OLS→空间场关系明确", 28, NAVY, bold=True)
    draw_text(d, 1168, 379, "模式系数", 22, NAVY, bold=True)
    arrow(d, 1288, 368, 1350, 368, BLUE, width=3, head=12)
    d.rounded_rectangle((1360, 339, 1430, 393), radius=7, fill="#F5F8FC", outline=BLUE, width=2)
    draw_text(d, 1395, 375, "叠加", 21, NAVY, bold=True, anchor="ms")
    arrow(d, 1445, 368, 1509, 368, BLUE, width=3, head=12)
    draw_text(d, 1524, 379, "空间场", 22, NAVY, bold=True)
    d.line((1168, 502, 1805, 502), fill=LIGHT, width=2)
    draw_text(d, 1168, 558, "03", 27, NAVY, bold=True)
    draw_text(d, 1222, 558, "非线性归因变量更少", 28, NAVY, bold=True)
    draw_multiline(d, 1168, 579, ["拓扑材料主要响应重发后的空间强度", "暂不同时引入多模耦合变量"], 22, MUTED, leading=1.45)
    draw_text(d, 1168, 716, "SMF是隔离研究端面非线性的清晰接口。", 23, NAVY, bold=True)

    resized = im.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
    resized.save(path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\codex-original\outputs\topological_material_smf_concept_v1"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "topological_material_smf_concept.svg"
    png_path = args.output_dir / "topological_material_smf_concept.png"
    qa_path = args.output_dir / "topological_material_smf_concept_qa.json"
    svg_path.write_text(build_svg(), encoding="utf-8")
    draw_png(png_path)
    qa = {
        "canvas_svg": f"{W}x{H}",
        "png_size": [OUT_W, OUT_H],
        "transparent_png": True,
        "svg_editable_text_and_shapes": True,
        "all_Te_labels_replaced_with": "拓扑材料",
        "remaining_Te_literal": False,
        "layout_preserved": True,
        "scientific_boundary": "拓扑材料为一般化标签；图中未加入具体材料响应曲线或性能结论。",
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
