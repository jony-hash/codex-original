from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 583, 497
OUT_W, OUT_H = 1600, 1364

NAVY = "#223F73"
MUTED = "#536579"
LIGHT = "#D5DFEA"
ORANGE = "#C97842"

FONT_REG = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REG
    return ImageFont.truetype(str(path), size=size)


def svg_text(x: float, y: float, text: str, size: float, color: str,
             *, weight: int = 500) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-family="Microsoft YaHei, '
        f'Microsoft JhengHei, Arial, sans-serif" font-size="{size}px" '
        f'font-weight="{weight}">{escape(text)}</text>'
    )


def svg_segmented(x: float, y: float, segments: list[tuple[str, str]],
                  size: float, *, weight: int = 700) -> str:
    parts = [
        f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, Microsoft JhengHei, '
        f'Arial, sans-serif" font-size="{size}px" font-weight="{weight}">'
    ]
    for text, color in segments:
        parts.append(f'<tspan fill="{color}">{escape(text)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str,
             width: float = 2) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<title>作用对象与物理目标示意图</title>',
        '<desc>透明背景、可编辑的中文科研示意图；虚线已移除，材料标签统一为拓扑材料。</desc>',
        svg_segmented(29, 74, [("A", NAVY), ("   作用对象", NAVY)], 25, weight=700),
        svg_text(29, 112, "SMF输出复空间场", 26, NAVY, weight=700),
        svg_text(29, 220, "局部强度由端面复场得到", 18, MUTED, weight=500),
        svg_line(29, 252, 519, 252, LIGHT, 2.3),
        svg_segmented(29, 306, [("B", ORANGE), ("   物理目标", NAVY)], 25, weight=700),
        svg_segmented(29, 344, [("拓扑材料", ORANGE), ("强度相关复传输", NAVY)], 26, weight=700),
        svg_text(29, 452, "幅度响应与相位响应尚待实测", 18, MUTED, weight=500),
        '</svg>',
    ]
    return "".join(parts)


def draw_text(draw: ImageDraw.ImageDraw, x: int, baseline: int, text: str,
              size: int, color: str, *, bold: bool = False) -> None:
    draw.text((x, baseline), text, fill=color, font=font(size, bold), anchor="ls")


def draw_png(path: Path) -> None:
    image = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw_text(draw, 29, 74, "A   作用对象", 25, NAVY, bold=True)
    draw_text(draw, 29, 112, "SMF输出复空间场", 26, NAVY, bold=True)
    draw_text(draw, 29, 220, "局部强度由端面复场得到", 18, MUTED)
    draw.line((29, 252, 519, 252), fill=LIGHT, width=2)
    draw_text(draw, 29, 306, "B", 25, ORANGE, bold=True)
    x = 29 + int(draw.textlength("B   ", font=font(25, True)))
    draw_text(draw, x, 306, "物理目标", 25, NAVY, bold=True)
    draw_text(draw, 29, 344, "拓扑材料", 26, ORANGE, bold=True)
    x = 29 + int(draw.textlength("拓扑材料", font=font(26, True)))
    draw_text(draw, x, 344, "强度相关复传输", 26, NAVY, bold=True)
    draw_text(draw, 29, 452, "幅度响应与相位响应尚待实测", 18, MUTED)
    image.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS).save(path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\codex-original\outputs\topological_material_effect_target_v1"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "effect_target_topological_material.svg"
    png_path = args.output_dir / "effect_target_topological_material.png"
    qa_path = args.output_dir / "effect_target_topological_material_qa.json"
    svg_path.write_text(build_svg(), encoding="utf-8")
    draw_png(png_path)
    qa = {
        "canvas_svg": f"{W}x{H}",
        "png_size": [OUT_W, OUT_H],
        "transparent_png": True,
        "dashed_lines_removed": True,
        "Te_replacement": "拓扑材料",
        "editable_svg": True,
        "scientific_boundary": "仅修改标签，不添加材料响应数据或物理结论。",
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
