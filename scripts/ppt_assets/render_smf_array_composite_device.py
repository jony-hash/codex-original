from __future__ import annotations

"""Render an editable SVG and transparent PNG for the SMF-array callout."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "smf_array_composite_device_v1"
SVG_W, SVG_H = 1600, 640
PNG_W, PNG_H = 2400, 960
SCALE = PNG_W / SVG_W

NAVY = "#223F73"
BLUE = "#4F79C8"
BLUE_GRAY = "#8094AA"
LIGHT = "#EAF0F7"
PALE = "#F5F8FC"
BORDER = "#D5DFEA"
MUTED = "#536579"
ORANGE = "#C97842"
ORANGE_LIGHT = "#F4E3D7"
CORE = "#AFC4D7"
CORE_DARK = "#6E93B5"
GREY = "#68757D"

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def f(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def svg_text(x: float, y: float, value: str, size: float, color: str = NAVY, *, anchor: str = "middle", weight: int = 400) -> str:
    return (
        f'<text x="{f(x)}" y="{f(y)}" text-anchor="{anchor}" '
        f'dominant-baseline="middle" font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif" '
        f'font-size="{f(size)}px" font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def svg_lines(x: float, y: float, lines: list[str], size: float, color: str = NAVY, *, anchor: str = "middle", weight: int = 400, leading: float = 1.25) -> str:
    parts = [
        f'<text x="{f(x)}" y="{f(y)}" text-anchor="{anchor}" '
        f'dominant-baseline="middle" font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif" '
        f'font-size="{f(size)}px" font-weight="{weight}" fill="{color}">'
    ]
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size * leading
        parts.append(f'<tspan x="{f(x)}" dy="{f(dy)}">{esc(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def svg_begin() -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" viewBox="0 0 {SVG_W} {SVG_H}">',
        "<defs>",
        f'<marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker>',
        f'<marker id="arrow-navy" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 10 5 L 0 10 z" fill="{NAVY}"/></marker>',
        f'<marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker>',
        "</defs>",
    ]


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", sw: float = 0, rx: float = 0, opacity: float = 1.0) -> str:
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="{f(rx)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" opacity="{f(opacity)}"/>'


def svg_circle(cx: float, cy: float, r: float, fill: str, stroke: str = "none", sw: float = 0, opacity: float = 1.0) -> str:
    return f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}" opacity="{f(opacity)}"/>'


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, sw: float = 2, dash: str | None = None, marker: str | None = None, opacity: float = 1.0) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{stroke}" stroke-width="{f(sw)}" stroke-linecap="round" opacity="{f(opacity)}"{dash_attr}{marker_attr}/>'


def svg_polyline(points: list[tuple[float, float]], stroke: str, sw: float = 2, dash: str | None = None, marker: str | None = None) -> str:
    point_text = " ".join(f"{f(x)},{f(y)}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return f'<polyline points="{point_text}" fill="none" stroke="{stroke}" stroke-width="{f(sw)}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}{marker_attr}/>'


def build_svg() -> str:
    parts = svg_begin()

    # Left: a bounded but transparent-friendly array region.
    parts.append(svg_rect(70, 135, 370, 300, PALE, BORDER, 2, rx=22))
    parts.append(svg_text(255, 92, "10×10 单模光纤阵列", 25, NAVY, weight=700))
    parts.append(svg_text(255, 465, "每个圆点代表一根单模光纤", 15, MUTED))

    grid_x0, grid_y0, spacing = 135, 170, 25
    selected_col, selected_row = 7, 4
    selected_x = grid_x0 + selected_col * spacing
    selected_y = grid_y0 + selected_row * spacing
    for row in range(10):
        for col in range(10):
            cx = grid_x0 + col * spacing
            cy = grid_y0 + row * spacing
            is_selected = row == selected_row and col == selected_col
            parts.append(svg_circle(cx, cy, 7.2, CORE if not is_selected else CORE_DARK, NAVY if is_selected else BLUE_GRAY, 1.2 if is_selected else 0.9))
    parts.append(svg_rect(selected_x - 17, selected_y - 17, 34, 34, "none", BLUE, 2.5, rx=5))
    parts.append(svg_text(selected_x, selected_y - 31, "代表性光纤", 13, BLUE, weight=700))

    # Middle: explicit extraction / enlargement callout.
    parts.append(svg_polyline([(selected_x + 20, selected_y), (560, selected_y), (760, 300)], BLUE, 2.4, dash="8 6", marker="arrow-blue"))
    parts.append(svg_text(570, 230, "单根光纤放大引导", 18, BLUE, weight=700))
    parts.append(svg_text(570, 255, "从阵列中抽取一根代表性光纤", 14, MUTED))

    # Right: enlarged single-fiber composite device.
    parts.append(svg_text(1035, 92, "单根光纤端面放大示意", 25, NAVY, weight=700))
    parts.append(svg_text(1035, 125, "光纤复合器件的结构组成", 15, MUTED))

    body_x, body_y, body_w, body_h = 760, 200, 400, 150
    parts.append(svg_rect(body_x, body_y, body_w, body_h, LIGHT, BLUE_GRAY, 2.2, rx=22))
    parts.append(svg_rect(800, 260, 330, 46, "#C9D9E9", NAVY, 1.8, rx=14))
    parts.append(svg_line(825, 283, 1105, 283, BLUE_GRAY, 1.2, marker="arrow-navy"))
    parts.append(svg_text(945, 227, "SMF 光纤柱体", 18, NAVY, weight=700))
    parts.append(svg_text(945, 327, "光沿光纤传播", 14, MUTED))

    # Flat output facet, followed immediately by the orange end-face film.
    parts.append(svg_rect(1150, 200, 40, 150, "#F3F6F7", GREY, 2.0, rx=3))
    parts.append(svg_rect(1162, 260, 17, 46, "#9DB5CE", BLUE_GRAY, 1.0, rx=5))
    parts.append(svg_rect(1190, 192, 14, 166, ORANGE_LIGHT, ORANGE, 2.0, rx=2))
    parts.append(svg_line(1194, 198, 1194, 352, ORANGE, 1.0))
    parts.append(svg_line(1200, 198, 1200, 352, ORANGE, 1.0))

    parts.append(svg_text(1170, 392, "输出端面", 15, NAVY, weight=700))
    parts.append(svg_line(1204, 235, 1230, 235, ORANGE, 1.7, marker="arrow-orange"))
    parts.append(svg_lines(1240, 220, ["端面功能材料层", "候选材料：Te / CoSi / TaP"], 16, ORANGE, anchor="start", weight=700, leading=1.35))
    parts.append(svg_text(1035, 430, "光纤复合器件", 21, NAVY, weight=700))
    parts.append(svg_text(1035, 460, "材料仅沉积于输出端面", 14, MUTED))

    parts.append("</svg>")
    return "\n".join(parts)


def font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(FONT_PATH), int(round(size * SCALE)), index=0)


def xy(value: float) -> int:
    return int(round(value * SCALE))


def draw_text(draw: ImageDraw.ImageDraw, x: float, y: float, value: str, size: float, fill: str = NAVY, *, anchor: str = "mm", bold: bool = False, spacing: float = 1.25) -> None:
    draw.multiline_text((xy(x), xy(y)), value, fill=fill, font=font(size, bold), anchor=anchor, spacing=int(round(size * SCALE * (spacing - 1))))


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: str, width: float = 2.4, dashed: bool = False) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        import math

        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0.0
        while pos < length - 12:
            a = pos
            b = min(pos + 8, length - 12)
            draw.line((xy(x1 + ux * a), xy(y1 + uy * a), xy(x1 + ux * b), xy(y1 + uy * b)), fill=fill, width=max(1, xy(width)))
            pos += 14
    else:
        draw.line((xy(x1), xy(y1), xy(x2), xy(y2)), fill=fill, width=max(1, xy(width)))
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    head = 12
    left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
    right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
    draw.polygon([(xy(x2), xy(y2)), (xy(left[0]), xy(left[1])), (xy(right[0]), xy(right[1]))], fill=fill)


def build_png() -> Image.Image:
    image = Image.new("RGBA", (PNG_W, PNG_H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # Array region and labels.
    draw.rounded_rectangle((xy(70), xy(135), xy(440), xy(435)), radius=xy(22), fill=PALE, outline=BORDER, width=xy(2))
    draw_text(draw, 255, 92, "10×10 单模光纤阵列", 25, NAVY, bold=True)
    draw_text(draw, 255, 465, "每个圆点代表一根单模光纤", 15, MUTED)

    grid_x0, grid_y0, spacing = 135, 170, 25
    selected_col, selected_row = 7, 4
    selected_x = grid_x0 + selected_col * spacing
    selected_y = grid_y0 + selected_row * spacing
    for row in range(10):
        for col in range(10):
            cx, cy = grid_x0 + col * spacing, grid_y0 + row * spacing
            is_selected = row == selected_row and col == selected_col
            radius = 7.2
            draw.ellipse((xy(cx - radius), xy(cy - radius), xy(cx + radius), xy(cy + radius)), fill=CORE_DARK if is_selected else CORE, outline=NAVY if is_selected else BLUE_GRAY, width=xy(1.2 if is_selected else 0.9))
    draw.rounded_rectangle((xy(selected_x - 17), xy(selected_y - 17), xy(selected_x + 17), xy(selected_y + 17)), radius=xy(5), outline=BLUE, width=xy(2.5))
    draw_text(draw, selected_x, selected_y - 31, "代表性光纤", 13, BLUE, bold=True)

    draw_arrow(draw, (selected_x + 20, selected_y), (760, 300), BLUE, dashed=True)
    draw_text(draw, 570, 230, "单根光纤放大引导", 18, BLUE, bold=True)
    draw_text(draw, 570, 255, "从阵列中抽取一根代表性光纤", 14, MUTED)

    # Enlarged device.
    draw_text(draw, 1035, 92, "单根光纤端面放大示意", 25, NAVY, bold=True)
    draw_text(draw, 1035, 125, "光纤复合器件的结构组成", 15, MUTED)
    draw.rounded_rectangle((xy(760), xy(200), xy(1160), xy(350)), radius=xy(22), fill=LIGHT, outline=BLUE_GRAY, width=xy(2.2))
    draw.rounded_rectangle((xy(800), xy(260), xy(1130), xy(306)), radius=xy(14), fill="#C9D9E9", outline=NAVY, width=xy(1.8))
    draw_arrow(draw, (825, 283), (1105, 283), BLUE_GRAY, width=1.8)
    draw_text(draw, 945, 227, "SMF 光纤柱体", 18, NAVY, bold=True)
    draw_text(draw, 945, 327, "光沿光纤传播", 14, MUTED)
    draw.rounded_rectangle((xy(1150), xy(200), xy(1190), xy(350)), radius=xy(3), fill="#F3F6F7", outline="#68757D", width=xy(2))
    draw.rounded_rectangle((xy(1162), xy(260), xy(1179), xy(306)), radius=xy(5), fill="#9DB5CE", outline=BLUE_GRAY, width=xy(1))
    draw.rounded_rectangle((xy(1190), xy(192), xy(1204), xy(358)), radius=xy(2), fill=ORANGE_LIGHT, outline=ORANGE, width=xy(2))
    draw.line((xy(1194), xy(198), xy(1194), xy(352)), fill=ORANGE, width=xy(1))
    draw.line((xy(1200), xy(198), xy(1200), xy(352)), fill=ORANGE, width=xy(1))
    draw_text(draw, 1170, 392, "输出端面", 15, NAVY, bold=True)
    draw_arrow(draw, (1204, 235), (1230, 235), ORANGE, width=1.7)
    draw_text(draw, 1240, 220, "端面功能材料层\n候选材料：Te / CoSi / TaP", 16, ORANGE, anchor="lm", bold=True, spacing=1.35)
    draw_text(draw, 1035, 430, "光纤复合器件", 21, NAVY, bold=True)
    draw_text(draw, 1035, 460, "材料仅沉积于输出端面", 14, MUTED)
    return image


def write_assets(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    (out / "smf_array_composite_device.svg").write_text(svg, encoding="utf-8")
    build_png().save(out / "smf_array_composite_device.png")
    qa = {
        "canvas_svg": "1600x640",
        "png_size": [PNG_W, PNG_H],
        "transparent_png": True,
        "svg_has_no_embedded_raster": "<image" not in svg,
        "array": "10x10 regular fiber-endface grid",
        "selected_representative_fiber": [7, 4],
        "composite_device": "SMF body + output facet + end-face functional film",
        "material_candidates": ["Te", "CoSi", "TaP"],
        "not_full_system_architecture": True,
    }
    (out / "smf_array_composite_device_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    write_assets(args.output_dir)
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
