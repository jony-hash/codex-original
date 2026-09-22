"""Create a transparent, editable annotation layer for the confusion-matrix slide."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "ppt_confusion_annotation_v1"
DEFAULT_INPUT = Path(r"C:\TEMP\codex-clipboard-a1a35f6a-1132-44fc-9f10-b2cabae39554.png")

W, H = 1600, 900
NAVY = "#223F73"
BLUE = "#4F79C8"
BLUE_GRAY = "#8094AA"
MUTED = "#607589"
BORDER = "#D5DFEA"
PALE = "#EEF3F8"
WHITE = "#FFFFFF"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_FAMILY = "Microsoft YaHei, DengXian, Noto Sans CJK SC, Arial, sans-serif"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def svg_text(x, y, text, size, color=NAVY, *, weight=400, anchor="start", opacity=1.0):
    return (
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-family="{FONT_FAMILY}" '
        f'font-size="{size}px" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}" opacity="{fmt(opacity)}">{esc(text)}</text>'
    )


def svg_line(x1, y1, x2, y2, color=BORDER, width=1.5, *, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" stroke="{color}" stroke-width="{fmt(width)}"{extra}/>'


def svg_rect(x, y, w, h, fill="none", stroke="none", width=0, *, rx=0, opacity=1.0):
    return f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="{fmt(rx)}" fill="{fill}" stroke="{stroke}" stroke-width="{fmt(width)}" opacity="{fmt(opacity)}"/>'


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        # A quiet separator between the matrix figure and the interpretation.
        svg_line(48, 610, 1550, 610, BORDER, 1.2),
        svg_text(50, 635, "行归一化：每行表示一个真实类别的预测分布；对角线为正确识别。", 15, MUTED),
        svg_line(518, 660, 518, 812, BORDER, 1.0),
        svg_line(1005, 660, 1005, 812, BORDER, 1.0),
    ]

    columns = [
        {
            "x": 70,
            "accent": NAVY,
            "title": "主导失败：1 → 8",
            "value": "1.00（A–D均一致）",
            "detail": ["true-1 recall ≈ 0", "四组条件保持同一错误方向"],
        },
        {
            "x": 560,
            "accent": BLUE,
            "title": "主要错误：5 → 8",
            "value": "0.65–0.72",
            "detail": ["true-5 recall = 0.07–0.18", "类别 5 的主要预测目的地为 8"],
        },
        {
            "x": 1045,
            "accent": BLUE_GRAY,
            "title": "参数组差异：9 类",
            "value": "2L 0.49–0.50  →  4L 0.36–0.39",
            "detail": ["主要错误转向 4 / 8", "不要单独解释为层数因果效应"],
        },
    ]
    for col in columns:
        x = col["x"]
        parts.append(svg_rect(x, 662, 42, 4, col["accent"], "none", 0, rx=2))
        parts.append(svg_text(x, 695, col["title"], 19, NAVY, weight=600))
        parts.append(svg_text(x, 735, col["value"], 24, col["accent"], weight=600))
        parts.append(svg_text(x, 770, col["detail"][0], 15, MUTED))
        parts.append(svg_text(x, 797, col["detail"][1], 15, MUTED))

    parts.extend(
        [
            svg_line(48, 830, 1550, 830, BORDER, 1.2),
            svg_text(50, 865, "总体观察：A≈B，C≈D；Strict ReLU90 未改变主要混淆结构。", 19, NAVY, weight=600),
            svg_text(1550, 865, "描述性结果", 14, BLUE_GRAY, anchor="end"),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


class Canvas:
    def __init__(self, width: int, height: int, scale: int = 2):
        self.scale = scale
        self.image = Image.new("RGBA", (width * scale, height * scale), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.image)

    def p(self, v: float) -> int:
        return int(round(v * self.scale))

    def font(self, size: int, bold: bool = False):
        path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
        return ImageFont.truetype(str(path), self.p(size), index=0) if path.exists() else ImageFont.load_default()

    def text(self, x, y, value, size, color=NAVY, *, bold=False, anchor="la"):
        self.draw.text((self.p(x), self.p(y)), value, font=self.font(size, bold), fill=tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) + (255,), anchor=anchor)

    def line(self, points, color=BORDER, width=1.5):
        rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) + (255,)
        self.draw.line([(self.p(x), self.p(y)) for x, y in points], fill=rgb, width=max(1, self.p(width)))

    def rect(self, x, y, w, h, fill, radius=2):
        rgb = tuple(int(fill[i : i + 2], 16) for i in (1, 3, 5)) + (255,)
        self.draw.rounded_rectangle((self.p(x), self.p(y), self.p(x + w), self.p(y + h)), radius=self.p(radius), fill=rgb)


def build_png() -> Image.Image:
    c = Canvas(W, H, scale=2)
    c.line([(48, 610), (1550, 610)], BORDER, 1.2)
    c.text(50, 635, "行归一化：每行表示一个真实类别的预测分布；对角线为正确识别。", 15, MUTED)
    c.line([(518, 660), (518, 812)], BORDER, 1.0)
    c.line([(1005, 660), (1005, 812)], BORDER, 1.0)
    columns = [
        (70, NAVY, "主导失败：1 → 8", "1.00（A–D均一致）", "true-1 recall ≈ 0", "四组条件保持同一错误方向"),
        (560, BLUE, "主要错误：5 → 8", "0.65–0.72", "true-5 recall = 0.07–0.18", "类别 5 的主要预测目的地为 8"),
        (1045, BLUE_GRAY, "参数组差异：9 类", "2L 0.49–0.50  →  4L 0.36–0.39", "主要错误转向 4 / 8", "不要单独解释为层数因果效应"),
    ]
    for x, accent, title, value, detail1, detail2 in columns:
        c.rect(x, 662, 42, 4, accent, radius=2)
        c.text(x, 695, title, 19, NAVY, bold=True)
        c.text(x, 735, value, 24, accent, bold=True)
        c.text(x, 770, detail1, 15, MUTED)
        c.text(x, 797, detail2, 15, MUTED)
    c.line([(48, 830), (1550, 830)], BORDER, 1.2)
    c.text(50, 865, "总体观察：A≈B，C≈D；Strict ReLU90 未改变主要混淆结构。", 19, NAVY, bold=True)
    c.text(1550, 865, "描述性结果", 14, BLUE_GRAY, anchor="ra")
    return c.image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    (out / "06_confusion_annotation_overlay.svg").write_text(build_svg(), encoding="utf-8")
    overlay = build_png()
    overlay.save(out / "06_confusion_annotation_overlay.png")

    if args.input.exists():
        base = Image.open(args.input).convert("RGBA")
        if base.size != (W, H):
            base = base.resize((W, H), Image.Resampling.LANCZOS)
        composite = Image.alpha_composite(base, overlay.resize((W, H), Image.Resampling.LANCZOS))
        composite.save(out / "slide06_confusion_annotation_preview.png")

    (out / "assembly_guide.md").write_text(
        """# 第06页混淆矩阵文字叠加层\n\n"
        "将 `06_confusion_annotation_overlay.svg` 放置在原混淆矩阵图上方，画布按 1600×900 对齐。\n"
        "叠加层只占用原图下方空白区域，不覆盖矩阵和 colorbar。\n\n"
        "建议汇报顺序：先说明行归一化，再讲 1→8、5→8、9 类三个模式，最后给出 A≈B、C≈D 的总体观察。\n""",
        encoding="utf-8",
    )
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()
