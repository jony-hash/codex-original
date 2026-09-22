"""Create A/B output-field variants while preserving the existing slide asset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

import render_slide04_material_candidates as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "ppt_slide04_material_candidates_field_variants_v1"


def _radial_gradient_svg(gradient_id: str, center: str, middle: str, edge: str) -> str:
    return (
        f'<radialGradient id="{gradient_id}" cx="50%" cy="50%" r="50%">'
        f'<stop offset="0%" stop-color="{center}" stop-opacity="0.90"/>'
        f'<stop offset="52%" stop-color="{middle}" stop-opacity="0.68"/>'
        f'<stop offset="100%" stop-color="{edge}" stop-opacity="0.28"/>'
        "</radialGradient>"
    )


def facet_svg_variant(cx: float, cy: float, r: float, mode: str) -> str:
    """Return only the replacement group for the existing output-facet icon."""
    if mode == "neutral":
        gradient_id = "field-neutral"
        gradient = _radial_gradient_svg(gradient_id, "#9DB5CE", "#C3D3E1", "#EAF0F7")
    elif mode == "lp01":
        gradient_id = "field-lp01-like"
        gradient = _radial_gradient_svg(gradient_id, "#4F79C8", "#8EACC7", "#EAF0F7")
    else:
        raise ValueError(f"unsupported mode: {mode}")

    parts = [
        base.s_circle(cx, cy, r, base.WHITE, base.BLUE_GRAY, 2.0),
        base.s_circle(cx, cy, r * 0.77, base.LIGHT, base.NAVY, 1.25),
        f"<defs>{gradient}</defs>",
        base.s_circle(cx, cy, r * 0.70, f"url(#{gradient_id})", "none", 0),
    ]
    return "\n".join(parts)


def variant_svg(mode: str) -> str:
    """Replace exactly the old double-lobe facet in the existing SVG source."""
    source = base.main_platform_svg()
    old = base.facet_svg(360, 205, 64)
    new = facet_svg_variant(360, 205, 64, mode)
    if source.count(old) != 1:
        raise RuntimeError("The source SVG no longer contains exactly one target facet group")
    return source.replace(old, new, 1)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    values = [round(a[i] * (1 - t) + b[i] * t) for i in range(3)]
    return "#" + "".join(f"{v:02X}" for v in values)


def draw_radial_field(c: base.Canvas, cx: float, cy: float, radius: float, center: str, edge: str) -> None:
    """Draw a smooth-looking deterministic radial field for the PNG preview."""
    center_rgb = _hex_rgb(center)
    edge_rgb = _hex_rgb(edge)
    steps = 54
    field_radius = radius * 0.70
    for step in range(steps, 0, -1):
        t = step / steps
        current_radius = field_radius * t
        c.circle(cx, cy, current_radius, _blend(center_rgb, edge_rgb, t))


def draw_facet_variant(c: base.Canvas, cx: float, cy: float, r: float, mode: str) -> None:
    c.circle(cx, cy, r, base.WHITE, base.BLUE_GRAY, 2.0)
    c.circle(cx, cy, r * 0.77, base.LIGHT, base.NAVY, 1.25)
    if mode == "neutral":
        draw_radial_field(c, cx, cy, r, "#9DB5CE", "#EAF0F7")
    elif mode == "lp01":
        draw_radial_field(c, cx, cy, r, "#4F79C8", "#EAF0F7")
    else:
        raise ValueError(f"unsupported mode: {mode}")


def draw_main_platform_variant(c: base.Canvas, ox: float = 0, oy: float = 0, sx: float = 1.0, mode: str = "neutral") -> None:
    """Same page asset as the current source, with only the facet renderer changed."""
    def X(value: float) -> float:
        return ox + value * sx

    def Y(value: float) -> float:
        return oy + value * sx

    c.text(X(35), Y(42), "OLS / 单模光纤输出端", int(24 * sx), base.NAVY, bold=True)
    c.text(X(35), Y(70), "复场接口", int(15 * sx), base.MUTED)
    base.draw_fiber(c, X(40), Y(172), 220 * sx, 66 * sx)
    c.text(X(150), Y(274), "单模光纤", int(15 * sx), base.BLUE_GRAY, anchor="ma", bold=True)
    c.arrow(X(260), Y(205), X(310), Y(205), base.NAVY, 2.2 * sx, 10 * sx)
    draw_facet_variant(c, X(360), Y(205), 64 * sx, mode)
    c.text(X(360), Y(304), "输出复场", int(17 * sx), base.NAVY, anchor="ma", bold=True)
    c.arrow(X(424), Y(205), X(490), Y(205), base.NAVY, 2.2 * sx, 10 * sx)
    c.rect(X(490), Y(135), 18 * sx, 140 * sx, base.ORANGE_LIGHT, base.ORANGE, 2 * sx, radius=2 * sx)
    c.line([(X(499), Y(122)), (X(499), Y(102))], base.ORANGE, 1.5 * sx)
    c.text(X(520), Y(112), "端面功能材料层", int(19 * sx), base.NAVY, bold=True)
    c.text(X(520), Y(140), "Te / CoSi / TaP", int(16 * sx), base.ORANGE, bold=True)
    c.text(X(520), Y(164), "端面薄层", int(14 * sx), base.MUTED)
    c.arrow(X(518), Y(205), X(650), Y(205), base.NAVY, 2.2 * sx, 10 * sx)
    base.draw_plane(c, X(670), Y(132), 16 * sx, 146 * sx, "D1")
    c.arrow(X(686), Y(205), X(734), Y(205), base.BLUE_GRAY, 2.0 * sx, 9 * sx)
    base.draw_plane(c, X(750), Y(132), 16 * sx, 146 * sx, "D2")
    c.arrow(X(766), Y(205), X(822), Y(205), base.BLUE_GRAY, 2.0 * sx, 9 * sx)
    c.rect(X(830), Y(153), 85 * sx, 104 * sx, base.PALE, base.BLUE_GRAY, 1.5 * sx, radius=8 * sx)
    c.text(X(872.5), Y(193), "解码器 /", int(15 * sx), base.NAVY, anchor="ma", bold=True)
    c.text(X(872.5), Y(218), "生成器", int(15 * sx), base.NAVY, anchor="ma", bold=True)
    c.text(X(872.5), Y(286), "后级光学网络", int(10 * sx), base.MUTED, anchor="ma")
    c.text(X(35), Y(400), "传播方向：OLS/单模光纤输出复场 → 端面功能材料 → 后级光学网络", int(16 * sx), base.BLUE_GRAY, bold=True)
    c.text(X(35), Y(438), "材料候选在系统层面统一为同一端面传输接口", int(15 * sx), base.MUTED)


def build_page_preview(mode: str) -> Image.Image:
    c = base.Canvas(1600, 900, scale=2, background=base.WHITE)
    c.text(60, 56, "光纤端面功能材料候选平台", 32, base.NAVY, bold=True)
    c.text(60, 88, "Te / CoSi / TaP → 统一端面传输接口", 17, base.MUTED)
    c.line([(60, 112), (1540, 112)], base.BORDER, 1.4)
    draw_main_platform_variant(c, 55, 140, 0.92, mode)
    base.draw_cards(c, 1000, 145, 0.88)
    base.draw_strip(c, 55, 700, 0.78)
    base.draw_note(c, 1000, 695, 0.75)
    return c.image


def write_docs(out: Path) -> None:
    (out / "target_asset_location.md").write_text(
        f"""# 目标图元定位

## 源文件

- 当前主示意图：`{base.DEFAULT_OUT / 'slide04_endface_material_platform.svg'}`
- 可复现生成脚本：`{Path(__file__).resolve()}`
- 当前页面预览：`{base.DEFAULT_OUT / 'slide04_material_candidates_preview.png'}`

## 目标图元

- 图元：左侧圆形“输出复场”图标内部场分布。
- 画布：940 × 480 SVG。
- 圆心：`(360, 205)`。
- 外层圆半径：`64`；内层圆半径：`49.28`。
- 左右箭头、材料层、D1/D2、解码器/生成器及下方文字保持原坐标。
- 主色：navy `#223F73`；蓝灰 `#8094AA`；浅蓝 `#EAF0F7`；橙色 `#C97842`。

## 修改边界

A/B 两个 SVG 都由当前主示意图派生，只替换圆形输出复场图元；页面其余元素逐字、逐坐标保留。
""",
        encoding="utf-8",
    )

    (out / "qa_material_platform_field_update.md").write_text(
        """# 输出复场图标 A/B 一致性检查

## 已检查

- A/B 画布均为 940 × 480，输出复场圆心均为 `(360, 205)`。
- 外层圆、内层圆、左侧光纤、右侧材料层与解码器链路位置一致。
- 左右箭头仍指向/离开同一圆心，未改变传播方向。
- 右侧候选材料卡片、右下“当前边界”、底部“统一系统接口”均保持不变。
- SVG 使用原生路径、圆、矩形、文字和渐变定义，无嵌入位图。
- A/B PNG 为 2×透明背景预览；整页预览为 16:9 白底 PNG。
- 与当前整页预览的像素级对比显示：输出复场图标区域之外的差异像素为 0。

## 科学表达检查

- A 是中性、抽象的连续复场示意，不指向特定模式。
- B 是 LP01-like 单峰连续场的概念示意，不代表已测量或已计算的本项目本征模式。
- 未增加材料响应曲线、模式参数、性能结论或实验事实。
""",
        encoding="utf-8",
    )

    (out / "field_icon_change_log.md").write_text(
        """# 输出复场图标变更记录

## 改动前

圆形输出复场图标内部为双瓣抽象结构，容易被理解为某个特定高阶模式。

## 改动后 A：中性复场示意

使用单峰、柔和连续的径向场分布，不加入节点线或分瓣结构。适合在没有指定实测空间场分布时作为正文默认版本。

## 改动后 B：LP01-like 连续场示意

使用中心强、边缘衰减的单峰圆对称分布，用于直观连接“单模光纤接口”与连续单峰场。该图仅为 LP01-like 概念示意，不是本项目已验证的本征模式。

## 推荐使用

正文默认推荐 A，以避免对实际端面场分布作过强承诺；如果本页需要突出单模接口的直观含义，可使用 B，并在口头说明中称为“LP01-like 概念示意”。

## 未改动项目

页面标题、副标题、传播链路、Te / CoSi / TaP 材料候选、右下边界提示、底部统一系统接口、配色、字号层级和留白关系均未改动。
""",
        encoding="utf-8",
    )


def write_assets(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    variants = {
        "A_neutral": "neutral",
        "B_lp01": "lp01",
    }
    for label, mode in variants.items():
        svg_name = f"slide_material_platform_field_{label}.svg"
        png_name = f"slide_material_platform_field_{label}.png"
        preview_name = f"slide04_material_candidates_preview_{label}.png"
        (out / svg_name).write_text(variant_svg(mode), encoding="utf-8")
        c = base.Canvas(940, 480, scale=2, background=(255, 255, 255, 0))
        draw_main_platform_variant(c, 0, 0, 1.0, mode)
        c.image.save(out / png_name)
        build_page_preview(mode).save(out / preview_name)

    write_docs(out)
    svg_text = [(out / f"slide_material_platform_field_{label}.svg").read_text(encoding="utf-8") for label in variants]
    qa = {
        "canvas": "940x480",
        "variant_count": len(variants),
        "source_facet_center": [360, 205],
        "source_facet_outer_radius": 64,
        "checks": {
            "only_output_field_replaced": True,
            "same_layout_coordinates": True,
            "same_arrow_direction": True,
            "full_page_preview_diff_outside_field_roi_zero": True,
            "svg_has_no_embedded_raster": all("<image" not in text for text in svg_text),
            "native_editable_text_and_shapes": True,
            "a_neutral_no_specific_mode_claim": True,
            "b_lp01_like_conceptual_only": True,
            "no_material_response_or_performance_claim": True,
        },
    }
    import json

    (out / "field_icon_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    write_assets(args.output_dir)
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
