"""Build one target mechanism schematic with numerical Gaussian alpha fields."""
from pathlib import Path
from io import BytesIO
import base64
import json
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
STEM = "01_ols_nonlinear_mechanism_gaussian_2p5d"
WIDTH, HEIGHT, SCALE = 2800, 1000, 2
NAVY, MUTED, PALE, ORANGE = "#203F72", "#7186A5", "#EDF2F7", "#D98945"
ROWS = [1, 3, 4, 5, 7, 8]
MATERIAL_X = 1435
DETECTOR_X = 2570


def core(row, col):
    u, v = (col + 0.5) / 10, (row + 0.5) / 10
    return 160 + 400 * u + 20 * v, 300 - 25 * u + 380 * v


SOURCES = [core(row, 9) for row in ROWS]
WEIGHTS = [0.98, 1.02, 1.0, 0.99, 1.03, 0.97]


def gaussian_layer(post=False):
    # These dimensionless visual parameters are not calibrated optical geometry.
    x0, x1 = (1428, DETECTOR_X + 15) if post else (525, 1444)
    y0, y1 = 100, 900
    xs = (x0 + (np.arange((x1 - x0) * SCALE) + 0.5) / SCALE)[None, :]
    ys = (y0 + (np.arange((y1 - y0) * SCALE) + 0.5) / SCALE)[:, None]
    total = np.zeros((ys.shape[0], xs.shape[1]), dtype=np.float32)
    near_widths, final_widths = [], []
    for index, ((sx, sy), weight) in enumerate(zip(SOURCES, WEIGHTS)):
        if post:
            t = np.maximum(xs - MATERIAL_X, 0) / (DETECTOR_X - MATERIAL_X)
            initial_width = 7 * np.sqrt(1 + ((MATERIAL_X - sx) / 80) ** 2)
            widths = initial_width * (1 + 0.045 * t)
            centers = sy + 2.5 * np.sin(t * 1.4) * (-1 if index % 2 else 1)
            visual_weight = weight * (1 + (0.025 if index % 2 else -0.025) * t)
            envelope = np.exp(-((ys - centers) / widths) ** 2)
            total += (visual_weight * envelope).astype(np.float32)
        else:
            distance = np.maximum(xs - sx, 0)
            widths = 7 * np.sqrt(1 + (distance / 80) ** 2)
            onset = np.clip((xs - sx) / 3, 0, 1)
            onset = onset * onset * (3 - 2 * onset)
            envelope = np.exp(-((ys - sy) / widths) ** 2)
            total += (weight * onset * envelope).astype(np.float32)
            near_widths.append(float(7 * np.sqrt(1 + (40 / 80) ** 2)))
            final_widths.append(float(7 * np.sqrt(1 + ((MATERIAL_X - sx) / 80) ** 2)))
    opacity = 1 - np.exp(-0.16 * total)
    transition = np.clip((xs - 1432) / 8, 0, 1)
    transition = transition * transition * (3 - 2 * transition)
    if post:
        opacity *= transition
        opacity *= 1 - np.exp(-((x1 - xs) / 45) ** 2)
    else:
        opacity *= 1 - transition
    alpha = np.rint(opacity * 255).astype(np.uint8)
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = (137, 187, 198)
    rgba[..., 3] = alpha
    stream = BytesIO()
    Image.fromarray(rgba).save(stream, format="PNG", compress_level=9)
    uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    checks = {
        "constant_rgb_alpha_only": True,
        "noise_particle_or_texture_generation": False,
        "outer_top_bottom_alpha_max": int(max(alpha[0].max(), alpha[-1].max())),
        "all_raster_edge_alpha_max": int(max(alpha[0].max(), alpha[-1].max(), alpha[:, 0].max(), alpha[:, -1].max())),
        "alpha_max": int(alpha.max()),
        "numerically_generated": True,
    }
    return (
        f'<image x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
        f'href="{uri}" preserveAspectRatio="none"/>',
        checks,
    )


def smf():
    body = [
        f'<g id="smf_array" data-stage="smf-array">',
        f'<path d="M110 276 L510 251 L560 275 L160 300 Z" fill="{PALE}" stroke="{NAVY}" stroke-width="1.5"/>',
        f'<path d="M110 276 L130 656 L180 680 L160 300 Z" fill="#E6EDF3" stroke="{NAVY}" stroke-width="1.4"/>',
        f'<path d="M160 300 L560 275 L580 655 L180 680 Z" fill="#FAFBFC" stroke="{NAVY}" stroke-width="1.6"/>',
    ]
    for row in range(10):
        for col in range(10):
            x, y = core(row, col)
            active = row in ROWS and col == 9
            body.append(
                f'<g data-core="{row},{col}" data-selected="{str(active).lower()}">'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="#F4F7FA" stroke="{MUTED}" stroke-width="1"/>'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.1" fill="#B5C7D5"/>'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.7" fill="{NAVY}" fill-opacity="{0.95 if active else 0.75}"/>'
                '</g>'
            )
    body.append("</g>")
    return "".join(body)


def decoder(name, x, top, bottom):
    left, right = x - 24, x + 23
    parts = [
        f'<g id="{name}" data-stage="decoder-{name.lower()}">',
        f'<path d="M{left-10} {top+8} L{right-10} {top-10} L{right} {top} L{left} {top+18} Z" fill="{PALE}" stroke="{MUTED}" stroke-width="1"/>',
        f'<path d="M{right} {top} L{right+5} {top-2} L{right+5} {bottom-2} L{right} {bottom} Z" fill="#E4EBF2" stroke="{NAVY}" stroke-width="1"/>',
        f'<path d="M{left} {top+18} L{right} {top} L{right} {bottom} L{left} {bottom+18} Z" fill="{PALE}" fill-opacity=".70" stroke="{NAVY}" stroke-width="1.4"/>',
    ]
    # Sparse planar structure cues, with no dotted or data-like texture.
    for fraction in [0.25, 0.5, 0.75]:
        xx = left + fraction * (right - left)
        yy = top + 18 * (1 - fraction)
        parts.append(f'<path d="M{xx:.2f} {yy+8:.2f} V{bottom+18*(1-fraction)-8:.2f}" stroke="{MUTED}" stroke-width=".7" stroke-opacity=".38" fill="none"/>')
    for fraction in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]:
        yy = top + fraction * (bottom - top)
        parts.append(f'<path d="M{left+5} {yy+16:.2f} L{right-5} {yy+2:.2f}" stroke="{MUTED}" stroke-width=".7" stroke-opacity=".38" fill="none"/>')
    parts.append("</g>")
    return "".join(parts)


def material():
    return (
        '<g id="nonlinear_plane" data-stage="generic-nonlinear-material" data-material="unspecified">'
        f'<path d="M1426 266 L1432 257 L1440 261 L1434 269 Z" fill="{ORANGE}" fill-opacity=".52" stroke="#BC783D" stroke-width=".8"/>'
        f'<path d="M1432 257 L1440 261 L1440 708 L1432 714 Z" fill="{ORANGE}" fill-opacity=".68" stroke="#BC783D" stroke-width="1"/>'
        f'<path d="M1426 266 L1432 257 L1432 714 L1426 722 Z" fill="{ORANGE}" fill-opacity=".36" stroke="#BC783D" stroke-width=".8"/>'
        '</g>'
    )


def detector():
    return (
        '<g id="detector" data-stage="detector-plane">'
        f'<path d="M2534 361 L2588 344 L2598 354 L2544 371 Z" fill="{PALE}" stroke="{MUTED}" stroke-width="1"/>'
        f'<path d="M2598 354 L2604 352 L2604 651 L2598 653 Z" fill="#E4EBF2" stroke="{NAVY}" stroke-width="1"/>'
        f'<path d="M2544 371 L2598 354 L2598 653 L2544 670 Z" fill="#F5F8FA" stroke="{NAVY}" stroke-width="1.4"/>'
        '</g>'
    )


def build():
    pre, pre_check = gaussian_layer()
    post, post_check = gaussian_layer(post=True)
    source_tags = "".join(
        f'<g data-source-core="{row},9" data-source-x="{x:.2f}" data-source-y="{y:.2f}"/>'
        for row, (x, y) in zip(ROWS, SOURCES)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" data-figure-mode="optical-mechanism" '
        'data-scientific-status="TARGET" data-material-composition="unspecified">'
        + smf()
        + f'<g id="gaussian_field_pre_nonlinear" data-stage="gaussian-re-emission-and-overlap">{source_tags}{pre}</g>'
        + f'<g id="gaussian_field_post_nonlinear" data-stage="post-material-gaussian-field">{post}</g>'
        + material()
        + decoder("D1", 1810, 294, 690)
        + decoder("D2", 2170, 306, 678)
        + detector()
        + '</svg>'
    )
    (OUT / (STEM + ".svg")).write_text(svg, encoding="utf-8")
    report = {
        "scientific_status": "TARGET conceptual mechanism; Gaussian alpha is a schematic, not experimental or phase-resolved simulation data",
        "hybrid_authoring": "editable SVG devices plus embedded, numerically generated constant-color RGBA fields",
        "field_generation": "sum of smooth Gaussian-like envelopes; no polygons, particles, random noise, or wavefront grids",
        "counts": {"cores": 100, "representative_modes": len(ROWS), "pre_wavefronts": 0, "post_wavefronts": 0},
        "pre_field": pre_check,
        "post_field": post_check,
        "required_groups": ["smf_array", "nonlinear_plane", "D1", "D2", "detector", "gaussian_field_pre_nonlinear", "gaussian_field_post_nonlinear"],
        "legacy_skill_validator": "not applicable: user explicitly permits numerical raster fields and generic material; no false Te tag",
        "render_and_visual_review": "pending",
    }
    (OUT / (STEM + ".qa.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"svg": str(OUT / (STEM + ".svg")), "cores": 100, "modes": len(ROWS), "pre_field": pre_check, "post_field": post_check}))


if __name__ == "__main__":
    build()
