"""Build a transparent, editable 2.5D optical-mechanism schematic."""
from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


OUT = Path(__file__).resolve().parent
STEM = "01_ols_nonlinear_mechanism_corrected_2p5d"
WIDTH, HEIGHT, SCALE = 3000, 1050, 2
NAVY, MUTED, PALE = "#203F72", "#7186A5", "#EDF2F7"
ORANGE, ORANGE_EDGE = "#D98945", "#B8773F"
BEAM_RGB = (89, 164, 190)
FACET = ((155, 315), (515, 289), (540, 650), (180, 676))
BACK_OFFSET = (-36, -24)
SHEAR = -26.0 / 360.0
SOURCE_ROWS, SOURCE_COL = (0, 1, 3, 5, 7, 9), 9
MAT_X = 1600
SPLIT_START, SPLIT_END = 850, 1100
D1_X, D2_X, DETECTOR_X = 1970, 2300, 2690
Y_FIELD_0, Y_FIELD_1 = 105, 900


def mix(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def core_center(row, col):
    tl, tr, _, bl = FACET
    u, v = (col + 0.5) / 10, (row + 0.5) / 10
    upper = mix(tl, tr, u)
    lower = mix(bl, FACET[2], u)
    return mix(upper, lower, v)


SOURCES = []
for row in SOURCE_ROWS:
    cx, cy = core_center(row, SOURCE_COL)
    # The schematic beam starts at the selected core's right edge.
    SOURCES.append((cx + 3.1, cy))


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def png_data_uri(rgba):
    stream = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(stream, format="PNG", compress_level=9)
    return "data:image/png;base64," + b64encode(stream.getvalue()).decode("ascii")


def mode_field(index, x0, x1, stage):
    sx, sy = SOURCES[index]
    x0 = int(np.floor(x0))
    x1 = int(np.ceil(x1))
    px_w, px_h = (x1 - x0) * SCALE, (Y_FIELD_1 - Y_FIELD_0) * SCALE
    xs = (x0 + (np.arange(px_w, dtype=np.float32) + 0.5) / SCALE)[None, :]
    ys = (Y_FIELD_0 + (np.arange(px_h, dtype=np.float32) + 0.5) / SCALE)[:, None]

    waist = 14.0
    z_rayleigh = 500.0
    if stage in ("emitted", "overlap"):
        distance = np.maximum(xs - sx, 0.0)
        width = waist * np.sqrt(1.0 + (distance / z_rayleigh) ** 2)
        onset = smoothstep((xs - sx) / 5.0)
        offset = ys - sy
        envelope = np.exp(-(offset / width) ** 2)
        broad_alpha = 0.08 * envelope * onset
        # A second, narrower Gaussian peak keeps each selected LP01-like
        # emission legible near the facet; it fades before the common field.
        core_width = np.maximum(4.8, width * 0.34)
        core_envelope = np.exp(-(offset / core_width) ** 2)
        core_fade = 1.0 - smoothstep((distance - 260.0) / 640.0)
        core_alpha = 0.24 * core_envelope * onset * core_fade
        component = broad_alpha + core_alpha * (1.0 - broad_alpha)
        split = smoothstep((xs - SPLIT_START) / (SPLIT_END - SPLIT_START))
        emitted_alpha = component * (1.0 - split)
        if stage == "emitted":
            alpha = emitted_alpha
        else:
            # Alpha-compositing identity keeps the handoff between SVG groups
            # continuous while leaving each representative mode independently editable.
            alpha = (component - emitted_alpha) / np.maximum(1.0 - emitted_alpha, 1e-8)
    elif stage == "post":
        distance = np.maximum(xs - sx, 0.0)
        t = np.clip(distance / (DETECTOR_X - MAT_X), 0.0, 1.0)
        width = waist * np.sqrt(1.0 + (distance / z_rayleigh) ** 2) * (1.0 + 0.018 * t)
        direction = -1.0 if index % 2 else 1.0
        center = sy + direction * 2.2 * np.sin(np.pi * t)
        envelope = np.exp(-((ys - center) / width) ** 2)
        redistribution = 1.0 + 0.018 * np.cos(index * 1.31) * t
        arrival_fade = 1.0 - smoothstep((xs - (DETECTOR_X - 72.0)) / 72.0)
        alpha = 0.08 * envelope * redistribution * arrival_fade
    else:
        raise ValueError(stage)

    alpha = np.clip(alpha, 0.0, 0.24)
    rgba = np.zeros((px_h, px_w, 4), dtype=np.uint8)
    rgba[..., 0] = BEAM_RGB[0]
    rgba[..., 1] = BEAM_RGB[1]
    rgba[..., 2] = BEAM_RGB[2]
    rgba[..., 3] = np.rint(alpha * 255).astype(np.uint8)
    uri = png_data_uri(rgba)
    return (
        f'<image data-mode-index="{index + 1}" x="{x0}" y="{Y_FIELD_0}" '
        f'width="{x1 - x0}" height="{Y_FIELD_1 - Y_FIELD_0}" '
        f'href="{uri}" preserveAspectRatio="none"/>',
        {"shape": [px_h, px_w], "alpha_max": int(rgba[..., 3].max()),
         "edge_alpha_max": int(max(rgba[0, :, 3].max(), rgba[-1, :, 3].max(),
                                    rgba[:, 0, 3].max(), rgba[:, -1, 3].max()))},
    )


def field_group(group_id, stage, bounds):
    images, checks = [], []
    for index, (sx, _) in enumerate(SOURCES):
        x0, x1 = bounds(index, sx)
        uri, check = mode_field(index, x0, x1, stage)
        images.append(uri)
        checks.append(check)
    return (
        f'<g id="{group_id}" data-stage="{stage}-gaussian-field" '
        f'data-field-meaning="schematic-envelope-not-measured-or-phase-resolved">'
        + "".join(images) + "</g>",
        checks,
    )


def smf_array_block():
    tl, tr, br, bl = FACET
    dx, dy = BACK_OFFSET
    back_tl, back_tr = (tl[0] + dx, tl[1] + dy), (tr[0] + dx, tr[1] + dy)
    back_bl = (bl[0] + dx, bl[1] + dy)
    path = lambda pts: "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    out = [
        '<g id="smf_array_block" data-stage="smf-array" data-core-count="100">',
        f'<path d="{path((back_tl, back_tr, tr, tl))}" fill="#EDF2F7" stroke="{NAVY}" stroke-width="1.8"/>',
        f'<path d="{path((back_tl, tl, bl, back_bl))}" fill="#E5ECF2" stroke="{NAVY}" stroke-width="1.8"/>',
        f'<path d="{path(FACET)}" fill="#FBFCFD" stroke="{NAVY}" stroke-width="2"/>',
    ]
    for row in range(10):
        for col in range(10):
            x, y = core_center(row, col)
            selected = row in SOURCE_ROWS and col == SOURCE_COL
            cladding = "#E7EEF3" if selected else "#F4F7F9"
            core = NAVY if selected else "#9EAFBC"
            out.append(
                f'<g data-core="{row},{col}" data-selected="{str(selected).lower()}">'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9.1" fill="{cladding}" stroke="{MUTED}" stroke-width="1.1"/>'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.0" fill="#D3DFE8" stroke="#A8BAC8" stroke-width=".65"/>'
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.15" fill="{core}"/>'
                '</g>'
            )
    for index, (sx, sy) in enumerate(SOURCES):
        cx, cy = core_center(SOURCE_ROWS[index], SOURCE_COL)
        out.append(
            f'<g data-source-core="{SOURCE_ROWS[index]},{SOURCE_COL}" '
            f'data-core-x="{cx:.3f}" data-core-y="{cy:.3f}" '
            f'data-output-x="{sx:.3f}" data-output-y="{sy:.3f}"/>'
        )
    out.append("</g>")
    return "".join(out)


def diffractive_plane(group_id, x, top, bottom):
    face_w = 64
    dep_x, dep_y, thickness = 18, 12, 7
    left, right = x, x + face_w
    skew = SHEAR * face_w
    parts = [
        f'<g id="{group_id}" data-stage="diffractive-plane">',
        f'<path d="M{left-dep_x} {top-dep_y} L{right-dep_x} {top+skew-dep_y:.2f} L{right} {top+skew:.2f} L{left} {top} Z" fill="#F5F8FA" stroke="{MUTED}" stroke-width="1.5"/>',
        f'<path d="M{left-dep_x} {top-dep_y} L{left} {top} L{left} {bottom} L{left-dep_x} {bottom-dep_y} Z" fill="#E3EAF0" stroke="{NAVY}" stroke-width="1.5"/>',
        f'<path d="M{right} {top+skew:.2f} L{right+thickness} {top+SHEAR*(face_w+thickness)-2:.2f} L{right+thickness} {bottom+SHEAR*(face_w+thickness)-2:.2f} L{right} {bottom+skew:.2f} Z" fill="#DDE7EF" stroke="{NAVY}" stroke-width="1.5"/>',
        f'<g transform="matrix(1 {SHEAR:.8f} 0 1 0 {-SHEAR*left:.2f})">',
        f'<rect x="{left}" y="{top}" width="{face_w}" height="{bottom-top}" fill="#EDF2F7" fill-opacity=".95" stroke="{NAVY}" stroke-width="1.8"/>',
    ]
    cols, rows = 4, 15
    margin_x, margin_y = 6.0, 8.0
    gap_x, gap_y = 1.2, 1.6
    cell_w = (face_w - 2 * margin_x - (cols - 1) * gap_x) / cols
    cell_h = (bottom - top - 2 * margin_y - (rows - 1) * gap_y) / rows
    for row in range(rows):
        for col in range(cols):
            cx = left + margin_x + col * (cell_w + gap_x)
            cy = top + margin_y + row * (cell_h + gap_y)
            parts.append(
                f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" '
                'fill="#E1EAF1" fill-opacity=".78" stroke="#9DAFC0" stroke-width=".7"/>'
                f'<path d="M{cx+cell_w*.28:.2f} {cy+cell_h*.52:.2f} H{cx+cell_w*.72:.2f}" '
                'stroke="#7186A5" stroke-width=".75" stroke-opacity=".68"/>'
            )
    parts.append("</g></g>")
    return "".join(parts)


def nonlinear_plane():
    x, face_w, top, bottom = MAT_X, 9, 324, 666
    dep_x, dep_y, thickness = 18, 12, 5
    skew = SHEAR * face_w
    return (
        '<g id="nonlinear_plane" data-stage="generic-nonlinear-material" data-material="unspecified">'
        f'<path d="M{x-dep_x} {top-dep_y} L{x+face_w-dep_x} {top+skew-dep_y:.2f} L{x+face_w} {top+skew:.2f} L{x} {top} Z" fill="{ORANGE}" fill-opacity=".42" stroke="{ORANGE_EDGE}" stroke-width="1.4"/>'
        f'<path d="M{x-dep_x} {top-dep_y} L{x} {top} L{x} {bottom} L{x-dep_x} {bottom-dep_y} Z" fill="{ORANGE}" fill-opacity=".38" stroke="{ORANGE_EDGE}" stroke-width="1.4"/>'
        f'<path d="M{x+face_w} {top+skew:.2f} L{x+face_w+thickness} {top+SHEAR*(face_w+thickness)-2:.2f} L{x+face_w+thickness} {bottom+SHEAR*(face_w+thickness)-2:.2f} L{x+face_w} {bottom+skew:.2f} Z" fill="{ORANGE}" fill-opacity=".62" stroke="{ORANGE_EDGE}" stroke-width="1.4"/>'
        f'<g transform="matrix(1 {SHEAR:.8f} 0 1 0 {-SHEAR*x:.2f}"><rect x="{x}" y="{top}" width="{face_w}" height="{bottom-top}" fill="{ORANGE}" fill-opacity=".56" stroke="{ORANGE_EDGE}" stroke-width="1.7"/></g>'
        '</g>'
    )


def cmos_detector():
    x, face_w, top, bottom = DETECTOR_X, 114, 348, 642
    dep_x, dep_y, thickness = 18, 12, 15
    skew = SHEAR * face_w
    parts = [
        '<g id="CMOS_detector" data-stage="cmos-detector">',
        f'<path d="M{x-dep_x} {top-dep_y} L{x+face_w-dep_x} {top+skew-dep_y:.2f} L{x+face_w} {top+skew:.2f} L{x} {top} Z" fill="#E8EEF3" stroke="{MUTED}" stroke-width="1.5"/>',
        f'<path d="M{x+face_w} {top+skew:.2f} L{x+face_w+thickness} {top+SHEAR*(face_w+thickness)-3:.2f} L{x+face_w+thickness} {bottom+SHEAR*(face_w+thickness)-3:.2f} L{x+face_w} {bottom+skew:.2f} Z" fill="#D8E2EA" stroke="{NAVY}" stroke-width="1.8"/>',
        f'<g transform="matrix(1 {SHEAR:.8f} 0 1 0 {-SHEAR*x:.2f})">',
        f'<rect x="{x}" y="{top}" width="{face_w}" height="{bottom-top}" fill="#F9FBFC" stroke="{NAVY}" stroke-width="2.1"/>',
        f'<rect x="{x+9}" y="{top+10}" width="{face_w-18}" height="{bottom-top-20}" fill="#E8F0F5" stroke="#7186A5" stroke-width="1.1"/>',
    ]
    cols, rows = 8, 12
    gx, gy = x + 12, top + 13
    active_w, active_h = face_w - 24, bottom - top - 26
    cell_w, cell_h = active_w / cols, active_h / rows
    for row in range(rows):
        for col in range(cols):
            cx, cy = gx + col * cell_w, gy + row * cell_h
            parts.append(
                f'<rect x="{cx+1:.2f}" y="{cy+1:.2f}" width="{cell_w-2:.2f}" height="{cell_h-2:.2f}" '
                'fill="#D8E5ED" stroke="#A8BBC9" stroke-width=".55"/>'
            )
    parts.append("</g></g>")
    return "".join(parts)


def build():
    emitted, emitted_checks = field_group(
        "emitted_beamlets", "emitted", lambda i, sx: (sx, SPLIT_END)
    )
    overlap, overlap_checks = field_group(
        "overlap_region", "overlap", lambda i, sx: (SPLIT_START, MAT_X)
    )
    post, post_checks = field_group(
        "post_nonlinear_field", "post", lambda i, sx: (MAT_X, DETECTOR_X)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" data-figure-mode="optical-mechanism" '
        'data-scientific-status="TARGET" data-material-composition="unspecified" '
        'data-projection="shared-slight-top-down-2p5d">'
        + smf_array_block() + emitted + overlap + post + nonlinear_plane()
        + diffractive_plane("D1", D1_X, 286, 692)
        + diffractive_plane("D2", D2_X, 294, 684)
        + cmos_detector() + "</svg>"
    )
    svg_path = OUT / (STEM + ".svg")
    svg_path.write_text(svg, encoding="utf-8")

    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    ids = {node.attrib.get("id") for node in root.iter()}
    required = ["smf_array_block", "emitted_beamlets", "overlap_region",
                "nonlinear_plane", "post_nonlinear_field", "D1", "D2", "CMOS_detector"]
    cores = root.findall(".//s:g[@data-core]", ns)
    texts = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] in
             {"text", "tspan", "title", "desc", "foreignObject"}]
    source_tags = root.findall(".//s:g[@data-source-core]", ns)
    core_positions = {(node.attrib["data-core"], round(float(node.find("s:circle", ns).attrib["cx"]), 2),
                       round(float(node.find("s:circle", ns).attrib["cy"]), 2)) for node in cores}
    source_matches = all(
        (tag.attrib["data-source-core"], round(float(tag.attrib["data-core-x"]), 2),
         round(float(tag.attrib["data-core-y"]), 2)) in core_positions
        for tag in source_tags
    )
    qa = {
        "figure": STEM,
        "scientific_status": "TARGET conceptual mechanism; no material response, phase map, measurement, or performance is encoded",
        "optical_field": "individually embedded constant-color RGBA Gaussian-like envelopes; alpha is a visual schematic and is not phase-resolved",
        "projection": "all physical planes use the same shallow top-down offset direction",
        "editable_content": "SMF facet and cores, material plane, diffractive planes/cells, and CMOS face/pixels are native SVG vectors; optical envelopes are embedded RGBA images",
        "representative_mode_count": len(SOURCES),
        "checks": {
            "required_groups_present": all(name in ids for name in required),
            "exact_smf_core_count_100": len(cores) == 100,
            "six_selected_core_sources_match_facet": len(source_tags) == 6 and source_matches,
            "visible_text_or_labels_absent": len(texts) == 0,
            "transparent_svg_no_background_rectangle": not any(
                node.tag.rsplit("}", 1)[-1] == "rect" and node.attrib.get("width") == str(WIDTH)
                and node.attrib.get("height") == str(HEIGHT) for node in root.iter()),
            "phase_cell_grid_on_both_diffusers": all(len(root.findall(f'.//s:g[@id="{name}"]//s:rect', ns)) >= 61
                                                      for name in ("D1", "D2")),
            "distinct_detector_receiving_pixel_face": len(root.findall('.//s:g[@id="CMOS_detector"]//s:rect', ns)) == 98,
        },
        "embedded_field_checks": {
            "emitted_beamlets": emitted_checks,
            "overlap_region": overlap_checks,
            "post_nonlinear_field": post_checks,
        },
        "rendered_visual_review": "pending",
    }
    qa_path = OUT / (STEM + ".qa.json")
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"svg": str(svg_path), "qa": str(qa_path), "checks": qa["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    build()
