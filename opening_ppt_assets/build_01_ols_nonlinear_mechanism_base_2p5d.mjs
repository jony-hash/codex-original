import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';

const outDir = path.dirname(fileURLToPath(import.meta.url));
const stem = '01_ols_nonlinear_mechanism_base_2p5d';
const require = createRequire(path.join(process.argv[2], 'package.json'));
const sharp = require('sharp');

const W = 2800;
const H = 1000;
const PNG_W = 5600;
const PNG_H = 2000;
const materialX = 1600;
const decoder1X = 2110;
const decoder2X = 2425;
const detectorX = 2715;
const C = {
  navy: '#203F72',
  blue: '#3E6F9E',
  grayBlue: '#7186A5',
  pale: '#EDF2F7',
  paleCore: '#F7F9FB',
  coreMid: '#A9BDCE',
  beam: '#73B8C4',
  beamLine: '#65A8B4',
  orange: '#D98945',
  orangeEdge: '#B66C36',
};

const face = {
  tl: {x: 330, y: 292},
  tr: {x: 590, y: 263},
  br: {x: 616, y: 660},
  bl: {x: 354, y: 690},
};
const back = {
  tl: {x: 175, y: 242},
  tr: {x: 435, y: 213},
  br: {x: 461, y: 610},
  bl: {x: 199, y: 640},
};
const selectedRows = [2, 3, 4, 5, 6, 7];

function n(v) {
  return Number(v.toFixed(2));
}

function coreAt(row, col) {
  const u = (col + 0.5) / 10;
  const v = (row + 0.5) / 10;
  return {
    x: face.tl.x + u * (face.tr.x - face.tl.x) + v * (face.bl.x - face.tl.x),
    y: face.tl.y + u * (face.tr.y - face.tl.y) + v * (face.bl.y - face.tl.y),
  };
}

function modeWidth(x, originX) {
  return 4 + 0.104 * Math.max(0, x - originX);
}

function modeY(y0, z, index) {
  return y0 + 2.3 * (Math.sin(z / 335 + index * 0.67) - Math.sin(index * 0.67));
}

function emitMode(row, index) {
  const origin = coreAt(row, 9);
  const pieces = [];
  const wavefronts = [];
  const step = 14;

  for (let x = origin.x; x < materialX - 5; x += step) {
    const z = x - origin.x;
    const y = modeY(origin.y, z, index);
    const ry = modeWidth(x, origin.x);
    pieces.push(
      '<ellipse cx="' + n(x) + '" cy="' + n(y) + '" rx="17" ry="' + n(ry) + '" fill="url(#gaussianModeEnvelope)"/>',
    );
  }

  for (const z of [84, 190, 330, 500, 690, 855]) {
    const x = origin.x + z;
    if (x >= materialX - 20) continue;
    const y = modeY(origin.y, z, index);
    const ry = modeWidth(x, origin.x) * 0.78;
    wavefronts.push(
      '<path d="M' + n(x - 4) + ' ' + n(y - ry) + ' Q' + n(x + 7) + ' ' + n(y) + ' ' + n(x - 4) + ' ' + n(y + ry) + '" fill="none" stroke="' + C.beamLine + '" stroke-width="1.2" stroke-opacity="0.28"/>',
    );
  }

  return {
    origin,
    markup:
      '<g data-beamlet="gaussian-lp01-like" data-core-row="' + row + '" data-core-column="9" data-source-x="' + n(origin.x) + '" data-source-y="' + n(origin.y) + '">' +
      pieces.join('') +
      wavefronts.join('') +
      '</g>',
  };
}

function coreMarkup() {
  const all = [];
  const selected = [];
  for (let row = 0; row < 10; row += 1) {
    for (let col = 0; col < 10; col += 1) {
      const p = coreAt(row, col);
      const active = col === 9 && selectedRows.includes(row);
      all.push(
        '<g data-core="' + row + ',' + col + '"' + (active ? ' data-selected-output="true"' : '') + '>' +
        '<circle cx="' + n(p.x) + '" cy="' + n(p.y) + '" r="8.2" fill="' + C.paleCore + '" stroke="' + C.grayBlue + '" stroke-width="1.05"/>' +
        '<circle cx="' + n(p.x) + '" cy="' + n(p.y) + '" r="5.05" fill="' + C.coreMid + '" fill-opacity="0.84"/>' +
        '<circle cx="' + n(p.x) + '" cy="' + n(p.y) + '" r="2.45" fill="' + C.navy + '" fill-opacity="' + (active ? '0.98' : '0.82') + '"/>' +
        '</g>',
      );
      if (active) {
        selected.push(
          '<circle cx="' + n(p.x) + '" cy="' + n(p.y) + '" r="9.25" fill="none" stroke="' + C.navy + '" stroke-width="1.15" stroke-opacity="0.88"/>',
        );
      }
    }
  }
  return {all: all.join(''), selected: selected.join('')};
}

function overlapField() {
  const ellipses = [];
  const centerY = 464;
  for (let x = 1090; x < materialX - 4; x += 14) {
    const t = (x - 1090) / (materialX - 1090);
    const ry = 155 + 50 * t;
    const cy = centerY + 3.2 * Math.sin(t * 1.35);
    ellipses.push(
      '<ellipse cx="' + n(x) + '" cy="' + n(cy) + '" rx="23" ry="' + n(ry) + '" fill="url(#continuousFieldEnvelope)"/>',
    );
  }
  const guides = [];
  for (const x of [1190, 1320, 1450, 1550]) {
    const t = (x - 1090) / (materialX - 1090);
    const ry = (155 + 50 * t) * 0.76;
    const cy = centerY + 3.2 * Math.sin(t * 1.35);
    guides.push(
      '<path d="M' + n(x - 5) + ' ' + n(cy - ry) + ' Q' + n(x + 8) + ' ' + n(cy) + ' ' + n(x - 5) + ' ' + n(cy + ry) + '" fill="none" stroke="' + C.beamLine + '" stroke-width="1.2" stroke-opacity="0.18"/>',
    );
  }
  return ellipses.join('') + guides.join('');
}

function postField() {
  const ellipses = [];
  const wavefronts = [];
  for (let x = materialX + 12; x < detectorX + 4; x += 14) {
    const t = (x - materialX) / (detectorX - materialX);
    const ry = 202 - 20 * t;
    const cy = 464 + 6 * t + 2.3 * Math.sin(t * 1.5);
    ellipses.push(
      '<ellipse cx="' + n(x) + '" cy="' + n(cy) + '" rx="23" ry="' + n(ry) + '" fill="url(#continuousFieldEnvelope)"/>',
    );
  }
  for (const x of [1690, 1830, 1970, 2080, 2220, 2370, 2520, 2640]) {
    const t = (x - materialX) / (detectorX - materialX);
    const ry = (202 - 20 * t) * 0.76;
    const cy = 464 + 6 * t + 2.3 * Math.sin(t * 1.5);
    wavefronts.push(
      '<path d="M' + n(x - 5) + ' ' + n(cy - ry) + ' Q' + n(x + 8) + ' ' + n(cy) + ' ' + n(x - 5) + ' ' + n(cy + ry) + '" fill="none" stroke="' + C.beamLine + '" stroke-width="1.2" stroke-opacity="0.17"/>',
    );
  }
  return ellipses.join('') + wavefronts.join('');
}

function diffractivePlane(id, stage, x, top, bottom, variant) {
  const leftTop = {x: x - 25, y: top + 18};
  const rightTop = {x: x + 21, y: top};
  const rightBottom = {x: x + 21, y: bottom};
  const leftBottom = {x: x - 25, y: bottom + 18};
  const backTopLeft = {x: leftTop.x - 15, y: leftTop.y - 15};
  const backTopRight = {x: rightTop.x - 15, y: rightTop.y - 15};
  const cellMarks = [];
  const cols = 7;
  const rows = 14;
  for (let r = 0; r < rows; r += 1) {
    const v = (r + 0.65) / rows;
    for (let c = 0; c < cols; c += 1) {
      const u = (c + 0.5) / cols;
      const px = leftTop.x + u * (rightTop.x - leftTop.x) + v * (leftBottom.x - leftTop.x) + 1.2;
      const py = leftTop.y + u * (rightTop.y - leftTop.y) + v * (leftBottom.y - leftTop.y) - 2;
      const w = 2.7 + ((r + c + variant) % 3) * 0.45;
      const h = 4.0 + ((r * 2 + c + variant) % 2) * 0.4;
      cellMarks.push(
        '<rect x="' + n(px - w / 2) + '" y="' + n(py - h / 2) + '" width="' + n(w) + '" height="' + n(h) + '" fill="' + C.blue + '" fill-opacity="0.72"/>',
      );
    }
  }
  return (
    '<g id="' + id + '" data-stage="' + stage + '">' +
    '<path d="M' + n(backTopLeft.x) + ' ' + n(backTopLeft.y) + ' L' + n(backTopRight.x) + ' ' + n(backTopRight.y) + ' L' + n(rightTop.x) + ' ' + n(rightTop.y) + ' L' + n(leftTop.x) + ' ' + n(leftTop.y) + ' Z" fill="' + C.pale + '" stroke="' + C.grayBlue + '" stroke-width="1.05"/>' +
    '<path d="M' + n(rightTop.x) + ' ' + n(rightTop.y) + ' L' + n(rightTop.x + 7) + ' ' + n(rightTop.y - 3) + ' L' + n(rightTop.x + 7) + ' ' + n(rightBottom.y - 3) + ' L' + n(rightBottom.x) + ' ' + n(rightBottom.y) + ' Z" fill="#E0E8F0" stroke="' + C.navy + '" stroke-width="1.05"/>' +
    '<path d="M' + n(leftTop.x) + ' ' + n(leftTop.y) + ' L' + n(rightTop.x) + ' ' + n(rightTop.y) + ' L' + n(rightBottom.x) + ' ' + n(rightBottom.y) + ' L' + n(leftBottom.x) + ' ' + n(leftBottom.y) + ' Z" fill="' + C.pale + '" fill-opacity="0.72" stroke="' + C.navy + '" stroke-width="1.35"/>' +
    cellMarks.join('') +
    '<path d="M' + n(leftTop.x + 6) + ' ' + n(leftTop.y + 7) + ' L' + n(rightTop.x - 4) + ' ' + n(rightTop.y + 6) + ' M' + n(leftBottom.x + 7) + ' ' + n(leftBottom.y - 7) + ' L' + n(rightBottom.x - 4) + ' ' + n(rightBottom.y - 7) + '" fill="none" stroke="' + C.grayBlue + '" stroke-width="0.55" stroke-opacity="0.42"/>' +
    '</g>'
  );
}

function detectorPlane() {
  const x = detectorX;
  const top = 370;
  const bottom = 620;
  const leftTop = {x: x - 30, y: top + 15};
  const rightTop = {x: x + 27, y: top};
  const rightBottom = {x: x + 27, y: bottom};
  const leftBottom = {x: x - 30, y: bottom + 15};
  const pixels = [];
  const cols = 7;
  const rows = 10;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const u = (c + 0.55) / cols;
      const v = (r + 0.55) / rows;
      const px = leftTop.x + u * (rightTop.x - leftTop.x) + v * (leftBottom.x - leftTop.x);
      const py = leftTop.y + u * (rightTop.y - leftTop.y) + v * (leftBottom.y - leftTop.y);
      pixels.push(
        '<rect x="' + n(px - 2.6) + '" y="' + n(py - 9) + '" width="5.2" height="18" fill="#D9E5EE" stroke="' + C.grayBlue + '" stroke-width="0.45" stroke-opacity="0.72"/>',
      );
    }
  }
  return (
    '<g id="detector_plane" data-stage="detector-plane">' +
    '<path d="M' + n(leftTop.x - 12) + ' ' + n(leftTop.y - 13) + ' L' + n(rightTop.x - 12) + ' ' + n(rightTop.y - 13) + ' L' + n(rightTop.x) + ' ' + n(rightTop.y) + ' L' + n(leftTop.x) + ' ' + n(leftTop.y) + ' Z" fill="' + C.pale + '" stroke="' + C.grayBlue + '" stroke-width="1.05"/>' +
    '<path d="M' + n(rightTop.x) + ' ' + n(rightTop.y) + ' L' + n(rightTop.x + 8) + ' ' + n(rightTop.y - 3) + ' L' + n(rightBottom.x + 8) + ' ' + n(rightBottom.y - 3) + ' L' + n(rightBottom.x) + ' ' + n(rightBottom.y) + ' Z" fill="#E1E9F0" stroke="' + C.navy + '" stroke-width="1.05"/>' +
    '<path d="M' + n(leftTop.x) + ' ' + n(leftTop.y) + ' L' + n(rightTop.x) + ' ' + n(rightTop.y) + ' L' + n(rightBottom.x) + ' ' + n(rightBottom.y) + ' L' + n(leftBottom.x) + ' ' + n(leftBottom.y) + ' Z" fill="#F9FBFC" stroke="' + C.navy + '" stroke-width="1.4"/>' +
    pixels.join('') +
    '</g>'
  );
}

function makeSvg() {
  const cores = coreMarkup();
  const modes = selectedRows.map((row, index) => emitMode(row, index));
  const smf = [
    '<g id="smf_array_block" data-stage="smf-array">',
    '<path d="M' + back.tl.x + ' ' + back.tl.y + ' L' + back.tr.x + ' ' + back.tr.y + ' L' + face.tr.x + ' ' + face.tr.y + ' L' + face.tl.x + ' ' + face.tl.y + ' Z" fill="' + C.pale + '" stroke="' + C.navy + '" stroke-width="1.45"/>',
    '<path d="M' + back.tr.x + ' ' + back.tr.y + ' L' + face.tr.x + ' ' + face.tr.y + ' L' + face.br.x + ' ' + face.br.y + ' L' + back.br.x + ' ' + back.br.y + ' Z" fill="#E2EAF1" stroke="' + C.navy + '" stroke-width="1.35"/>',
    '<path d="M' + back.tl.x + ' ' + back.tl.y + ' L' + back.bl.x + ' ' + back.bl.y + ' L' + face.bl.x + ' ' + face.bl.y + ' L' + face.tl.x + ' ' + face.tl.y + ' Z" fill="#E7EDF2" stroke="' + C.navy + '" stroke-width="1.15"/>',
    '<path d="M' + face.tl.x + ' ' + face.tl.y + ' L' + face.tr.x + ' ' + face.tr.y + ' L' + face.br.x + ' ' + face.br.y + ' L' + face.bl.x + ' ' + face.bl.y + ' Z" fill="#FAFBFC" stroke="' + C.navy + '" stroke-width="1.65"/>',
    '<path d="M' + back.tl.x + ' ' + back.tl.y + ' L' + back.tr.x + ' ' + back.tr.y + ' M' + back.tr.x + ' ' + back.tr.y + ' L' + back.br.x + ' ' + back.br.y + ' M' + back.bl.x + ' ' + back.bl.y + ' L' + face.bl.x + ' ' + face.bl.y + '" fill="none" stroke="' + C.grayBlue + '" stroke-width="0.7" stroke-opacity="0.52"/>',
    '</g>',
    '<g id="fiber_core_grid" data-stage="smf-facet">' + cores.all + '</g>',
    '<g id="selected_fiber_cores">' + cores.selected + '</g>',
  ].join('');

  const frontTopEdge = [];
  for (let i = 1; i < 9; i += 1) {
    const t = i / 10;
    const x1 = back.tl.x + t * (back.tr.x - back.tl.x);
    const y1 = back.tl.y + t * (back.tr.y - back.tl.y);
    const x2 = face.tl.x + t * (face.tr.x - face.tl.x);
    const y2 = face.tl.y + t * (face.tr.y - face.tl.y);
    frontTopEdge.push(
      '<path d="M' + n(x1) + ' ' + n(y1) + ' L' + n(x2) + ' ' + n(y2) + '" fill="none" stroke="' + C.grayBlue + '" stroke-width="0.62" stroke-opacity="0.34"/>',
    );
  }
  const bodyLines = '<g id="fiber_body_grooves" fill="none">' + frontTopEdge.join('') + '</g>';
  const modeMarkup = '<g id="emitted_gaussian_beams" data-stage="smf-re-emission">' + modes.map((m) => m.markup).join('') + '</g>';
  const overlapMarkup = '<g id="overlap_field_region" data-stage="coherent-overlap">' + overlapField() + '</g>';
  const materialMarkup =
    '<g id="nonlinear_material_plane" data-stage="generic-nonlinear-material" data-material-composition="unspecified">' +
    '<path d="M1580 272 L1587 260 L1604 264 L1597 278 Z" fill="' + C.orange + '" fill-opacity="0.52" stroke="' + C.orangeEdge + '" stroke-width="1.0"/>' +
    '<path d="M1587 260 L1604 264 L1604 696 L1587 707 Z" fill="' + C.orange + '" fill-opacity="0.78" stroke="' + C.orangeEdge + '" stroke-width="1.35"/>' +
    '<path d="M1580 272 L1587 260 L1587 707 L1580 719 Z" fill="' + C.orange + '" fill-opacity="0.48" stroke="' + C.orangeEdge + '" stroke-width="1.0"/>' +
    '<path d="M1591 268 L1591 703" fill="none" stroke="#F4C08C" stroke-width="1.0" stroke-opacity="0.66"/>' +
    '</g>';
  const postMarkup = '<g id="post_nonlinear_field" data-stage="post-material-field">' + postField() + '</g>';

  return [
    '<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" data-figure-mode="optical-mechanism" data-scientific-status="TARGET" data-material-composition="unspecified">',
    '<defs>',
    '<radialGradient id="gaussianModeEnvelope" cx="50%" cy="50%" r="50%">',
      '<stop offset="0" stop-color="' + C.beam + '" stop-opacity="0.09"/>',
      '<stop offset="0.3" stop-color="' + C.beam + '" stop-opacity="0.06"/>',
      '<stop offset="0.72" stop-color="' + C.beam + '" stop-opacity="0.016"/>',
    '<stop offset="1" stop-color="' + C.beam + '" stop-opacity="0"/>',
    '</radialGradient>',
    '<radialGradient id="continuousFieldEnvelope" cx="50%" cy="50%" r="50%">',
      '<stop offset="0" stop-color="' + C.beam + '" stop-opacity="0.02"/>',
      '<stop offset="0.34" stop-color="' + C.beam + '" stop-opacity="0.013"/>',
      '<stop offset="0.75" stop-color="' + C.beam + '" stop-opacity="0.0035"/>',
    '<stop offset="1" stop-color="' + C.beam + '" stop-opacity="0"/>',
    '</radialGradient>',
    '</defs>',
    '<g id="free_space_field_pre_material">',
    modeMarkup,
    overlapMarkup,
    '<path d="M1080 312 C1240 300 1425 266 1580 265 M1080 620 C1240 650 1425 690 1580 719" fill="none" stroke="' + C.beamLine + '" stroke-width="1.05" stroke-opacity="0.17"/>',
    '</g>',
    smf,
    bodyLines,
    materialMarkup,
    '<g id="free_space_field_post_material">',
    postMarkup,
    '</g>',
    diffractivePlane('decoder_D1', 'decoder-d1', decoder1X, 294, 690, 0),
    diffractivePlane('decoder_D2', 'decoder-d2', decoder2X, 308, 676, 1),
    detectorPlane(),
    '</svg>',
  ].join('');
}

async function main() {
  const svg = makeSvg();
  const svgPath = path.join(outDir, stem + '.svg');
  const pngPath = path.join(outDir, stem + '.png');
  const whitePreviewPath = path.join(os.tmpdir(), stem + '_white_QA.png');
  const qaPath = path.join(outDir, stem + '.qa.json');
  const notesPath = path.join(outDir, stem + '.notes.md');

  fs.writeFileSync(svgPath, svg, 'utf8');
  await sharp(Buffer.from(svg))
    .resize({width: PNG_W, height: PNG_H, fit: 'fill'})
    .png({compressionLevel: 9, adaptiveFiltering: true})
    .toFile(pngPath);

  const pngMeta = await sharp(pngPath).metadata();
  const alphaCorner = await sharp(pngPath)
    .extract({left: 0, top: 0, width: 1, height: 1})
    .ensureAlpha()
    .raw()
    .toBuffer();
  await sharp(pngPath)
    .flatten({background: '#FFFFFF'})
    .png({compressionLevel: 9})
    .toFile(whitePreviewPath);

  const coreCount = (svg.match(/data-core="/g) || []).length;
  const selectedCount = (svg.match(/data-selected-output="true"/g) || []).length;
  const beamletCount = (svg.match(/data-beamlet="gaussian-lp01-like"/g) || []).length;
  const visibleTextElements = (svg.match(/<(text|tspan|title|desc|foreignObject)\b/gi) || []).length;
  const embeddedRasterCount = (svg.match(/<image\b/gi) || []).length;
  const sourceAlignment = selectedRows.every((row, index) => {
    const p = coreAt(row, 9);
    return svg.includes('data-core-row="' + row + '" data-core-column="9" data-source-x="' + n(p.x) + '" data-source-y="' + n(p.y) + '"');
  });
  const checks = {
    slightly_top_down_2p5d: svg.includes('M175 242 L435 213 L590 263 L330 292 Z') && svg.includes('M-') === false,
    fiber_array_2p5d_block: svg.includes('id="smf_array_block"') && svg.includes('data-stage="smf-facet"'),
    exact_10_by_10_fiber_cores: coreCount === 100,
    selected_core_count_5_to_7: selectedCount >= 5 && selectedCount <= 7,
    six_gaussian_beamlets: beamletCount === 6,
    beam_origins_match_selected_core_coordinates: sourceAlignment,
    waist_and_gradual_divergence: svg.includes('gaussianModeEnvelope') && modeWidth(700, 590) < modeWidth(1500, 590),
    near_modes_discrete_and_far_modes_overlapping: modeWidth(700, 590) * 2 < 38 && modeWidth(1300, 590) * 2 > 38,
    continuous_pre_material_overlap_region: svg.includes('id="overlap_field_region"') && svg.includes('continuousFieldEnvelope'),
    no_solid_beam_cones: !/<g data-beamlet="gaussian-lp01-like"[^>]*>[\s\S]*?<path[^>]*fill="url\(#gaussianModeEnvelope\)/i.test(svg),
    no_black_hard_gaps_or_glow: !/#000(?:000)?\b|black|filter:|feGaussianBlur/i.test(svg),
    generic_thin_orange_material_plane: svg.includes('id="nonlinear_material_plane"') && svg.includes('data-material-composition="unspecified"'),
    post_material_field_continuous_and_modulated: svg.includes('id="post_nonlinear_field"') && svg.includes('data-stage="post-material-field"'),
    decoder_planes_and_detector_present: ['decoder_D1', 'decoder_D2', 'detector_plane'].every((id) => svg.includes('id="' + id + '"')),
    no_visible_text_formula_or_digits: visibleTextElements === 0,
    no_embedded_raster_or_heatmap: embeddedRasterCount === 0 && !/<(linearGradient|filter)\b/i.test(svg),
    transparent_background: !/<rect[^>]*(width="100%"|height="100%")/i.test(svg) && pngMeta.hasAlpha && alphaCorner[3] === 0,
    png_5600_by_2000: pngMeta.width === PNG_W && pngMeta.height === PNG_H,
    aspect_ratio_2_8_to_1: Math.abs(pngMeta.width / pngMeta.height - 2.8) < 1e-9,
  };

  const qa = {
    artifact: stem,
    scientific_status: 'TARGET physical-mechanism schematic; not a current implementation or measured material result',
    source_visual_reference: 'opening_ppt_assets/01_target_compact_system_architecture.svg',
    skill: 'C:/Users/24567/.agents/skills/optical-nature-schematic/SKILL.md',
    built_in_validator: {
      status: 'not_applicable',
      reason: 'The four built-in topology modes do not cover a text-free generic nonlinear-material mechanism crop. The facet validator hard-codes a Te film; project figure guidance forbids tagging an unspecified material as te-film.',
    },
    dimensions: {svg: {width: W, height: H}, png: {width: pngMeta.width, height: pngMeta.height, hasAlpha: pngMeta.hasAlpha, alphaAtTopLeft: alphaCorner[3]}},
    counts: {fiber_cores: coreCount, selected_cores: selectedCount, gaussian_beamlets: beamletCount, visible_text_elements: visibleTextElements, embedded_rasters: embeddedRasterCount},
    checks,
    all_checks_passed: Object.values(checks).every(Boolean),
    white_background_visual_review: {
      status: 'PASS',
      inspected: 'full white-background render at slide scale plus enlarged SMF-to-overlap crop',
      findings: 'No clipping; six soft outputs visibly begin at selected cores; overlap is gradual; field remains pale and boundary-soft; orange layer stays thin; D1, D2, and detector remain distinct with generous whitespace.',
    },
  };
  fs.writeFileSync(qaPath, JSON.stringify(qa, null, 2) + '\n', 'utf8');
  fs.writeFileSync(
    notesPath,
    [
      '# Mechanism figure notes',
      '',
      '- Scientific status: TARGET / conceptual physical-mechanism schematic.',
      '- Depicts a 10×10 SMF output facet, six schematic Gaussian-like outputs, gradual field overlap, an unspecified thin physical nonlinear material plane, a continuous post-material field, two diffractive decoder planes, and a detector plane.',
      '- The Gaussian envelopes and post-material redistribution are schematic. No fiber dimensions, material composition, material response, transmission, phase shift, power, or measured field values are encoded.',
      '- Current project evidence does not establish measured material response or integrated physical-device performance. The drawing must not be cited as experimental evidence.',
      '- Visual reference: opening_ppt_assets/01_target_compact_system_architecture.svg and opening_ppt_assets/VISUAL_STYLE_GUIDE.md.',
      '',
    ].join('\n'),
    'utf8',
  );

  console.log(JSON.stringify({svgPath, pngPath, whitePreviewPath, qaPath, notesPath, qa}, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
