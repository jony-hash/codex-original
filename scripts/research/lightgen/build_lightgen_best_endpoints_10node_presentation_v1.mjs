import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const OUTPUT = path.join(ROOT, "outputs/20261010_lightgen_best_endpoints_10node_visualization_v1/runs/attempt_01");
const PRESENTATION_DIR = path.join(OUTPUT, "presentation");
const STAGING_DIR = path.join(PRESENTATION_DIR, ".build");
const VALIDATION_DIR = path.join(OUTPUT, ".presentation-validation");
const FINAL_PPTX = path.join(PRESENTATION_DIR, "lightgen_best_endpoints_10node_8slides_v1.pptx");
const SKILL_DIR = process.env.CODEX_PRESENTATIONS_SKILL_DIR;
const PYTHON = process.env.CODEX_PRIMARY_RUNTIME_PYTHON;
if (!SKILL_DIR || !path.isAbsolute(SKILL_DIR)) throw new Error("Set CODEX_PRESENTATIONS_SKILL_DIR to the installed Presentations skill directory.");
if (!PYTHON || !path.isAbsolute(PYTHON)) throw new Error("Set CODEX_PRIMARY_RUNTIME_PYTHON to the bundled validation Python executable.");
const { importRuntimeModule } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/runtime_helpers.mjs")).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const FONT = "Microsoft YaHei";
const C = { ink: "#172B4D", muted: "#526176", line: "#D7E0EA", blue: "#2878B5", orange: "#E36A2E", white: "#FFFFFF", pale: "#F5F7FA" };

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

function addText(slide, name, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 18,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    alignment: options.align ?? "left",
    verticalAlignment: options.valign ?? "middle",
    wrap: "square",
    autoFit: "shrinkText",
    insets: options.insets ?? 0,
    lineSpacing: 1.0,
  };
  return shape;
}

function addRect(slide, name, position, fill) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position,
    fill,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

async function addImage(slide, imagePath, name, alt, position) {
  const bytes = await fs.readFile(imagePath);
  const blob = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  slide.images.add({ blob, contentType: "image/png", alt, fit: "contain", position, name });
}

function addHeader(slide, item, page) {
  addText(slide, `title-${page}`, item.title, { left: 55, top: 22, width: 1170, height: 52 }, { fontSize: 31, bold: true });
  const kindLabel = item.kind === "amplitude" ? "振幅 |E|" : "相位 arg(E) · [−π, π)";
  addText(slide, `subtitle-${page}`, `${kindLabel}　|　A1000 ↔ B5400 前景候选　|　同一来源行 ${item.source_row_index}　|　节点 ${item.node_start}–${item.node_end}`, { left: 58, top: 76, width: 1164, height: 35 }, { fontSize: 17, color: C.muted });
  addRect(slide, `header-rule-${page}`, { left: 56, top: 120, width: 1168, height: 2 }, C.line);
}

function addNotes(slide, item, page) {
  const source = item.panels[0];
  slide.speakerNotes.text = [
    `固定展示端点：small_A1000 与 large_B5400_fg。示例为 canonical TRAIN 中从 A1000 确认人口按保存顺序选择的类别 ${item.label} 首行，来源行 ${item.source_row_index}。`,
    `振幅使用两端点全部20张展示样本的逐节点共同 P99.5 标尺；相位范围 [-π, π)，振幅低于共同 P99.5 的1%留白。每幅复场热图为独立图片对象，坐标和物理窗口见图题及 field_coordinate_metadata.json。`,
    `B5400 在自身父任务保存人口上的读出重放已通过；本页 B5400 场为同一来源行上的冻结展示重放。该行不属于 B5400 父任务的 fit/confirmation 人口；所有样本仍来自 canonical TRAIN。页面 ${page} 展示节点 ${item.node_start}–${item.node_end}，共 ${source ? item.panels.length : 0} 个独立场图对象。`,
  ].join("\n\n");
}

async function buildSlide(presentation, item, page) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, item, page);
  const left = 86;
  const colWidth = 218;
  const gap = 18;
  const imageSize = 173;
  const colHeaderTop = 127;
  const colHeaderHeight = 80;
  const topRowY = 212;
  const bottomRowY = 430;
  const imageXOffset = Math.round((colWidth - imageSize) / 2);
  const nodes = item.panels.filter((panel) => panel.endpoint === "small_A1000").sort((a, b) => a.node_index - b.node_index);
  if (nodes.length !== 5) throw new Error(`Slide ${page} expected five nodes; found ${nodes.length}`);
  const smallLabelColor = C.blue;
  const largeLabelColor = C.orange;
  addText(slide, `small-row-${page}`, "小架构 · A1000", { left: 10, top: topRowY + 65, width: 68, height: 55 }, { fontSize: 16, bold: true, color: smallLabelColor, align: "center" });
  addText(slide, `large-row-${page}`, "大架构 · B5400", { left: 10, top: bottomRowY + 65, width: 68, height: 55 }, { fontSize: 16, bold: true, color: largeLabelColor, align: "center" });
  for (let col = 0; col < nodes.length; col += 1) {
    const nodeIndex = nodes[col].node_index;
    const x = left + col * (colWidth + gap);
    addText(slide, `node-header-${page}-${nodeIndex}`, nodes[col].node_label, { left: x, top: colHeaderTop, width: colWidth, height: colHeaderHeight }, { fontSize: 12, bold: true, align: "center" });
    const smallPanel = item.panels.find((panel) => panel.node_index === nodeIndex && panel.endpoint === "small_A1000");
    const largePanel = item.panels.find((panel) => panel.node_index === nodeIndex && panel.endpoint === "large_B5400_fg");
    for (const [panel, endpoint, y] of [[smallPanel, "small_A1000", topRowY], [largePanel, "large_B5400_fg", bottomRowY]]) {
      const panelPath = path.join(OUTPUT, panel.image);
      await addImage(slide, panelPath, `heatmap-${page}-${endpoint}-${nodeIndex}`, panel.alt, { left: x + imageXOffset, top: y, width: imageSize, height: imageSize });
    }
  }
  addRect(slide, `footer-rule-${page}`, { left: 56, top: 649, width: 1168, height: 1 }, C.line);
  const foot = item.kind === "amplitude"
    ? "逐节点共同 P99.5 色阶；图题给出数值上限。显示裁窗保留原生像素，不插值。"
    : "环形色图范围 [−π, π)；相位只在振幅达到共同 P99.5 的1%处显示。新增面显示完整 Et。";
  addText(slide, `footer-${page}`, foot, { left: 58, top: 656, width: 1090, height: 30 }, { fontSize: 14, color: C.muted });
  addText(slide, `page-${page}`, `${page} / 8`, { left: 1155, top: 656, width: 68, height: 30 }, { fontSize: 14, color: C.muted, align: "right" });
  addNotes(slide, item, page);
  return slide;
}

async function main() {
  await fs.access(path.join(OUTPUT, "preflight.json"));
  await fs.access(path.join(OUTPUT, "presentation_panels.json"));
  await fs.mkdir(PRESENTATION_DIR, { recursive: true });
  await fs.mkdir(STAGING_DIR, { recursive: true });
  await fs.mkdir(VALIDATION_DIR, { recursive: true });
  try {
    await fs.access(FINAL_PPTX);
    throw new Error(`Refusing to overwrite existing presentation: ${FINAL_PPTX}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const panelManifest = await readJson(path.join(OUTPUT, "presentation_panels.json"));
  if (panelManifest.slides.length !== 8) throw new Error(`Expected 8 slide descriptors, got ${panelManifest.slides.length}`);
  const { Presentation, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  for (let index = 0; index < panelManifest.slides.length; index += 1) await buildSlide(presentation, panelManifest.slides[index], index + 1);
  if (presentation.slides.items.length !== 8) throw new Error(`Expected 8 slides, found ${presentation.slides.items.length}`);
  const candidatePath = path.join(STAGING_DIR, "candidate.pptx");
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
  const result = await finalizePresentation({
    explicitTotalSlideCount: 8,
    requiredNativeChartOwnerSlides: [],
    requiredNativeTableOwnerSlides: [],
    workspaceDir: OUTPUT,
    candidatePath,
    finalPath: FINAL_PPTX,
    pythonExecutable: PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(VALIDATION_DIR, "lightgen_best_endpoints_10node_8slides_v1.validation.json"),
  });
  const stats = await fs.stat(FINAL_PPTX);
  const report = {
    status: "PPTX_FINALIZED",
    path: path.relative(OUTPUT, FINAL_PPTX).replaceAll("\\", "/"),
    bytes: stats.size,
    slides: presentation.slides.items.length,
    independentHeatmapPictures: 80,
    editableTextAndDiagramShapes: true,
    speakerNotes: 8,
    nativePowerPointOpenVerified: false,
    validationReceipt: path.relative(OUTPUT, path.join(VALIDATION_DIR, "lightgen_best_endpoints_10node_8slides_v1.validation.json")).replaceAll("\\", "/"),
    finalizer: result,
  };
  await fs.writeFile(path.join(OUTPUT, "presentation_build_report.json"), JSON.stringify(report, null, 2), "utf8");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
