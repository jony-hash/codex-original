import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const OUTPUT = path.join(ROOT, "outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01");
const PRESENTATION_DIR = path.join(OUTPUT, "presentation");
const STAGING_DIR = path.join(PRESENTATION_DIR, ".finalizer");
const VALIDATION_DIR = path.join(OUTPUT, ".presentation-validation");
const FINAL_PPTX = path.join(PRESENTATION_DIR, "lightgen_architecture_comparison_7slides_v2.pptx");
const SKILL_DIR = process.env.CODEX_PRESENTATIONS_SKILL_DIR;
const PYTHON = process.env.CODEX_PRIMARY_RUNTIME_PYTHON;
if (!SKILL_DIR || !path.isAbsolute(SKILL_DIR)) throw new Error("Set CODEX_PRESENTATIONS_SKILL_DIR to the installed Presentations skill directory.");
if (!PYTHON || !path.isAbsolute(PYTHON)) throw new Error("Set CODEX_PRIMARY_RUNTIME_PYTHON to the bundled validation Python executable.");
const { importRuntimeModule } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/runtime_helpers.mjs")).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);
const FONT = "Microsoft YaHei";
const C = {
  ink: "#172B4D",
  muted: "#526176",
  line: "#D7E0EA",
  blue: "#2878B5",
  orange: "#E36A2E",
  paleBlue: "#EAF2F8",
  paleOrange: "#FCF0E9",
  pale: "#F5F7FA",
  white: "#FFFFFF",
  green: "#237A57",
  red: "#A74332",
  gold: "#9A6B12",
};

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

function addShape(slide, geometry, name, position, fill, lineFill = C.line, lineWidth = 1) {
  return slide.shapes.add({
    geometry,
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
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
    fontSize: options.fontSize ?? 22,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    alignment: options.align ?? "left",
    verticalAlignment: options.valign ?? "middle",
    wrap: "square",
    autoFit: "shrinkText",
    insets: options.insets ?? 0,
    lineSpacing: options.lineSpacing ?? 1.0,
  };
  return shape;
}

function addCard(slide, name, text, position, options = {}) {
  const card = addShape(
    slide,
    options.geometry ?? "roundRect",
    name,
    position,
    options.fill ?? C.pale,
    options.line ?? C.line,
    options.lineWidth ?? 1,
  );
  if (text) {
    const inset = options.inset ?? 14;
    addText(
      slide,
      `${name}-text`,
      text,
      {
        left: position.left + inset,
        top: position.top + inset,
        width: position.width - 2 * inset,
        height: position.height - 2 * inset,
      },
      {
        fontSize: options.fontSize ?? 22,
        bold: options.bold ?? false,
        color: options.color ?? C.ink,
        align: options.align ?? "left",
      },
    );
  }
  return card;
}

function addHeader(slide, title, subtitle) {
  addText(slide, "slide-title", title, { left: 64, top: 27, width: 1152, height: 54 }, { fontSize: 43, bold: true });
  addText(slide, "slide-subtitle", subtitle, { left: 66, top: 83, width: 1148, height: 34 }, { fontSize: 21, color: C.muted });
  addShape(slide, "rect", "header-rule", { left: 64, top: 126, width: 1152, height: 2 }, C.line, "none", 0);
}

function addFooter(slide, page, text) {
  addShape(slide, "rect", `footer-rule-${page}`, { left: 64, top: 660, width: 1152, height: 1 }, C.line, "none", 0);
  addText(slide, `footer-note-${page}`, text, { left: 66, top: 665, width: 1084, height: 30 }, { fontSize: 15, color: C.muted });
  addText(slide, `footer-page-${page}`, `${page} / 7`, { left: 1150, top: 665, width: 64, height: 30 }, { fontSize: 15, color: C.muted, align: "right" });
}

async function addImage(slide, imagePath, name, position, alt) {
  const bytes = await fs.readFile(imagePath);
  const blob = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  slide.images.add({ blob, contentType: "image/png", alt, fit: "contain", position, name });
}

function addArrow(slide, name, x, y, width = 36) {
  addShape(slide, "rightArrow", name, { left: x, top: y, width, height: 22 }, C.muted, "none", 0);
}

function addColumnChart(slide, name, categories, series, position, options = {}) {
  const chart = slide.charts.add("bar", {
    position,
    categories,
    series,
    barOptions: { direction: "column", grouping: "clustered", gapWidth: options.gapWidth ?? 75 },
    chartFill: C.white,
    plotAreaFill: C.white,
    hasLegend: options.legend !== false,
    legend: options.legend === false ? undefined : {
      position: "bottom",
      overlay: false,
      textStyle: { fill: C.muted, fontSize: 15 },
    },
    xAxis: {
      textStyle: { fill: C.muted, fontSize: 15 },
      line: { style: "solid", fill: C.line, width: 1 },
      majorGridlines: null,
    },
    yAxis: {
      min: options.min ?? 0,
      max: options.max,
      majorUnit: options.majorUnit,
      numberFormatCode: options.numberFormatCode,
      textStyle: { fill: C.muted, fontSize: 15 },
      line: { style: "solid", fill: C.line, width: 1 },
      majorGridlines: { style: "solid", fill: C.line, width: 1 },
    },
    dataLabels: options.dataLabels ? {
      showValue: true,
      position: "outEnd",
      numberFormatCode: options.labelFormat,
      numberFormatSourceLinked: false,
      textStyle: { fill: C.ink, fontSize: 14, bold: true },
    } : undefined,
  });
  for (const style of [chart.legend?.textStyle, chart.dataLabels?.textStyle, chart.xAxis?.textStyle, chart.yAxis?.textStyle]) {
    if (style) style.typeface = FONT;
  }
  return chart;
}

function notes(slide, text) {
  slide.speakerNotes.text = text;
}

function populateSlideOne(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "两种架构：共享编码器，扩展 Generator", "主对照固定为 small_A1000 对 large_B1000，容量变化集中在 Generator");

  const topY = 151;
  const boxes = [
    { x: 78, w: 235, title: "振幅 SLM", body: "输入网格\n200 × 200" },
    { x: 414, w: 300, title: "Encoder（共享）", body: "相位器件窗口 0.20 mm\n两端点参数完全相同" },
    { x: 816, w: 300, title: "OLS", body: "100 个端口\nidentity 读出" },
  ];
  for (let i = 0; i < boxes.length; i += 1) {
    const item = boxes[i];
    addCard(slide, `shared-stage-${i}`, `${item.title}\n${item.body}`, { left: item.x, top: topY, width: item.w, height: 93 }, { fill: C.paleBlue, fontSize: 20, align: "center" });
    if (i < boxes.length - 1) addArrow(slide, `shared-arrow-${i}`, item.x + item.w + 38, topY + 35, 34);
  }

  addCard(slide, "small-generator-card", "", { left: 72, top: 284, width: 548, height: 250 }, { fill: "#F5F9FC", line: "#B7D0E3" });
  addText(slide, "small-generator-title", "小架构 · A1000", { left: 96, top: 298, width: 500, height: 39 }, { fontSize: 27, bold: true, color: C.blue });
  addText(slide, "small-generator-detail", "Generator：2 个原始相位面", { left: 98, top: 342, width: 480, height: 31 }, { fontSize: 20, color: C.muted });
  const smallPlanes = [
    { x: 105, text: "原面 1\n100 × 100\n0.20 mm" },
    { x: 348, text: "原面 2\n100 × 100\n0.20 mm" },
  ];
  for (let i = 0; i < smallPlanes.length; i += 1) {
    const plane = smallPlanes[i];
    addCard(slide, `small-plane-${i}`, plane.text, { left: plane.x, top: 390, width: 185, height: 90 }, { fill: C.white, align: "center", fontSize: 18 });
  }
  addText(slide, "small-generator-params", "Generator 相位参数：20k", { left: 99, top: 489, width: 480, height: 28 }, { fontSize: 20, bold: true, color: C.blue });

  addCard(slide, "large-generator-card", "", { left: 660, top: 284, width: 548, height: 250 }, { fill: "#FCF7F3", line: "#E8C4B0" });
  addText(slide, "large-generator-title", "大架构 · B1000", { left: 684, top: 298, width: 500, height: 39 }, { fontSize: 27, bold: true, color: C.orange });
  addText(slide, "large-generator-detail", "原面与新增面按真实光路交替", { left: 686, top: 342, width: 490, height: 31 }, { fontSize: 20, color: C.muted });
  const largePlanes = [
    { x: 682, text: "原面 1\n100²\n0.20 mm" },
    { x: 810, text: "新增面 1\n400²\n0.80 mm" },
    { x: 938, text: "原面 2\n100²\n0.20 mm" },
    { x: 1066, text: "新增面 2\n400²\n0.80 mm" },
  ];
  for (let i = 0; i < largePlanes.length; i += 1) {
    const plane = largePlanes[i];
    addCard(slide, `large-plane-${i}`, plane.text, { left: plane.x, top: 391, width: 112, height: 90 }, { fill: C.white, align: "center", fontSize: 16 });
    if (i < largePlanes.length - 1) addArrow(slide, `large-plane-arrow-${i}`, plane.x + 112, 424, 16);
  }
  addText(slide, "large-generator-params", "新增面 z = 0.75、2.25 mm；相位参数：340k", { left: 685, top: 489, width: 500, height: 28 }, { fontSize: 19, bold: true, color: C.orange });

  addCard(slide, "comparison-budget", "主对照：同一起点、目标、300 行确认人口与更新预算\nGenerator 参数约 17 倍；两模型均在第 1000 步比较", { left: 72, top: 553, width: 1136, height: 70 }, { fill: C.pale, fontSize: 20, align: "center" });
  addFooter(slide, 1, "结构图中的器件尺寸来自固定模型配置；参数数是相位张量元素数，不代表实际芯片制造或独立泛化能力。");
  notes(slide, [
    "来源：small_A1000 与 large_B1000 检查点：outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/arm_a_step1000.pt；arm_b_step1000.pt。父报告：outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/report.json。",
    "两端点使用同一起点、目标、人口和更新预算。Encoder 相位参数逐元素相同；小 Generator 有 2×100×100 相位参数，大 Generator 由 2 个 100×100 原面和 2 个 400×400 新增面构成。新增面位置 0.75 mm 与 2.25 mm；小架构对应位置只记录自由传播参照面。",
    "此页仅解释仿真架构。参数规模按保存张量元素计数；不外推到实际器件产率或制造公差。",
  ].join("\n\n"));
}

async function populateSlideTwo(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "器件相位：Encoder 共享，差异位于 Generator", "所有相位面按 [0, 2π) 显示；每幅图标出对应物理窗口");
  await addImage(slide, path.join(OUTPUT, "figures/phase_parameters_presentation.png"), "phase-parameters", { left: 67, top: 145, width: 1146, height: 414 }, "Encoder 与小、大 Generator 的相位参数图；两行排布；相位统一为 0 到 2π，并标注实际窗口尺寸");
  addCard(slide, "phase-shared-note", "Encoder 共享；A1000 与 B1000 参数相同。大 Generator 新增 0.80 mm 窗口相位面。", { left: 78, top: 568, width: 1124, height: 74 }, { fill: C.paleBlue, fontSize: 20, align: "center" });
  addFooter(slide, 2, "参数图来自固定 A1000/B1000 检查点；传播场中的相位另见第 4 页。");
  notes(slide, "相位参数直接读取固定 small_A1000 与 large_B1000 检查点。共享 Encoder 由逐元素相等检查确认。相位映射到 [0,2π)，没有插值。物理窗口尺寸来自各层配置与 PlaneSpec。完整图件：outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/figures/phase_parameters.{png,pdf,svg}。该图描述优化后的参数相位，不等价于沿程传播光场相位或器件实测相位。");
}

async function populateSlideThree(presentation, metrics) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "主对照振幅：逐节点并排比较传播", "类别 1、7 各取确认顺序第一张；样本逐张配对，包含 OLS 100 端口复场");
  await addImage(slide, path.join(OUTPUT, "figures/main_amplitude_ols_presentation.png"), "main-amplitude-ols", { left: 69, top: 156, width: 1142, height: 373 }, "small_A1000 与 large_B1000 的 OLS、Generator关键平面和CMOS入射振幅图");
  addCard(slide, "main-result-callout", `完整 300 行确认人口的 E_D 正确数：114 → 166（+52）\n类别 8：26/30 → 18/30，显示分类收益伴随类别代价`, { left: 73, top: 536, width: 1134, height: 82 }, { fill: C.paleBlue, fontSize: 22, align: "center" });
  addFooter(slide, 3, "场图使用共同节点 P99.5 振幅色阶；OLS 依真实 100 端口顺序排为 10×10。仅中央 400×400 原生像素窗，无场插值。");
  notes(slide, [
    "复场来源：outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/fields/{small_A1000,large_B1000}/。捕获检查点为 outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/arm_a_step1000.pt 与 arm_b_step1000.pt。",
    "每个主组端点从同一 300 行 canonical TRAIN 确认人口取每类前两行，共 20 张；本图展示类别 1 与 7 的确认顺序第一张。节点振幅显示标尺为对应节点的两端联合 P99.5，记录截幅比例；保存的复场未截幅、未归一化。大窗口仅显示中心 400×400 个原生像素，不做复数插值。",
    `完整人口指标来自父任务 report.json 与 paired_metrics.csv：E_D 为 ${metrics.comparisons.main.small_A1000.e_d_correct_total}/300 对 ${metrics.comparisons.main.large_B1000.e_d_correct_total}/300；类别 8 为 26/30 对 18/30。分类读数不是人眼可读性判断。`,
  ].join("\n\n"));
}

async function populateSlideFour(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "相位差：只比较同一物理平面和有效光场", "低于共同振幅 P99.5 的 1% 区域留白；全局相位对齐只作辅助显示");
  await addImage(slide, path.join(OUTPUT, "figures/main_phase_comparison_presentation.png"), "main-phase-comparison", { left: 67, top: 151, width: 1146, height: 407 }, "类别1和7样本在z=0.75mm相位前后复场以及相位差图");
  addCard(slide, "phase-difference-explanation", "Δφ = arg(E大 · E小*)，仅在共同有效振幅区域计算。\n去整体偏移图只移除整体相位 δ，实际相位保留在图中。\n相位差描述两端不同，不表示相位正确性；z=2.25 mm 见完整图谱。", { left: 72, top: 566, width: 1136, height: 76 }, { fill: C.pale, fontSize: 18, align: "center" });
  addFooter(slide, 4, "逐点差值只对同一物理 z 平面计算；坐标按捕获时的原生网格记录。");
  notes(slide, "主组同一物理平面的相位差为 arg(E_large·conj(E_small))。差值区域同时要求两端振幅达到各节点共同 P99.5 显示标尺的 1%。辅助对齐图减去该样本有效区域的一个整体相位偏移，实际相位图未移除该偏移。相位差仅描述两个仿真复场不同，不表示相位正确性、器件校准误差或模型优劣。样本规则与第 3 页相同。逐样本偏移、有效像素数见 outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/phase_alignment.csv。");
}

async function populateSlideFive(presentation, metrics) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "成像结果：总体分类读数提高，类别 8 回落", "主组固定确认人口完整 300 行；左侧案例与右侧逐类统计分开呈现");
  await addImage(slide, path.join(OUTPUT, "figures/main_cmos_cases.png"), "main-cmos-cases", { left: 65, top: 167, width: 550, height: 442 }, "类别 0、1、7、8 的 raw CMOS 与评价器视图案例");
  addText(slide, "recall-chart-title", "E_D 逐类正确数（每类 30 行）", { left: 650, top: 159, width: 550, height: 34 }, { fontSize: 22, bold: true });
  const a = metrics.comparisons.main.small_A1000.e_d_correct_by_class;
  const b = metrics.comparisons.main.large_B1000.e_d_correct_by_class;
  addColumnChart(slide, "main-class-recall-native", Array.from({ length: 10 }, (_, index) => String(index)), [
    { name: "小架构 A1000", values: a, fill: C.blue },
    { name: "大架构 B1000", values: b, fill: C.orange },
  ], { left: 640, top: 198, width: 570, height: 400 }, { max: 30, majorUnit: 10, gapWidth: 90 });
  addText(slide, "main-imaging-callout", "raw CMOS MSE：0.058287 → 0.058463", { left: 78, top: 611, width: 1100, height: 32 }, { fontSize: 21, bold: true, color: C.ink, align: "center" });
  addFooter(slide, 5, "Raw CMOS 共用两端点联合 P99.5 色阶；评价器视图固定 [0,1]。案例用于观察，不替代完整人口统计。");
  notes(slide, [
    "主组固定检查点及父报告：outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03/。完整逐行结果：outputs/20261010_lightgen_architecture_visual_comparison_v1/runs/attempt_01/paired_metrics.csv；汇总：metric_summary.json。",
    "右侧原生可编辑柱状图消费完整确认人口的 E_D 逐类正确数，每类 30 行；总数 114/300 与 166/300。类别 8 从 26/30 降至 18/30。raw CMOS 均方误差 0.0582866→0.0584632，轻微升高。",
    "左侧案例来自每类确认顺序第一行，类别 0、1、7、8。raw CMOS 用两模型与 300 行确认人口联合 P99.5 标尺；评价器图像是固定 [0,1] 的 28×28 视图。两者分别是物理相机入射强度和冻结评价器输入变换，不能混称同一种图像。全体均为 canonical TRAIN 确认数据，不是独立泛化测试。",
  ].join("\n\n"));
}

async function populateSlideSix(presentation, metrics) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "前景目标候选：类别 2、4 的前景误差下降", "B5400 相对同任务 A5400 对照；两端点来自固定 A4700 起点并各追加 700 步");
  await addImage(slide, path.join(OUTPUT, "figures/foreground_cmos_cases.png"), "foreground-cmos-cases", { left: 66, top: 183, width: 560, height: 249 }, "类别 2 与 4 的 A5400/B5400 raw CMOS 和评价器视图");
  addText(slide, "qfg-chart-title", "类别均值 Q_fg（越低越好）", { left: 670, top: 159, width: 530, height: 34 }, { fontSize: 22, bold: true });
  const fg = metrics.foreground_effects;
  addColumnChart(slide, "foreground-qfg-native", ["类别 2", "类别 4"], [
    { name: "A5400 原目标", values: [fg.A5400.class_2_mean_q_fg, fg.A5400.class_4_mean_q_fg].map(value => Number(value.toFixed(4))), fill: C.blue },
    { name: "B5400 前景目标", values: [fg.B5400.class_2_mean_q_fg, fg.B5400.class_4_mean_q_fg].map(value => Number(value.toFixed(4))), fill: C.orange },
  ], { left: 654, top: 199, width: 550, height: 329 }, { max: 0.5, majorUnit: 0.1, gapWidth: 100, dataLabels: true, labelFormat: "0.0000" });
  addCard(slide, "foreground-qfg-result", "类别 2、4 类平衡 Q_fg 下降 15.61%\nE_D 正确数 237 → 242 / 300", { left: 80, top: 467, width: 1100, height: 112 }, { fill: C.paleOrange, fontSize: 23, bold: true, align: "center" });
  addFooter(slide, 6, "这项收益限定于固定前景目标、固定 A4700 起点与 700 步对照；不等于总体最优或采用结论。");
  notes(slide, [
    "父报告：outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01/report.json。控制端点 A5400 与候选 B5400 的父读出、检查点均在同任务目录。候选复场来自本交付 fields/large_B5400_fg/。",
    "右侧可编辑图显示类别 2 与 4 各 30 行确认人口的 Q_fg 类均值：A5400 为 0.43105886、0.42511475；B5400 为 0.36631583、0.35624131。等权类均值下降 15.606%。父报告完整人口 E_D 为 237/300 与 242/300。",
    "两端点由同一 A4700 检查点起步，各增加 700 个 Generator 更新；确认人口只用于评估。前景误差下降不构成整体最优、可读性或候选采用证据。",
  ].join("\n\n"));
}

function populateSlideSeven(presentation, metrics) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addHeader(slide, "能量目标候选：分类读数增加，但能量守线失败", "B6700 对 A6700：分类、形态和积分能量分开看，不合并成综合排名");
  addText(slide, "energy-chart-title-1", "E_D 正确数 / 300", { left: 72, top: 160, width: 345, height: 34 }, { fontSize: 22, bold: true, align: "center" });
  addText(slide, "energy-chart-title-2", "平均 Q（越低越好）", { left: 467, top: 160, width: 345, height: 34 }, { fontSize: 22, bold: true, align: "center" });
  addText(slide, "energy-chart-title-3", "CMOS 积分能量（A4700=100%）", { left: 862, top: 160, width: 345, height: 34 }, { fontSize: 21, bold: true, align: "center" });

  const energy = metrics.energy_effects;
  const energyRows = metrics.comparisons.energy;
  addColumnChart(slide, "energy-classification-native", ["A6700", "B6700"], [
    { name: "E_D", values: [energyRows.A6700.e_d_correct_total, energyRows.B6700.e_d_correct_total], fill: C.blue },
  ], { left: 72, top: 198, width: 345, height: 319 }, { max: 300, majorUnit: 60, legend: false, dataLabels: true });
  addColumnChart(slide, "energy-morphology-native", ["A6700", "B6700"], [
    { name: "Q", values: [energy.A6700.mean_q, energy.B6700.mean_q].map(value => Number(value.toFixed(4))), fill: C.orange },
  ], { left: 467, top: 198, width: 345, height: 319 }, { max: 0.45, majorUnit: 0.1, legend: false, dataLabels: true, labelFormat: "0.0000" });
  addColumnChart(slide, "energy-retention-native", ["A4700", "A6700", "B6700"], [
    { name: "相对起点能量", values: [100, Number((energy.A6700.mean_energy / energy.A4700_start_mean_energy * 100).toFixed(2)), Number((energy.B6700.mean_energy / energy.A4700_start_mean_energy * 100).toFixed(2))], fill: C.blue },
  ], { left: 862, top: 198, width: 345, height: 319 }, { min: 88, max: 103, majorUnit: 5, numberFormatCode: "0'%'", legend: false, dataLabels: true, labelFormat: "0.00'%'" });

  addCard(slide, "energy-result", "E_D：243 → 257（+14）；平均 Q 下降 5.14%\nB6700 能量为 A4700 起点的 94.71%，低于 95% 保留线\n父任务结论：MIXED_OR_INSUFFICIENT，未通过能量检查", { left: 72, top: 536, width: 1136, height: 94 }, { fill: C.paleOrange, fontSize: 21, bold: true, align: "center", color: C.red });
  addFooter(slide, 7, "能量百分比来自同一能量目标任务的保存读出；分类与 Q 是完整 300 行确认人口结果。");
  notes(slide, [
    "父报告：outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03/report.json；逐行数据见该目录 readouts 与 paired rows。候选复场来自本交付 fields/large_B6700_energy/。",
    `A6700 与 B6700 的 E_D 为 ${energyRows.A6700.e_d_correct_total}/300 与 ${energyRows.B6700.e_d_correct_total}/300；平均 Q 为 ${energy.A6700.mean_q.toFixed(6)} 与 ${energy.B6700.mean_q.toFixed(6)}。B/A CMOS 积分能量为 ${(energy.energy_ratio_B_to_A * 100).toFixed(2)}%；B 相对 A4700 起点为 ${(energy.energy_ratio_B_to_start * 100).toFixed(2)}%。图中第三项柱状图以 A4700 起点归一为 100%，父任务的 95% 能量约束未满足。`,
    "这是固定目标与确认人口下的分类、形态、能量取舍。沿用父任务 MIXED_OR_INSUFFICIENT 判定，不据本页建立跨任务排名或采用模型。",
  ].join("\n\n"));
}

async function main() {
  await fs.access(path.join(OUTPUT, "preflight.json"));
  await fs.mkdir(PRESENTATION_DIR, { recursive: true });
  await fs.mkdir(STAGING_DIR, { recursive: true });
  await fs.mkdir(VALIDATION_DIR, { recursive: true });
  try {
    await fs.access(FINAL_PPTX);
    throw new Error(`Refusing to overwrite existing presentation: ${FINAL_PPTX}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }

  const metrics = await readJson(path.join(OUTPUT, "metric_summary.json"));
  const { Presentation, PresentationFile } = await importRuntimeModule("@oai/artifact-tool");
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  populateSlideOne(presentation);
  await populateSlideTwo(presentation);
  await populateSlideThree(presentation, metrics);
  await populateSlideFour(presentation);
  await populateSlideFive(presentation, metrics);
  await populateSlideSix(presentation, metrics);
  populateSlideSeven(presentation, metrics);

  if (presentation.slides.items.length !== 7) throw new Error(`Expected 7 slides, found ${presentation.slides.items.length}`);
  const candidatePath = path.join(STAGING_DIR, "candidate.pptx");
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
  const result = await finalizePresentation({
    explicitTotalSlideCount: 7,
    requiredNativeChartOwnerSlides: [5, 6, 7],
    requiredNativeTableOwnerSlides: [],
    materializeLiteralChartWorkbooks: true,
    workspaceDir: OUTPUT,
    candidatePath,
    finalPath: FINAL_PPTX,
    pythonExecutable: PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(VALIDATION_DIR, "lightgen_architecture_comparison_7slides_v2.validation.json"),
  });
  const finalStats = await fs.stat(FINAL_PPTX);
  const report = {
    status: "PPTX_FINALIZED",
    path: path.relative(OUTPUT, FINAL_PPTX).replaceAll("\\", "/"),
    bytes: finalStats.size,
    slides: presentation.slides.items.length,
    nativeChartOwnerSlides: [5, 6, 7],
    editableTextAndDiagramShapes: true,
    rasterHeatmapsEmbeddedAsImageObjects: true,
    validationReceipt: path.relative(OUTPUT, path.join(VALIDATION_DIR, "lightgen_architecture_comparison_7slides_v2.validation.json")).replaceAll("\\", "/"),
    finalizer: result,
  };
  await fs.writeFile(path.join(OUTPUT, "presentation_build_report.json"), JSON.stringify(report, null, 2), "utf8");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
