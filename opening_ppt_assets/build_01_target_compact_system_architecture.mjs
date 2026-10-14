// Native, editable SVG authoring and headless render QA for this one TARGET figure.
// Usage: node this-file.mjs <bundled-node-modules> <Chromium-or-Edge-executable>
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const [modules, executablePath] = process.argv.slice(2);
if (!modules || !executablePath) throw new Error('Provide the dependency directory and browser executable.');
const require = createRequire(path.join(path.resolve(modules), 'package.json'));
const { chromium } = require('playwright');
const sharp = require('sharp');
const folder = path.dirname(fileURLToPath(import.meta.url));
const stem = '01_target_compact_system_architecture';
const svgPath = path.join(folder, stem + '.svg');
const pngPath = path.join(folder, stem + '.png');
const qaFolder = path.join(folder, stem + '.qa');
fs.mkdirSync(qaFolder, { recursive: true });
const navy = '#203F72', blue = '#7186A5', pale = '#EDF2F7', gray = '#6B7280', orange = '#D98945';
const chunks = [];
function label(x, y, text, size = 36, kind = 'label') {
  return '<text x="' + x + '" y="' + y + '" text-anchor="middle" class="' + kind + '" font-size="' + size + '">' + text + '</text>';
}
function arrow(x1, x2, y = 250, readout = false) {
  return '<line class="flow' + (readout ? ' readout' : '') + '" x1="' + x1 + '" y1="' + y + '" x2="' + x2 + '" y2="' + y + '" marker-end="url(#ons-arrow)"/>';
}
function beam(x1, x2, half1 = 28, half2 = 34) {
  return '<polygon class="beam" points="' + x1 + ',' + (250-half1) + ' ' + x2 + ',' + (250-half2) + ' ' + x2 + ',' + (250+half2) + ' ' + x1 + ',' + (250+half1) + '"/>';
}
function plane(stage, x) {
  return '<g data-stage="' + stage + '" data-scientific-status="TARGET" data-neurons="100,100" data-pixel-pitch-m="0.000002"><use href="#ons-diffractive-plane" x="' + x + '" y="135" width="80" height="230" data-device="phase-plane"/></g>';
}
const digitPaths = {
  3: 'M12 12 C29 1 55 8 50 25 C48 35 36 38 25 40 C42 39 59 48 53 65 C47 84 18 84 9 70',
  5: 'M51 10 L18 10 L13 37 C28 29 49 33 53 48 C61 76 28 86 10 69',
  8: 'M30 40 C4 33 8 5 31 6 C58 7 59 31 30 40 C2 49 6 79 31 79 C60 80 62 49 30 40'
};
function digit(number, x, y, scale = 1) {
  return '<g transform="translate(' + x + ' ' + y + ') scale(' + scale + ')"><path d="' + digitPaths[number] + '" fill="none" stroke="' + gray + '" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/></g>';
}

chunks.push('<?xml version="1.0" encoding="UTF-8"?>');
chunks.push('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1800 600" width="270.93mm" height="90.31mm" role="img" aria-labelledby="figure-title figure-desc" data-figure-mode="architecture" data-scientific-status="TARGET">');
chunks.push('<title id="figure-title">目标紧凑 SMF–OLS 非线性光学语义生成架构</title>');
chunks.push('<desc id="figure-desc">目标架构示意，非已完成实物实验。两层衍射编码器，10×10单模光纤阵列的100个复通道，紧贴输出端面的待标定非线性复合器件，两层衍射解码器及CMOS。每个目标衍射层100×100格点、2 μm节距，活动区约200 μm×200 μm。数字为手绘矢量示意，不是实验样本。器件、膜厚、传播和阵列不按物理比例绘制。</desc>');
chunks.push('<defs><style>');
chunks.push('.label{fill:' + navy + ';font-family:Arial,"Noto Sans SC","Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei",sans-serif;font-weight:400}.secondary{fill:' + gray + ';font-family:Arial,"Noto Sans SC","Microsoft YaHei",sans-serif}.status{fill:' + gray + ';font-family:Arial,"Noto Sans SC","Microsoft YaHei",sans-serif}.material-label{fill:' + orange + ';font-family:Arial,"Noto Sans SC","Microsoft YaHei",sans-serif}.beam{fill:' + pale + ';fill-opacity:.9;stroke:none}.flow{fill:none;stroke:' + gray + ';stroke-width:5.2;stroke-dasharray:12 9;stroke-linecap:round}.readout{stroke-width:3;stroke-dasharray:7 7}.ons-stroke{stroke:' + navy + ';stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round}.ons-thin{stroke:' + blue + ';stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}');
chunks.push('</style><marker id="ons-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="3" markerHeight="3" orient="auto"><path d="M0 0L10 5L0 10Z" fill="' + gray + '"/></marker>');
// Adapted embedded primitives from optical-nature-schematic/optical-components.svg.
chunks.push('<pattern id="ons-phase-pixels" width="10" height="10" patternUnits="userSpaceOnUse"><rect x="1.2" y="1.2" width="7.6" height="7.6" fill="' + pale + '" stroke="' + blue + '" stroke-width=".45"/></pattern>');
chunks.push('<pattern id="ons-sensor-pixels" width="7" height="7" patternUnits="userSpaceOnUse"><rect x=".8" y=".8" width="5.4" height="5.4" fill="' + pale + '" stroke="' + blue + '" stroke-width=".55"/></pattern>');
chunks.push('<symbol id="ons-diffractive-plane" viewBox="0 0 80 230"><polygon points="10,25 70,8 70,208 10,225" fill="url(#ons-phase-pixels)" class="ons-stroke"/><polygon points="70,8 74,10 74,210 70,208" fill="' + blue + '" opacity=".45"/><path d="M10 25L14 28M10 225L14 227" class="ons-thin"/></symbol>');
chunks.push('<symbol id="ons-cmos" viewBox="0 0 64 120"><rect x="18" y="8" width="25" height="104" rx="1.5" fill="url(#ons-sensor-pixels)" class="ons-stroke"/><rect x="43" y="16" width="6" height="88" rx="1" fill="' + blue + '"/></symbol>');
chunks.push('</defs>');
chunks.push('<g id="proposal-tag"><text x="54" y="62" class="status" font-size="24">目标架构 / Proposed target architecture</text></g>');
chunks.push('<g id="optical-propagation" aria-label="拟定主光路">');
for (const b of [[285,360,20,35],[420,500,35,40],[565,710,34,44],[986,1150,36,44],[1215,1270,44,44],[1335,1455,42,28]]) chunks.push(beam(...b));
chunks.push('</g>');
chunks.push('<g data-stage="input" data-scientific-status="TARGET" data-symbolic-digits="true">');
for (const [i,n] of [3,5,8].entries()) {
  const x = 52 + i*81;
  chunks.push('<rect x="' + x + '" y="204" width="66" height="97" rx="3" fill="white" stroke="' + pale + '" stroke-width="2"/>');
  chunks.push(digit(n,x+4,211,.94));
}
chunks.push(label(165,434,'输入 / 语义条件',36) + '</g>');
chunks.push(arrow(286,338));
chunks.push(plane('encoder-e1',350));
chunks.push(arrow(431,478));
chunks.push(plane('encoder-e2',490));
chunks.push(label(455,434,'2L Encoder',44));
chunks.push(label(455,484,'100×100 · 2 μm',36,'secondary'));
chunks.push(label(455,530,'200 μm × 200 μm',34,'secondary'));
chunks.push(arrow(584,695));
chunks.push('<g data-stage="smf-array" data-scientific-status="TARGET" data-channels="100" data-array-shape="10,10">');
chunks.push('<g data-device="smf-volume"><polygon points="715,155 785,137 975,137 905,155" fill="' + pale + '" class="ons-thin"/><polygon points="905,155 975,137 975,327 905,345" fill="' + pale + '" class="ons-stroke"/>');
for (let r=0;r<10;r++) chunks.push('<line x1="905" y1="' + (164+r*18.1) + '" x2="975" y2="' + (146+r*18.1) + '" class="ons-thin" opacity=".6"/>');
chunks.push('<rect x="715" y="155" width="190" height="190" fill="white" class="ons-stroke"/>');
for (let r=0;r<10;r++) for (let c=0;c<10;c++) {
  const x=725+c*18.88,y=165+r*18.88;
  chunks.push('<circle cx="' + x.toFixed(2) + '" cy="' + y.toFixed(2) + '" r="7.2" fill="' + pale + '" stroke="' + blue + '" stroke-width=".85"/>');
  chunks.push('<circle class="smf-core" data-row="' + r + '" data-column="' + c + '" cx="' + x.toFixed(2) + '" cy="' + y.toFixed(2) + '" r="2.25" fill="' + navy + '"/>');
}
chunks.push('</g>' + label(820,434,'10×10 SMF–OLS',42) + label(820,476,'100 complex channels',34,'secondary') + '</g>');
chunks.push('<g data-stage="smf-facet" data-scientific-status="TARGET"><line data-device="output-facet" x1="975" y1="137" x2="975" y2="327" class="ons-stroke"/></g>');
chunks.push('<g data-stage="fnl-film" data-scientific-status="TARGET" data-material-identity="unspecified" data-response="unmeasured">');
chunks.push('<rect data-device="functional-film" x="975" y="137" width="11" height="190" rx=".8" fill="' + orange + '" fill-opacity=".88" stroke="' + orange + '" stroke-width="2"/>');
chunks.push('<path d="M980 137V128H994" fill="none" stroke="' + orange + '" stroke-width="2.2"/>');
chunks.push('<text x="994" y="57" text-anchor="middle" fill="' + orange + '" font-family="Cambria Math,Times New Roman,serif" font-size="48"><tspan font-style="italic">F</tspan><tspan baseline-shift="sub" font-style="normal" font-size="31">NL</tspan></text>');
chunks.push(label(994,122,'SMF端面非线性复合器件',34,'material-label') + '</g>');
chunks.push(arrow(1008,1127));
chunks.push(plane('decoder-d1',1140));
chunks.push(arrow(1221,1248));
chunks.push(plane('decoder-d2',1260));
chunks.push(label(1250,434,'2L Decoder',44) + label(1250,484,'100×100 · 2 μm',36,'secondary') + label(1250,530,'200 μm × 200 μm',34,'secondary'));
chunks.push(arrow(1351,1434));
chunks.push('<g data-stage="cmos" data-scientific-status="TARGET"><use href="#ons-cmos" x="1438" y="143" width="102" height="215" data-device="sensor"/>' + label(1504,434,'CMOS',42) + '</g>');
chunks.push(arrow(1535,1590,250,true));
chunks.push('<g data-stage="output-image" data-scientific-status="TARGET" data-symbolic-digit="true"><rect x="1612" y="181" width="127" height="146" rx="2" fill="white" stroke="' + blue + '" stroke-width="2"/>');
chunks.push(digit(3,1629,199,1.55) + label(1674,434,'生成图像',36) + label(1674,476,'示意',28,'secondary') + '</g>');
chunks.push('</svg>');
const svg = chunks.join('\n') + '\n';
fs.writeFileSync(svgPath,svg,'utf8');

const browser = await chromium.launch({ executablePath, headless:true, args:['--disable-gpu','--disable-background-networking','--no-first-run'] });
const page = await browser.newPage({viewport:{width:2400,height:800},deviceScaleFactor:1});
try {
  await page.setContent('<html><head><meta charset="utf-8"/><style>html,body{margin:0;background:transparent}svg{display:block;width:2400px;height:800px}</style></head><body>'+svg+'</body></html>');
  await page.evaluate(()=>document.fonts.ready);
  const geometry = await page.evaluate(()=>{
    const root=document.querySelector('svg');
    const texts=[...root.querySelectorAll('text')].map((n,i)=>{
      const b=n.getBoundingClientRect(),style=getComputedStyle(n);
      return {index:i,text:n.textContent,x:b.x,y:b.y,width:b.width,height:b.height,fontSize:Number.parseFloat(style.fontSize),className:n.getAttribute('class')??'math'};
    });
    const arrows=[...root.querySelectorAll('.flow')].map(n=>({x1:+n.getAttribute('x1')*4/3,x2:+n.getAttribute('x2')*4/3,y:+n.getAttribute('y1')*4/3}));
    const stages=[...root.querySelectorAll('[data-stage]')].map(n=>n.getAttribute('data-stage'));
    return {texts,arrows,stages,fonts:{chinese:document.fonts.check('36px "Noto Sans SC"'),latin:document.fonts.check('36px Arial')},coreCount:root.querySelectorAll('.smf-core').length,phaseCount:root.querySelectorAll('[data-device="phase-plane"]').length};
  });
  await page.screenshot({path:pngPath,omitBackground:true});
  const collisions=[];
  const intersects=(a,b,pad=0)=>a.x+a.width+pad>b.x && b.x+b.width+pad>a.x && a.y+a.height+pad>b.y && b.y+b.height+pad>a.y;
  for (let i=0;i<geometry.texts.length;i++) {
    const a=geometry.texts[i];
    if(a.x<0||a.y<0||a.x+a.width>2400||a.y+a.height>800) collisions.push({type:'overflow',text:a.text});
    for(let j=i+1;j<geometry.texts.length;j++) if(intersects(a,geometry.texts[j],3)) collisions.push({type:'text-overlap',texts:[a.text,geometry.texts[j].text]});
    for(const ar of geometry.arrows) if(ar.x1<a.x+a.width&&ar.x2>a.x&&ar.y>a.y-4&&ar.y<a.y+a.height+4) collisions.push({type:'arrow-text',text:a.text});
  }
  const expected=['input','encoder-e1','encoder-e2','smf-array','smf-facet','fnl-film','decoder-d1','decoder-d2','cmos','output-image'];
  const visibleText=geometry.texts.map(t=>t.text).join(' ');
  const scientific={
    target_architecture_explicit:svg.includes('data-scientific-status="TARGET"')&&visibleText.includes('目标架构'),
    diffractive_neurons_not_fiber_array:geometry.phaseCount===4&&svg.includes('data-neurons="100,100"'),
    ols_exactly_10_by_10:geometry.coreCount===100&&svg.includes('data-array-shape="10,10"'),
    two_micrometers_only_on_diffraction_labels:(visibleText.match(/2 μm/g)||[]).length===2,
    no_historical_relu:/relu/i.test(svg)===false,
    no_current_result_claim:/实验结果|已测得|CURRENT result/i.test(visibleText)===false,
    no_fmf_mmf:/\bFMF\b|\bMMF\b/i.test(svg)===false,
    no_named_material_performance:/\bTe\b|\bCoSi\b|\bTaP\b/.test(svg)===false,
    fnl_correct_order:JSON.stringify(geometry.stages)===JSON.stringify(expected),
    film_attached_to_output_facet:svg.includes('x1="975" y1="137"')&&svg.includes('x="975" y="137" width="11"'),
    no_independent_channel_relu:svg.match(/data-device="functional-film"/g).length===1,
    symbolic_output_not_experiment:svg.includes('data-symbolic-digit="true"')&&visibleText.includes('示意')
  };
  const metadata=await sharp(pngPath).metadata(),stats=await sharp(pngPath).stats();
  await sharp(pngPath).flatten({background:'#ffffff'}).toFile(path.join(qaFolder,'white_background_preview.png'));
  await page.setViewportSize({width:1920,height:1080});
  await page.addStyleTag({content:'html,body{background:white;width:1920px;height:1080px}body{display:flex;align-items:center;justify-content:center}svg{width:1536px;height:512px;flex:none}'});
  await page.screenshot({path:path.join(qaFolder,'ppt_80percent_width_preview.png')});
  const labels=geometry.texts.filter(t=>!['status','math'].includes(t.className)&&t.text!=='示意').map(t=>({text:t.text,final_ppt_font_pt:t.fontSize*768/1800}));
  const report={
    status:collisions.length===0&&Object.values(scientific).every(Boolean)?'PASS':'FIX_BEFORE_DELIVERY',
    figure_mode:'architecture',scientific_status:'TARGET',aspect_ratio:3,
    png:{width:metadata.width,height:metadata.height,hasAlpha:metadata.hasAlpha,alpha_min:stats.channels[3]?.min,alpha_max:stats.channels[3]?.max},
    rendered_fonts:geometry.fonts,text_geometry:geometry.texts,text_collisions:collisions,scientific_consistency_audit:scientific,
    ppt_placement:{slide_aspect:'16:9',slide_inches:[13.333333,7.5],figure_width_fraction:.8,width_inches:10.666667,rendered_slide_px:[1920,1080],effective_main_arrow_pt:5.2*768/1800,labels},
    legacy_skill_validator:{status:'NOT_APPLICABLE_AS_TOPOLOGY_GATE',reason:'Its fixed architecture requires an SLM, named Te film, 1.5 mm text and aspect <=2.8. The explicitly requested target omits those labels/elements and uses F_NL, ratio 3. The exact requested topology is audited above; the skill files were not modified.'},
    visual_review:{status:'PENDING',review_files:['white_background_preview.png','ppt_80percent_width_preview.png']},
    source_data_note:'Hand-drawn vector digit icons; no MNIST, model sampling, measured material curve or invented result.'
  };
  if(metadata.width<2000||!metadata.hasAlpha||stats.channels[3]?.min!==0||!geometry.fonts.chinese||!geometry.fonts.latin) report.status='FIX_BEFORE_DELIVERY';
  fs.writeFileSync(path.join(folder,stem+'.qa.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({status:report.status,svg:svgPath,png:pngPath,text_collisions:collisions,scientific_checks:scientific,font_check:geometry.fonts,png_dimensions:[metadata.width,metadata.height],minimum_main_label_pt:Math.min(...labels.map(x=>x.final_ppt_font_pt))}));
  if(report.status!=='PASS') process.exitCode=1;
} finally {await browser.close();}
