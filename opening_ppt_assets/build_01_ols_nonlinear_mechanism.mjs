import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url));
const require=createRequire(path.join(process.argv[2],'package.json'));
const {chromium}=require('playwright');
const sharp=require('sharp');
const C={navy:'#173F5F',blue:'#3F7FB3',soft:'#DDECF3',teal:'#5C9EA7',tealSoft:'#E6F1F2',dec:'#6878B1',decSoft:'#E8EAF5',gray:'#68757D',dark:'#34444D',pale:'#F3F6F7',orange:'#C77A3C',orangeSoft:'#F5E6D8'};
const txt=(x,y,s,cls='label',anchor='middle')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${s}</text>`;
function beam(a,b,y=281,color=C.teal){return `<g class="beam" data-arrow-span="${a},${b},${y}"><path d="M${a} ${y} H${b}" fill="none" stroke="${C.navy}" stroke-width="2.2" marker-end="url(#ons-arrow)"/><path d="M${a} ${y-12} Q${(a+b)/2} ${y-22} ${b} ${y-12} M${a} ${y+12} Q${(a+b)/2} ${y+22} ${b} ${y+12}" stroke="${color}" stroke-width="1.35" opacity=".64" fill="none"/>${[.28,.64].map(f=>`<path d="M${a+(b-a)*f} ${y-22} v44" stroke="${color}" stroke-width=".8" opacity=".48"/>`).join('')}</g>`;}
function phaseLayer(x,label){return `<g data-stage="decoder-${label.toLowerCase()}" data-object="diffractive-plane"><rect x="${x-4.5}" y="208" width="9" height="146" fill="${C.decSoft}" stroke="${C.dec}" stroke-width=".9" opacity=".6"/><path d="M${x} 208 V354" stroke="${C.dec}" stroke-width="2"/>${Array.from({length:11},(_,i)=>{const y=215+i*13;const w=4+i%3;return `<path d="M${x-w} ${y} h${w*1.5} l${w/2} 5 h-${w*1.5} Z" fill="${C.dec}" opacity=".72"/>`;}).join('')}${txt(x,391,label)}</g>`;}
function field(x,after){
 const amp=['#E6F1F2','#DDECF3','#BCD5E4','#8CB8CE','#5C9EA7','#3F7FB3'];
 const cyc=['#A7BECD','#B1B8D0','#C4B5CA','#C7B9C4','#B8C6BF','#ABC4C4'];
 // Decorative, symbolic tiles only: no model field, response law, values or scales.
 const rows=[[0,1,2,1,0,0],[1,3,4,2,1,0],[2,4,5,3,2,1],[1,3,4,2,3,2],[0,1,2,3,4,2],[0,0,1,2,2,1]];
 const tile=(left,colors,phase)=>`<g data-representation="schematic-${phase?'phase':'amplitude'}">${rows.flatMap((r,i)=>r.map((v,j)=>`<rect x="${left+j*13}" y="${238+i*13}" width="13.2" height="13.2" fill="${colors[phase?(i+j)%6:v]}"/>`)).join('')}<rect x="${left}" y="238" width="78" height="78" fill="none" stroke="${C.gray}" stroke-width="1"/></g>`;
 return `<g data-stage="${after?'post-field':'pre-field'}" data-object="2d-complex-field"><path d="M${x-92} 218 H${x+92} L${x+99} 336 H${x-85} Z" fill="${C.pale}" fill-opacity=".94" stroke="${C.gray}" stroke-width="1.2"/>${tile(x-83,amp,false)}${tile(x+5,cyc,true)}${txt(x-44,360,after?'|E′|':'|E|','math small')}${txt(x+44,360,after?'φ′':'φ','math small')}${txt(x,181,after?'E′<tspan baseline-shift="sub" font-size="17">OLS</tspan>(x,y)':'E<tspan baseline-shift="sub" font-size="17">OLS</tspan>(x,y)','math')}</g>`;
}
function make(clean){
let body=txt(48,52,'目标机制 · 示意 / schematic','status','start');
if(!clean) body+=txt(487,110,'E<tspan baseline-shift="sub" font-size="17">OLS</tspan>(x,y) = ∑<tspan baseline-shift="sub" font-size="15">j</tspan> z<tspan baseline-shift="sub" font-size="15">j</tspan> v<tspan baseline-shift="sub" font-size="15">j</tspan>(x,y)','math');
body+=beam(161,223)+beam(391,437)+beam(641,727)+beam(747,834)+beam(1040,1138)+beam(1150,1250,281,C.dec)+beam(1262,1372,281,C.dec);
body+=`<g data-stage="smf-array" data-array-shape="10,10"><rect x="56" y="199" width="105" height="164" rx="6" fill="${C.tealSoft}" stroke="${C.teal}" stroke-width="1.8"/>${Array.from({length:100},(_,i)=>`<circle data-core="true" cx="${69+i%10*8.75}" cy="${214+Math.floor(i/10)*14.6}" r="3.1" fill="${C.teal}" opacity=".82"/>`).join('')}${txt(108,391,'10×10 SMF-OLS')}${!clean?txt(108,425,'100 complex coefficients','small'):''}${txt(108,171,'z₁ … z₁₀₀','math small')}${!clean?txt(108,463,'幅值 + 相位','small'):''}</g>`;
body+=`<g data-stage="coherent-reemission" data-mode-status="fundamental-mode-icon-not-mode-measurement">${[235,281,327].map((y,i)=>`<path d="M223 ${y} Q303 ${y-20+i*5} 391 ${281+(i-1)*19}" fill="none" stroke="${C.teal}" stroke-width="1.2" opacity=".5"/><ellipse cx="${252+i*34}" cy="${y}" rx="28" ry="17" fill="url(#mode-gaussian)"/>`).join('')}${txt(310,391,clean?'相干重发':'coherent re-emission','small')}${!clean?txt(310,463,'LP01 / Gaussian 基模示意','small'):''}</g>`;
body+=field(540,false);
body+=`<g data-stage="fnl-film" data-object="physical-nonlinearity-target"><rect x="729" y="198" width="14" height="166" fill="${C.orangeSoft}" stroke="${C.orange}" stroke-width="2"/><path d="M733 203 v156 M739 203 v156" stroke="${C.orange}" stroke-width="1.1"/><text x="736" y="180" class="math" text-anchor="middle" style="fill:${C.orange}">F<tspan baseline-shift="sub" font-size="18">NL</tspan></text>${txt(736,391,'物理非线性','accent')}${!clean?txt(736,433,'幅值 / 相位响应待标定','small'):''}</g>`;
body+=field(939,true)+phaseLayer(1144,'D1')+phaseLayer(1256,'D2');
if(!clean) body+=txt(1200,181,'自由空间传播','small')+txt(1200,425,'2L Decoder','small');
body+=`<g data-stage="detector-image" data-symbolic-output="true"><rect x="1379" y="233" width="101" height="96" rx="4" fill="${C.pale}" stroke="${C.navy}" stroke-width="1.8"/>${Array.from({length:30},(_,i)=>`<rect x="${1391+i%6*12}" y="${244+Math.floor(i/6)*14}" width="11" height="13" fill="${C.soft}" stroke="#ffffff" stroke-width=".5"/>`).join('')}<path d="M1404 254 Q1444 235 1448 257 Q1450 272 1428 277 Q1452 276 1448 298 Q1435 317 1403 302" fill="none" stroke="${C.blue}" stroke-width="5" stroke-linecap="round"/>${txt(1429,391,'生成图像')}${!clean?txt(1429,425,'CMOS','small'):''}</g>`;
if(!clean) body+=txt(739,508,'E′ = F<tspan baseline-shift="sub" font-size="17">NL</tspan>(E)','math')+txt(939,552,'场图与数字均为示意；不代表实测响应或生成结果','note');
return `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="600" viewBox="0 0 1600 600" data-figure-mode="facet" data-scientific-status="TARGET" role="img" aria-label="OLS相干重发复光场经目标物理非线性进入两层衍射解码器的机制示意"><title>OLS 重发复光场经过中间物理非线性后的生成机制</title><desc>Conceptual physical target only. Coefficients and two-dimensional complex fields are distinct. Fields and output digit are schematic, not measurements. No material response or quality gain is asserted.</desc><defs><marker id="ons-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="${C.navy}"/></marker><radialGradient id="mode-gaussian"><stop offset="0" stop-color="${C.teal}" stop-opacity=".78"/><stop offset=".55" stop-color="${C.teal}" stop-opacity=".3"/><stop offset="1" stop-color="${C.teal}" stop-opacity="0"/></radialGradient><style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:${C.navy}}.label{font-size:22px}.small{font-size:18px;fill:${C.dark}}.math{font-family:'Cambria Math',Cambria,'Times New Roman',serif;font-size:27px}.math.small{font-size:24px}.status{font-size:18px;fill:${C.gray}}.accent{font-size:22px;fill:${C.orange}}.note{font-size:17px;fill:${C.gray}}</style></defs>${body}</svg>`;
}
const stem='01_ols_nonlinear_mechanism';
for(const clean of [false,true])fs.writeFileSync(path.join(dir,stem+(clean?'_clean':'')+'.svg'),make(clean));
const browser=await chromium.launch({executablePath:process.argv[3],headless:true});
const qaDir=path.join(dir,stem+'.qa');fs.mkdirSync(qaDir,{recursive:true});
try{
 const page=await browser.newPage({viewport:{width:1600,height:600},deviceScaleFactor:3});
 const results=[];
 for(const clean of [false,true]){
  await page.setContent(`<html><head><style>html,body{margin:0;background:transparent}svg{display:block}</style></head><body>${make(clean)}</body></html>`);
  await page.evaluate(()=>document.fonts.ready);
  const geometry=await page.evaluate(()=>{const texts=[...document.querySelectorAll('svg > text,svg g text')].map(t=>{const b=t.getBBox();return {text:t.textContent,x:b.x,y:b.y,w:b.width,h:b.height,font:getComputedStyle(t).fontFamily};});return {texts,cores:document.querySelectorAll('[data-core]').length,stages:[...document.querySelectorAll('[data-stage]')].map(g=>g.dataset.stage)};});
  const overlaps=[];
  for(let i=0;i<geometry.texts.length;i++)for(let j=i+1;j<geometry.texts.length;j++){const a=geometry.texts[i],b=geometry.texts[j];if(a.x<b.x+b.w-2&&b.x<a.x+a.w-2&&a.y<b.y+b.h-2&&b.y<a.y+a.h-2)overlaps.push([a.text,b.text]);}
  const clipped=geometry.texts.filter(t=>t.x<0||t.y<0||t.x+t.w>1600||t.y+t.h>600);
  const outfile=path.join(clean?qaDir:dir,stem+(clean?'_clean':'')+'.png');
  await page.screenshot({path:outfile,omitBackground:true});
  const meta=await sharp(outfile).metadata();
  results.push({variant:clean?'clean':'main',overlaps,clipped,geometry,png:{width:meta.width,height:meta.height,alpha:meta.hasAlpha}});
 }
 await page.setViewportSize({width:1920,height:1080});
 await page.setContent(`<html><head><style>html,body{margin:0;background:white;width:1920px;height:1080px}body{display:flex;align-items:center;justify-content:center}svg{width:1536px;height:576px}</style></head><body>${make(false)}</body></html>`);
 await page.evaluate(()=>document.fonts.ready);
 await page.screenshot({path:path.join(qaDir,'ppt_80percent_width_preview.png')});
 const pass=results.every(r=>r.overlaps.length===0&&r.clipped.length===0&&r.geometry.cores===100&&r.png.alpha);
 fs.writeFileSync(path.join(dir,stem+'.qa.json'),JSON.stringify({status:pass?'PASS':'FIX_BEFORE_DELIVERY',scientific_status:'TARGET_SCHEMATIC',results,visual_review:'PENDING',skill_validator_limitation:'Legacy facet validator demands named Te, a different stage order and no gradients. This user explicitly requests unspecified F_NL after a reconstructed field and allows soft Gaussian mode gradients; custom topology and scientific audits are required.'},null,2));
 console.log(JSON.stringify({status:pass?'PASS':'FIX_BEFORE_DELIVERY',results:results.map(r=>({variant:r.variant,overlaps:r.overlaps,clipped:r.clipped,png:r.png}))}));
}finally{await browser.close();}
