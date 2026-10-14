import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';

const dir=path.dirname(fileURLToPath(import.meta.url));
const stem='01_ols_nonlinear_mechanism_base';
const require=createRequire(path.join(process.argv[2],'package.json'));
const sharp=require('sharp');
const {chromium}=require('playwright');
const C={navy:'#173F5F',blue:'#3F7FB3',teal:'#5C9EA7',orange:'#C77A3C',pale:'#EDF2F7',soft:'#DDECF3',grayBlue:'#7186A5',clad:'#E8EEF3'};
const emittingRows=[2,3,4,5,6,7];
const faceEdge=263,sourceX=109+9*16.25,materialX=800,materialOut=806;
const rowY=row=>196+row*22.2;
const beamWidth=z=>4*Math.sqrt(1+(z/45)**2);
const radialStops=`<stop offset="0" stop-color="${C.teal}" stop-opacity=".12"/><stop offset=".22" stop-color="${C.teal}" stop-opacity=".085"/><stop offset=".5" stop-color="${C.teal}" stop-opacity=".036"/><stop offset=".8" stop-color="${C.teal}" stop-opacity=".008"/><stop offset="1" stop-color="${C.teal}" stop-opacity="0"/>`;

function makeMode(row,index){
  const y=rowY(row),step=8,rx=12,defs=[],ellipses=[];
  for(let x=faceEdge;x<=materialX;x+=step){
    const w=beamWidth(x-faceEdge),id=`beam_${index}_${x}`;
    defs.push(`<radialGradient id="${id}" gradientUnits="userSpaceOnUse" cx="0" cy="0" r="1" gradientTransform="translate(${x} ${y}) scale(${rx} ${w})">${radialStops}</radialGradient>`);
    ellipses.push(`<ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${w}" fill="url(#${id})"/>`);
  }
  const fronts=[faceEdge+62,faceEdge+170,faceEdge+290,faceEdge+408,faceEdge+510].map(x=>{
    const w=beamWidth(x-faceEdge),sag=Math.min(2.2,(x-faceEdge)/180);
    return `<path d="M${x+sag} ${y-1.48*w} Q${x-sag} ${y} ${x+sag} ${y+1.48*w}" fill="none" stroke="${C.teal}" stroke-width=".75" stroke-opacity=".3"/>`;
  }).join('');
  return {
    defs:defs.join(''),
    group:`<g data-mode="lp01-like" data-source-core="${row},9" data-origin="${sourceX},${y}">${ellipses.join('')}<path d="M${sourceX} ${y} H${materialX}" fill="none" stroke="${C.teal}" stroke-width=".7" stroke-opacity=".1"/>${fronts}</g>`
  };
}

function drawFiberArray(){
  const coreMarkup=[];
  for(let r=0;r<10;r++)for(let c=0;c<10;c++){
    const x=109+c*16.25,y=rowY(r),active=c===9&&emittingRows.includes(r);
    coreMarkup.push(`<g data-core="${r},${c}"${active?' data-emitting-core="true"':''}><circle cx="${x}" cy="${y}" r="5.15" fill="${C.clad}" stroke="${active?C.navy:C.grayBlue}" stroke-width=".85"/><circle cx="${x}" cy="${y}" r="${active?2.65:2.05}" fill="${C.navy}" fill-opacity="${active?1:.78}"/></g>`);
  }
  return `<g id="smf_array">
    <path d="M58 188 H224 L263 168 V412 L224 432 H58 Z" fill="${C.clad}" stroke="${C.navy}" stroke-width="1.35"/>
    <path d="M58 188 L96 168 H263 L224 188 Z" fill="#E3EAF0" stroke="${C.navy}" stroke-width="1.2"/>
    <path d="M224 188 L263 168 V412 L224 432 Z" fill="#E5ECF2" stroke="${C.navy}" stroke-width="1.2"/>
    <path d="M96 168 H263 V432 H96 Z" fill="#F8FAFC" stroke="${C.navy}" stroke-width="1.6"/>
    ${coreMarkup.join('')}
    <path d="M96 168 H263 V432" fill="none" stroke="${C.navy}" stroke-width="1.7"/>
    <path d="M99 171 H260 V429" fill="none" stroke="${C.grayBlue}" stroke-width=".65" stroke-opacity=".62"/>
  </g>`;
}

function drawPlane(x,id){
  const marks=Array.from({length:15},(_,i)=>{
    const y=187+i*15.3,w=4+i%3;
    return `<path d="M${x-w} ${y} h${w*1.55} l${w*.45} 5 h-${w*1.55} Z" fill="${C.blue}" fill-opacity=".64"/>`;
  }).join('');
  return `<g id="${id}"><rect x="${x-4.5}" y="180" width="9" height="240" fill="${C.soft}" fill-opacity=".75" stroke="${C.blue}" stroke-width=".9"/><path d="M${x} 180 V420" stroke="${C.blue}" stroke-width="1.9"/>${marks}<path d="M${x-6} 184 V416 M${x+6} 184 V416" stroke="${C.blue}" stroke-width=".6" stroke-opacity=".5"/></g>`;
}

const modeRecords=emittingRows.map(makeMode);
const smf=drawFiberArray();
const emitted=`<g id="emitted_modes">${modeRecords.map(m=>m.group).join('')}</g>`;
const material=`<g id="nonlinear_material"><path d="M800 139 H806 V461 H800 Z" fill="#F5E6D8" fill-opacity=".94" stroke="${C.orange}" stroke-width="1.8"/><path d="M803 144 V456" stroke="${C.orange}" stroke-width=".9" stroke-opacity=".72"/></g>`;
const post=`<g id="post_nonlinear_field"><path d="M806 165 C906 169 1042 184 1139 193 L1139 407 C1027 423 910 429 806 427 Z" fill="${C.teal}" fill-opacity=".035"/><path d="M806 165 C906 169 1042 184 1139 193 M806 427 C910 429 1027 423 1139 407" stroke="${C.teal}" stroke-width="1.1" stroke-opacity=".65" fill="none"/>${[226,246,266,306,326,366].map((y,i)=>`<path d="M806 ${y} C910 ${y+(i%2?3:-2)} 1040 ${y+(i%2?5:-3)} 1139 ${y}" stroke="${C.teal}" stroke-width=".9" stroke-opacity="${.25+i*.015}" fill="none"/>`).join('')}<path d="M887 173 C906 226 900 362 887 426 M1005 183 C1022 244 1012 355 1005 420 M1099 190 C1111 250 1105 350 1099 412 M835 280 C927 275 1018 286 1116 278 M838 344 C932 351 1026 338 1118 347" fill="none" stroke="${C.teal}" stroke-width=".9" stroke-opacity=".36"/></g>`;
const propagation=`<g id="decoder_propagation"><path d="M1151 193 C1220 198 1250 194 1319 193 L1319 407 C1250 404 1210 410 1151 407 Z M1331 193 C1379 199 1423 202 1471 210 L1471 390 C1423 398 1379 402 1331 407 Z" fill="${C.teal}" fill-opacity=".075"/><path d="M1151 193 C1220 198 1250 194 1319 193 M1151 407 C1210 410 1250 404 1319 407 M1331 193 C1379 199 1423 202 1471 210 M1331 407 C1379 402 1423 398 1471 390" fill="none" stroke="${C.teal}" stroke-width="1.1" stroke-opacity=".65"/>${[255,300,345].map(y=>`<path d="M1151 ${y} Q1237 ${y+4} 1319 ${y} M1331 ${y} Q1400 ${y-3} 1471 ${y}" fill="none" stroke="${C.teal}" stroke-width=".8" stroke-opacity=".4"/>`).join('')}<path d="M1203 196 Q1215 300 1203 408 M1270 196 Q1280 300 1270 407 M1380 201 Q1390 300 1380 399 M1437 205 Q1445 300 1437 395" fill="none" stroke="${C.teal}" stroke-width=".85" stroke-opacity=".42"/></g>`;
const detector=`<g id="detector"><rect x="1478" y="192" width="10" height="216" fill="${C.pale}" stroke="${C.navy}" stroke-width="1.5"/><path d="M1471 192 H1478 V408 H1471 Z" fill="${C.soft}" stroke="${C.navy}" stroke-width="1.2"/>${Array.from({length:15},(_,i)=>`<path d="M1473 ${199+i*13.6} H1477" stroke="${C.blue}" stroke-width=".8" stroke-opacity=".62"/>`).join('')}</g>`;
const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="640" viewBox="0 0 1600 640"><defs>${modeRecords.map(m=>m.defs).join('')}</defs>${smf}${emitted}${material}${post}${propagation}${drawPlane(1145,'decoder_layer_1')}${drawPlane(1325,'decoder_layer_2')}${detector}</svg>`;
fs.writeFileSync(path.join(dir,stem+'.svg'),svg);

const browser=await chromium.launch({executablePath:process.argv[3],headless:true});
const qaDir=path.join(dir,stem+'.qa');
fs.mkdirSync(qaDir,{recursive:true});
try{
  const page=await browser.newPage({viewport:{width:1600,height:640},deviceScaleFactor:3});
  await page.setContent(`<html><head><style>html,body{margin:0;background:transparent}svg{display:block}</style></head><body>${svg}</body></html>`);
  await page.screenshot({path:path.join(dir,stem+'.png'),omitBackground:true});
  const geometry=await page.evaluate(()=>[...document.querySelectorAll('svg > g')].map(g=>{const b=g.getBBox();return{id:g.id,x:b.x,y:b.y,w:b.width,h:b.height};}));
  await page.addStyleTag({content:'html,body{background:white}'});
  await page.screenshot({path:path.join(qaDir,'white_background_preview.png')});
  const meta=await sharp(path.join(dir,stem+'.png')).metadata(),stats=await sharp(path.join(dir,stem+'.png')).stats();
  const sourceChecks=emittingRows.every(row=>svg.includes(`data-source-core="${row},9" data-origin="${sourceX},${rowY(row)}"`)&&svg.includes(`M${sourceX} ${rowY(row)} H${materialX}`));
  const checks={no_text_or_formula:!/<text|<title|<desc|<foreignObject/.test(svg),no_raster_or_heatmap:!/<image|linearGradient|filter/.test(svg),exact_100_fiber_cores:(svg.match(/data-core="\d+,\d+"/g)||[]).length===100,short_2_5d_array_block:svg.includes('id="smf_array"')&&svg.includes('M58 188 H224 L263 168 V412 L224 432 H58 Z'),six_gaussian_modes_from_selected_cores:sourceChecks&&(svg.match(/data-mode="lp01-like"/g)||[]).length===6,near_field_modes_remain_discrete:beamWidth(0)*2<22.2,midfield_modes_overlap:beamWidth(180)*2>22.2,far_field_modes_form_continuous_overlap:beamWidth(materialX-faceEdge)*2>22.2,smoothly_diverging_gaussian_envelopes:svg.includes('<radialGradient')&&svg.includes('stop-opacity="0"'),no_hard_triangular_edges_or_black_gaps:!/<polygon|#000000|black/i.test(svg),single_thin_orange_material:svg.includes('M800 139 H806 V461 H800 Z'),decoder_and_detector_preserved:['decoder_layer_1','decoder_layer_2','detector'].every(id=>svg.includes(`id="${id}"`)),editable_independent_groups:['smf_array','emitted_modes','nonlinear_material','post_nonlinear_field','decoder_layer_1','decoder_layer_2','detector'].every(id=>svg.includes(`id="${id}"`)),transparent_png:meta.hasAlpha&&stats.channels[3].min===0,png_width_at_least_2400:meta.width>=2400,aspect_ratio_2_5:meta.width/meta.height===2.5,no_clipping:geometry.every(g=>g.x>=0&&g.y>=0&&g.x+g.w<=1600&&g.y+g.h<=640)};
  const report={status:Object.values(checks).every(Boolean)?'PASS':'FIX_BEFORE_DELIVERY',checks,geometry,png:{width:meta.width,height:meta.height,alpha:meta.hasAlpha},scientific_status:'Target schematic; beam envelopes and wavefronts are illustrative rather than measured or simulated.',visual_review:'PENDING'};
  fs.writeFileSync(path.join(dir,stem+'.qa.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({status:report.status,checks,png:report.png}));
}finally{await browser.close();}
