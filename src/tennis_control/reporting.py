"""Self-contained, offline analytical report renderer and provenance manifest."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform


def publish_report(output, result, spec, inputs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    code_root = Path(__file__).resolve().parent
    repo_root = next((p for p in code_root.parents if (p/'pyproject.toml').exists()), code_root)
    code = [{'path': str(p.relative_to(repo_root)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(code_root.glob('*.py'))]
    manifest = {'python': platform.python_version(), 'dependencies': versions, 'code': code,
                'inputs': [{'name': Path(p).name, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                            'bytes': Path(p).stat().st_size} for p in inputs]}
    # Reject non-finite values rather than writing invalid JSON that browsers cannot read.
    payload = json.dumps({'report': spec, 'results': result, 'provenance': manifest}, allow_nan=False, indent=2)
    (output/'metrics.json').write_text(payload+'\n', encoding='utf-8')
    safe_payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = TEMPLATE.replace('__TITLE__', html.escape(spec['title'])).replace('__PAYLOAD__', safe_payload)
    (output/'index.html').write_text(page, encoding='utf-8')
    (output/'.nojekyll').touch()
    print(f'Wrote {output / "index.html"} and {output / "metrics.json"}')


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="An interactive technical case study by Feras Elghandour, with measured results and reproducible methods.">
<title>__TITLE__ | Feras Elghandour</title>
<style>
:root{--paper:oklch(98% .006 65);--ink:oklch(25% .025 315);--muted:oklch(47% .025 315);--rule:oklch(85% .012 315);--accent:oklch(48% .17 32);--wash:oklch(94% .012 315);--blue:oklch(45% .12 255);--orange:oklch(55% .15 50);--violet:oklch(47% .15 310)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 'Public Sans',Helvetica,Arial,sans-serif}a{color:var(--accent);text-underline-offset:4px}a:hover{text-decoration-thickness:2px}a:focus-visible,button:focus-visible,select:focus-visible{outline:3px solid var(--accent);outline-offset:4px}.skip{position:absolute;left:16px;top:-80px;background:var(--paper);padding:10px}.skip:focus{top:16px}header,main,footer{max-width:1320px;margin:auto;padding-inline:clamp(22px,5vw,78px)}header{padding-block:25px;border-bottom:1px solid var(--rule);display:flex;justify-content:space-between;gap:20px;font-size:14px}header a{color:var(--ink)}.author{font-weight:700;letter-spacing:-.025em}.intro{padding-block:70px 48px;max-width:980px}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:700;color:var(--accent)}h1{font-size:clamp(38px,5.7vw,74px);letter-spacing:-.055em;line-height:1.08;margin:17px 0 25px;font-weight:750}.subtitle{font-size:clamp(18px,2vw,23px);line-height:1.5;max-width:65ch;color:var(--muted)}.layout{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:55px}.layout>section,aside{min-width:0}.findings{padding:0 0 25px;list-style:none;max-width:78ch}.findings li{padding:15px 0;border-top:1px solid var(--rule);font-size:17px}.section-label{display:flex;align-items:baseline;gap:13px;margin:25px 0 20px}.section-label span{font-size:12px;font-weight:700;color:var(--accent)}h2{font-size:25px;line-height:1.3;letter-spacing:-.025em;margin:0}h3{font-size:17px;margin:23px 0 8px}.controls{display:flex;justify-content:space-between;align-items:end;gap:20px;flex-wrap:wrap;margin:20px 0 12px}label{display:block;font-size:12px;font-weight:700;color:var(--muted);margin-bottom:5px}select,button{font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:3px;padding:10px 14px;max-width:100%}select{width:100%;min-width:240px}button{cursor:pointer;font-size:13px}button:hover{border-color:var(--accent)}.plot{overflow-x:auto;border-block:1px solid var(--rule);padding:20px 0 10px}.plot svg{min-width:560px;width:100%;height:auto;display:block;overflow:visible}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;margin-bottom:12px}.legend span{display:flex;gap:7px;align-items:center}.legend i{height:3px;width:20px;display:inline-block}.caption{font-size:13px;color:var(--muted);max-width:72ch}.table-wrap{max-height:360px;overflow:auto;border-bottom:1px solid var(--rule);margin:20px 0 8px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid var(--rule);padding:11px 12px;white-space:nowrap}th{position:sticky;top:0;background:var(--wash);font-weight:600}td:not(:first-child){text-align:right}.secondary{font-size:14px;color:var(--muted)}aside{border-top:1px solid var(--rule);padding-top:24px;margin-top:16px}aside p{font-size:14px;color:var(--muted)}details{margin:24px 0}summary{cursor:pointer;font-size:14px;font-weight:600}.provenance{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--wash);padding:15px;overflow:auto;max-height:300px}.closing{padding-block:45px 65px;margin-top:50px;border-top:1px solid var(--rule)}.project-links{display:flex;gap:15px 28px;flex-wrap:wrap;font-size:14px}.project-links a{color:var(--ink)}footer{padding-block:23px 35px;border-top:1px solid var(--rule);font-size:12px;color:var(--muted);display:flex;justify-content:space-between;gap:20px}.error{background:var(--wash);padding:20px;color:var(--accent)}
@media(max-width:850px){.layout{grid-template-columns:1fr;gap:20px}.intro{padding-top:45px}aside{display:grid;grid-template-columns:1fr 1fr;gap:0 25px}aside .section-label{grid-column:1/-1}.controls>div{width:100%}select{min-width:0}.project-links{display:grid;gap:10px}footer{display:block}.plot{padding-top:14px}}@media(max-width:480px){aside{display:block}header{font-size:12px}h1{font-size:40px}.findings li{font-size:15px}}@media print{header,button,select,label,.project-links,.skip{display:none}body{background:var(--paper)}.intro{padding-top:25px}.layout{display:block}aside{columns:2}.table-wrap{max-height:none}.closing{display:none}.plot{break-inside:avoid}a{color:var(--ink)}}
</style></head><body><a class="skip" href="#evidence">Skip to evidence</a>
<header><a class="author" href="https://github.com/FerasElghandour">Feras Elghandour</a><span>Selected work / Interactive case study</span></header>
<main><section class="intro"><p class="eyebrow" id="category"></p><h1 id="title"></h1><p class="subtitle" id="subtitle"></p></section>
<div class="layout"><section id="evidence" aria-labelledby="evidence-heading"><ul class="findings" id="findings"></ul>
<div class="section-label"><span>01</span><h2 id="evidence-heading">Explore the evidence</h2></div>
<div class="controls"><div><label for="view">Comparison</label><select id="view"></select></div><button id="csv" type="button">Download chart data</button></div>
<div class="plot" tabindex="0" aria-label="Chart. Scroll horizontally on narrow screens."><div class="legend" id="legend"></div><svg id="chart" viewBox="0 0 900 440" role="img" aria-labelledby="chart-title chart-desc"></svg></div>
<p class="caption" id="chart-caption" aria-live="polite"></p><div class="table-wrap"><table id="data-table"><caption class="skip">Values shown in the current chart</caption><thead></thead><tbody></tbody></table></div><p class="caption" id="row-count"></p>
<details><summary>Reproducibility record</summary><pre class="provenance" id="provenance"></pre><button id="json" type="button">Download full results JSON</button></details><p class="caption" id="notes"></p>
</section><aside aria-labelledby="method-heading"><div class="section-label"><span>02</span><h2 id="method-heading">Method & judgment</h2></div><div id="methods"></div></aside></div>
<section class="closing"><div class="section-label"><span>03</span><h2>More work to explore</h2></div><nav class="project-links" aria-label="Portfolio projects"><a href="https://github.com/FerasElghandour/bikeshare-weather-analysis">Weather & demand</a><a href="https://github.com/FerasElghandour/bikeshare-customer-behavior">Customer behavior</a><a href="https://github.com/FerasElghandour/movie-revenue-analysis">Movie revenue</a><a href="https://github.com/FerasElghandour/tennis-reinforcement-learning">Continuous control</a></nav></section></main>
<footer><span>Analysis and implementation by Feras Elghandour</span><span>Offline report · No tracking · Aggregate evidence</span></footer>
<script type="application/json" id="payload">__PAYLOAD__</script><script>
'use strict';
const model=JSON.parse(document.getElementById('payload').textContent),report=model.report,$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg',colors=['#285b91','#ad5828','#754d91'];
const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
const svg=(tag,attrs,text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n};
const fmt=v=>v===null||v===undefined?'—':typeof v==='number'?new Intl.NumberFormat('en',{maximumFractionDigits:3}).format(v):String(v);
for(const id of ['title','subtitle','category','notes'])$(id).textContent=report[id];
for(const finding of report.findings)$('findings').append(el('li',finding));
for(const method of report.methodology){const section=el('section');section.append(el('h3',method.title),el('p',method.body));$('methods').append(section)}
report.views.forEach((view,i)=>{const option=el('option',view.label);option.value=i;$('view').append(option)});
$('provenance').textContent=JSON.stringify(model.provenance,null,2);
let rows=[];
function render(){const view=report.views[Number($('view').value)],chart=$('chart');chart.replaceChildren();$('legend').replaceChildren();
chart.append(svg('title',{id:'chart-title'},view.label),svg('desc',{id:'chart-desc'},'The corresponding values and any uncertainty intervals are available in the table below.'));
rows=[];view.series.forEach((series,i)=>{const label=el('span'),swatch=el('i');swatch.style.background=colors[i%3];label.append(swatch,document.createTextNode(series.name));$('legend').append(label);series.points.forEach(p=>{if(Number.isFinite(p.y))rows.push({series:series.name,x:p.x,y:p.y,low:p.low??null,high:p.high??null})})});
const values=rows.flatMap(r=>[r.y,r.low,r.high]).filter(Number.isFinite);if(!values.length){$('chart-caption').textContent='No finite values available for this comparison.';return}
const lo=Math.min(0,...values),hi=Math.max(...values,0),pad=(hi-lo||1)*.08,min=lo<0?lo-pad:0,max=hi+pad,x0=75,y0=25,w=795,h=320,y=v=>y0+h-(v-min)/(max-min)*h;
for(let i=0;i<=4;i++){const v=min+(max-min)*i/4,yy=y(v);chart.append(svg('line',{x1:x0,y1:yy,x2:x0+w,y2:yy,stroke:'#ded9df','stroke-width':1}),svg('text',{x:x0-12,y:yy+4,'text-anchor':'end',fill:'#655b68','font-size':12},fmt(v)))}
chart.append(svg('text',{x:x0,y:14,fill:'#655b68','font-size':12},view.unit));
if(view.kind==='bar'){const gap=w/rows.length;rows.forEach((r,i)=>{const xx=x0+gap*i+gap*.2,bw=gap*.6;chart.append(svg('rect',{x:xx,y:Math.min(y(r.y),y(0)),width:bw,height:Math.max(1,Math.abs(y(r.y)-y(0))),fill:colors[0]}));if(r.low!==null&&r.high!==null){const cx=xx+bw/2;chart.append(svg('line',{x1:cx,y1:y(r.low),x2:cx,y2:y(r.high),stroke:'#322735','stroke-width':2}));for(const v of [r.low,r.high])chart.append(svg('line',{x1:cx-7,y1:y(v),x2:cx+7,y2:y(v),stroke:'#322735','stroke-width':2}))}const name=String(r.x),label=svg('text',{x:xx+bw/2,y:y0+h+24,'text-anchor':'middle',fill:'#322735','font-size':12});const words=name.length>19?name.split(' '):[name];let line='';let lines=[];for(const word of words){if((line+' '+word).trim().length>20){lines.push(line);line=word}else line=(line+' '+word).trim()}lines.push(line);lines.slice(0,3).forEach((l,j)=>label.append(svg('tspan',{x:xx+bw/2,dy:j?15:0},l)));chart.append(label)})}
else{const allX=[...new Set(rows.map(r=>r.x))],numeric=allX.every(x=>typeof x==='number'),xmin=numeric?Math.min(...allX):0,xmax=numeric?Math.max(...allX):allX.length-1,x=v=>x0+((numeric?v:allX.indexOf(v))-xmin)/(xmax-xmin||1)*w;
view.series.forEach((series,i)=>{const points=series.points.filter(p=>Number.isFinite(p.y));const stride=Math.max(1,Math.ceil(points.length/500));const sampled=points.filter((p,j)=>j%stride===0||j===points.length-1);chart.append(svg('polyline',{points:sampled.map(p=>`${x(p.x)},${y(p.y)}`).join(' '),fill:'none',stroke:colors[i%3],'stroke-width':2.5,'stroke-dasharray':i===1?'7 4':i===2?'2 3':'none'}))});for(let i=0;i<=4;i++){const idx=Math.round((allX.length-1)*i/4),value=allX[idx];chart.append(svg('text',{x:x(value),y:y0+h+25,'text-anchor':'middle',fill:'#655b68','font-size':12},String(value)))}}
const header=$('data-table').querySelector('thead'),body=$('data-table').querySelector('tbody');header.replaceChildren();body.replaceChildren();const tr=el('tr');for(const label of ['Series','Observation','Value','95% lower','95% upper']){const th=el('th',label);th.scope='col';tr.append(th)}header.append(tr);for(const row of rows.slice(0,200)){const tr=el('tr');for(const value of [row.series,row.x,row.y,row.low,row.high])tr.append(el('td',fmt(value)));body.append(tr)}
$('row-count').textContent=rows.length>200?`Showing the first 200 of ${rows.length} values. Download chart data for the complete comparison.`:`${rows.length} values. Full numerical precision is preserved in the downloads.`;
$('chart-caption').textContent=view.label+'. '+(rows.some(r=>r.low!==null)?'Whiskers show the stated 95% bootstrap interval. ':'')+'Values are available in the table and CSV.'}
function download(name,text,type){const a=el('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
$('view').addEventListener('change',render);$('csv').addEventListener('click',()=>{const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';download('comparison.csv','series,observation,value,lower95,upper95\n'+rows.map(r=>[r.series,r.x,r.y,r.low,r.high].map(q).join(',')).join('\n'),'text/csv')});$('json').addEventListener('click',()=>download('results.json',JSON.stringify(model,null,2),'application/json'));render();
</script><noscript><p class="error">Enable JavaScript to use the interactive chart. The repository also includes metrics.json and written results.</p></noscript></body></html>'''
