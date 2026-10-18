"""Regenerate all-seed training and final-test figures from audited records."""
from pathlib import Path
import argparse,csv,html,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results/robust-v2');a=p.parse_args();root=a.results
report=json.loads((root/'verification.json').read_text())
if not report['pass']:raise ValueError('Audit must pass before figures are generated')
summary=report['summary'];methods=['ddpg_exploration','td3_layernorm'];names={'ddpg_exploration':'DDPG exploration','td3_layernorm':'TD3 / LayerNorm'};colors={'ddpg_exploration':'#396981','td3_layernorm':'#b25d36'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#332b3b','axes.labelcolor':'#332b3b','axes.edgecolor':'#d8d0da','savefig.facecolor':'#fcfaf6','figure.facecolor':'#fcfaf6'})
fig,axes=plt.subplots(2,1,figsize=(7.2,8.5),gridspec_kw={'height_ratios':[1.1,1]},layout=None)
fig.subplots_adjust(left=.14,right=.97,top=.86,bottom=.25,hspace=.55)
fig.text(.055,.96,'Five new training seeds per pipeline',fontsize=20,weight='bold')
fig.text(.055,.918,'40,000 environment steps each · all seeds retained',fontsize=12)
for method in methods:
 for i,row in enumerate(summary[method]['per_seed']):
  with (root/method/f'seed-{row["seed"]}/training.csv').open() as f:records=list(csv.DictReader(f))
  x=np.array([int(r['environment_step']) for r in records]);y=np.array([float(r['episode_score']) for r in records]);window=100
  if len(y)>=window:axes[0].plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),color=colors[method],alpha=.65,lw=1.2,label=names[method] if i==0 else None)
axes[0].set(xlim=(0,40000),ylim=(0,None),xlabel='Training environment steps',ylabel='Rolling-100 episode score',title='Training curves · separate from final evaluation')
axes[0].set_xticks([0,10000,20000,30000,40000],['0','10k','20k','30k','40k']);axes[0].grid(alpha=.15);axes[0].legend(frameon=False,fontsize=10)
seeds=[r['seed'] for r in summary[methods[0]]['per_seed']]
for i,seed in enumerate(seeds):
 values=[summary[m]['per_seed'][i]['mean'] for m in methods];axes[1].plot(values,[i,i],color='#d8d0da',lw=2)
 for method,value in zip(methods,values):axes[1].scatter(value,i,s=50,color=colors[method],zorder=4)
 axes[1].scatter(summary['random']['per_seed'][i]['mean'],i,s=28,marker='x',color='#796b80',zorder=5)
axes[1].set_yticks(range(5),[str(seed) for seed in seeds]);axes[1].invert_yaxis();axes[1].set(xlabel='Final-test mean episode score',ylabel='Training seed',title='100 final episodes per policy and training seed',xlim=(-.04,max(2.6,max(summary[m]['max'] for m in methods)+.15)))
axes[1].grid(axis='x',alpha=.15);axes[1].axvline(.5,color='#796b80',ls=':',lw=1);axes[1].text(.51,.96,'0.5 reference',transform=axes[1].get_xaxis_transform(),va='top',fontsize=9,color='#796b80')
fig.text(.055,.13,'Mean ≥ 0.5: '+', '.join(f'{names[m]} {summary[m]["seeds_at_least_0_5"]}/5' for m in methods)+'.',fontsize=11.5,weight='bold')
fig.text(.055,.095,'Final tests span two new environment seeds; random policies are marked ×.\nCheckpoint selection used validation only. Five trained policies per method.\nEqual interactions, different architecture/noise/gradient compute; descriptive comparison.',fontsize=10,va='top',linespacing=1.5)
fig.savefig(root/'training-and-final.png',dpi=180);plt.close(fig)
# A large-label rendering for narrow screens, using identical audited values.
s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 900" role="img" aria-labelledby="title desc"><title id="title">Every final-test training seed</title><desc id="desc">Five trained policies per method. Each value is a mean across one hundred held-out episodes and two environment seeds. Every DDPG, TD3 and matched random result is reported.</desc><rect width="400" height="900" fill="#fcfaf6"/><g font-family="Arial,sans-serif" fill="#332b3b">']
def text(x,y,t,size=16,extra=''):return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{html.escape(t)}</text>'
s += [text(20,35,'Every training seed',23,'font-weight="700"'),text(20,64,'Final-test mean / 100 episodes per policy',16)]
xmax=max(2.6,math.ceil(max(summary[m]['max'] for m in methods)*2)/2)
for i,seed in enumerate(seeds):
 y=110+i*120;s.append(text(20,y,f'Seed {seed}',17,'font-weight="700"'));s.append(text(378,y,f'Random {summary["random"]["per_seed"][i]["mean"]:.4f}',14,'text-anchor="end"'))
 for j,method in enumerate(methods):
  yy=y+29+j*34;value=summary[method]['per_seed'][i]['mean'];name='DDPG' if j==0 else 'TD3'
  s += [text(20,yy,name,15),text(378,yy,f'{value:.4f}',16,'text-anchor="end" font-weight="700"'),f'<rect x="94" y="{yy-12}" width="210" height="12" fill="#e9e3e9"/><rect x="94" y="{yy-12}" width="{210*value/xmax:.3f}" height="12" fill="{colors[method]}"/>']
for tick in [0,xmax/2,xmax]:s.append(text(94+210*tick/xmax,731,f'{tick:g}',14,'text-anchor="middle"'))
s += [text(20,769,'Five trained policies per method; descriptive.',14),text(20,792,'40,000 training interactions each.',14),text(20,815,'Two fresh test environments per training seed.',14),text(20,838,'Checkpoint selection uses validation only.',14),text(20,861,'Pipeline architecture, noise and compute differ.',14),'</g></svg>']
(root/'final-test-mobile.svg').write_text(''.join(s))
print('Wrote figures from all audited training seeds.')
