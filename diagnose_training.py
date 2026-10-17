"""Describe every finished fresh run and reward sparsity without changing selection."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Wait for the complete predeclared campaign')
    diagnostics=[]
    for seed in manifest['protocol']['training_seeds']:
        root=a.run/f'seed-{seed}';rows=list(csv.DictReader((root/'training.csv').open()));result=json.loads((root/'result.json').read_text())
        scores=np.array([float(r['episode_score']) for r in rows]);steps=np.array([int(r['steps']) for r in rows])
        rolling=np.convolve(scores,np.ones(100)/100,mode='valid')
        diagnostics.append({'seed':seed,'training_episodes':len(scores),'positive_score_episodes':int((scores>0).sum()),
            'positive_score_episode_fraction':float((scores>0).mean()),'zero_score_episode_fraction':float((scores==0).mean()),
            'mean_episode_steps':float(steps.mean()),'median_episode_steps':float(np.median(steps)),
            'best_rolling_100_training_mean':float(rolling.max()),'final_rolling_100_training_mean':float(rolling[-1]),
            'selected_episode':result['selected_episode'],'validation_mean':result['validation_mean'],
            'evaluation':{name:{k:v for k,v in item.items() if k!='episodes'} for name,item in result['evaluation'].items()}})
    summary={'per_seed':diagnostics,'scope':'All predeclared training seeds. Positive-score episode fraction describes observed reward sparsity, not per-transition reward density. No hyperparameters or checkpoints are chosen from these final evaluations.'}
    (a.run/'diagnostics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
