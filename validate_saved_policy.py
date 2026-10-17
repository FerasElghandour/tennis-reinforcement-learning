"""Fresh live evaluation of one fixed saved actor over explicit environment seeds.

Repeated environment sequences do not represent independently trained policies.
"""
import argparse,csv,json,platform,time
from pathlib import Path
import numpy as np
import torch
from fresh_training import MADDPGAgent,evaluate,sha,dump,ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--env',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',default='810001,810002,810003');p.add_argument('--episodes',type=int,default=100);a=p.parse_args()
    seeds=[int(s) for s in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or min(seeds)<0 or a.episodes<1 or a.out.exists():raise ValueError('Unique nonnegative seeds, positive budget and a new output directory required')
    a.out.mkdir(parents=True);torch.set_num_threads(1);agent=MADDPGAgent(24,2,2,0)
    agent.actor_local.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True)
    protocol={'checkpoint_sha256':sha(a.checkpoint),'environment_seeds':seeds,'episodes_per_seed':a.episodes,
              'learning':False,'exploration':False,'selection':'fixed input checkpoint; no selection from these evaluations',
              'source_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'fresh_training.py',ROOT/'src/tennis_control/model.py',ROOT/'src/tennis_control/maddpg_agent.py']},
              'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
              'scope':'Fresh environment sequences for one fixed historical actor, not multiple training seeds or exact replication of the old sequence.'}
    dump(a.out/'protocol.json',protocol);results=[];started=time.perf_counter()
    for seed in seeds:
        rows=evaluate(a.env.resolve(),seed,20,a.episodes,10000,agent)
        with (a.out/f'seed-{seed}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        result={'environment_seed':seed,'episodes':len(rows),'mean_score':float(np.mean([r['episode_score'] for r in rows])),
                'median_score':float(np.median([r['episode_score'] for r in rows])),'truncated_episodes':sum(r['truncated'] for r in rows)}
        results.append(result);dump(a.out/'results.json',results);print(json.dumps(result),flush=True)
    dump(a.out/'summary.json',{'sequences':results,'mean_of_sequence_means':float(np.mean([r['mean_score'] for r in results])),
                              'wall_seconds':time.perf_counter()-started,'scope':protocol['scope']})

if __name__=='__main__':main()
