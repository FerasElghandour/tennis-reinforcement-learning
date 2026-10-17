"""Audit published raw Tennis scores, validation selection and actor hashes offline."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def score_rows(rows):
    for row in rows:
        a,b,s=map(float,[row['agent_0_return'],row['agent_1_return'],row['episode_score']])
        if not all(map(math.isfinite,[a,b,s])) or not math.isclose(s,max(a,b),abs_tol=1e-9):
            raise ValueError('Episode score must be max of finite undiscounted returns')
        if int(row['steps'])<1:raise ValueError('Positive episode length required')
    return len(rows)

def verify(root):
    root=Path(root);checked=0;models=0
    for campaign in ['baseline','exploration']:
        folder=root/campaign;manifest=read(folder/'manifest.json');protocol=manifest['protocol']
        if manifest['status']!='completed':raise ValueError('Incomplete campaign')
        results=read(folder/'results.json')
        if [r['seed'] for r in results]!=protocol['training_seeds']:raise ValueError('All declared training seeds required in order')
        for result in results:
            out=folder/f"seed-{result['seed']}"
            if read(out/'result.json')!=result:raise ValueError('Campaign and per-seed records differ')
            with (out/'training.csv').open(newline='') as f:training=list(csv.DictReader(f))
            if len(training)!=protocol['episodes_per_seed'] or [int(r['episode']) for r in training]!=list(range(1,len(training)+1)):
                raise ValueError('Complete ordered training episode budget required')
            checked+=score_rows(training)
            validations=read(out/'validation.json')
            if [v['training_episode'] for v in validations]!=list(range(protocol['validation_every_episodes'],protocol['episodes_per_seed']+1,protocol['validation_every_episodes'])):
                raise ValueError('Every declared validation interval required')
            for v in validations:
                if len(v['episodes'])!=protocol['validation_episodes']:raise ValueError('Wrong validation count')
                checked+=score_rows(v['episodes'])
                if any(r['truncated'] for r in v['episodes']):raise ValueError('Truncated validation used for selection')
                mean=sum(r['episode_score'] for r in v['episodes'])/len(v['episodes'])
                if not math.isclose(mean,v['mean_score'],abs_tol=1e-9):raise ValueError('Wrong validation mean')
            selected=max(validations,key=lambda v:v['mean_score'])
            if selected['training_episode']!=result['selected_episode']:raise ValueError('Checkpoint must use first validation maximum')
            if not math.isclose(selected['mean_score'],result['validation_mean'],abs_tol=1e-9):raise ValueError('Selected validation mean differs')
            digest=hashlib.sha256((out/'actor.pt').read_bytes()).hexdigest()
            if digest!=result['checkpoint_sha256']:raise ValueError('Actor fingerprint mismatch')
            models+=1
            expected={'learned','random','zero'} if campaign=='baseline' else {'exploration','frozen_baseline','random','zero'}
            if set(result['evaluation'])!=expected:raise ValueError('All baseline comparisons required')
            for value in result['evaluation'].values():
                records=value['episodes'];checked+=score_rows(records)
                if len(records)!=protocol['test_episodes']:raise ValueError('Wrong final test count')
                mean=sum(r['episode_score'] for r in records)/len(records)
                if not math.isclose(mean,value['mean_score'],abs_tol=1e-9):raise ValueError('Wrong final mean')
                if sum(r['truncated'] for r in records)!=value['truncated_episodes']:raise ValueError('Wrong truncation count')
            if campaign=='exploration':
                baseline=read(root/f"baseline/seed-{result['seed']}/result.json")
                if result['baseline_checkpoint_sha256']!=baseline['checkpoint_sha256']:raise ValueError('Frozen comparison actor changed')
    return {'training_policies':models,'audited_episode_records':checked,'score_identity_and_budgets':True,
            'validation_only_first_maximum_selection':True,'actor_fingerprints':True,
            'limits':'Offline record/selection audit, not another simulator run or proof of cross-platform determinism.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('results/fresh-v1'));a=p.parse_args()
    print(json.dumps(verify(a.results),indent=2))
if __name__=='__main__':main()
