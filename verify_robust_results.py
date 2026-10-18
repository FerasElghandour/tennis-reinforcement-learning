"""Audit exact training budgets, validation selection and every nested final episode."""
import argparse,csv,hashlib,json,math,statistics,zipfile
from pathlib import Path
import fresh_training as base

def finite_scores(row):
    return all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in ['episode_score','agent_0_return','agent_1_return','steps'])

def verify(root):
    protocol=json.loads((base.ROOT/'ROBUSTNESS_PROTOCOL.json').read_text());errors=[];families={m:[] for m in [*protocol['methods'],'random','zero']}
    final=json.loads((root/'final-test/results.json').read_text())
    final_manifest=json.loads((root/'final-test/manifest.json').read_text())
    if final_manifest['status']!='completed' or final_manifest['protocol']!=protocol:errors.append('Final evaluation incomplete or changed protocol')
    if final_manifest['results_sha256']!=base.sha(root/'final-test/results.json'):errors.append('Final result fingerprint changed')
    expected=len(protocol['training_seeds'])*len(protocol['test_environment_seed_offsets'])*len(families)
    if len(final)!=expected:errors.append('Missing or additional final policy/environment cells')
    for method in protocol['methods']:
        manifest=json.loads((root/method/'manifest.json').read_text())
        if manifest['status']!='completed' or manifest['protocol']!=protocol:errors.append(method+': incomplete or changed protocol')
        if final_manifest['training_manifest_sha256'].get(method)!=base.sha(root/method/'manifest.json'):errors.append('Training manifest changed after final evaluation')
        if (root/method/'executed-source.zip').exists():
            with zipfile.ZipFile(root/method/'executed-source.zip') as z:
                for name,digest in manifest['source_sha256'].items():
                    if hashlib.sha256(z.read(name)).hexdigest()!=digest:errors.append('Executed source fingerprint mismatch')
        for seed in protocol['training_seeds']:
            folder=root/method/f'seed-{seed}'
            result=json.loads((folder/'training-result.json').read_text())
            rows=list(csv.DictReader((folder/'training.csv').open()))
            if sum(int(r['steps']) for r in rows)!=protocol['environment_steps_per_seed']:errors.append(f'{method}/{seed}: training budget')
            cumulative=0
            for n,row in enumerate(rows,1):
                cumulative+=int(row['steps'])
                if not 1<=int(row['steps'])<=protocol['max_steps_per_episode']:errors.append('Training episode exceeds declared step cap')
                if not all(math.isfinite(float(row[k])) for k in ['episode_score','agent_0_return','agent_1_return']):errors.append('Non-finite training score')
                if int(row['episode'])!=n or int(row['environment_step'])!=cumulative:errors.append(f'{method}/{seed}: discontinuous episode history')
                if abs(float(row['episode_score'])-max(float(row['agent_0_return']),float(row['agent_1_return'])))>1e-9:errors.append('Invalid training score')
                if row['budget_cutoff']=='True' and n!=len(rows):errors.append('Early budget cutoff')
            validation=json.loads((folder/'validation.json').read_text())
            if [v['environment_step'] for v in validation]!=list(range(protocol['validation_every_steps'],protocol['environment_steps_per_seed']+1,protocol['validation_every_steps'])):errors.append('Validation schedule')
            for cell in validation:
                if len(cell['episodes'])!=protocol['validation_episodes'] or any(r['truncated'] for r in cell['episodes']):errors.append('Incomplete validation')
                if not all(finite_scores(r) for r in cell['episodes']) or not math.isfinite(cell['mean_score']):errors.append('Non-finite validation score')
                mean=statistics.mean(r['episode_score'] for r in cell['episodes'])
                if abs(mean-cell['mean_score'])>1e-9:errors.append('Altered validation mean')
            best=max(validation,key=lambda v:v['mean_score'])
            if best['environment_step']!=result['selected_step'] or abs(best['mean_score']-result['validation_mean'])>1e-9:errors.append('Selection was not first validation maximum')
            if base.sha(folder/'actor.pt')!=result['checkpoint_sha256']:errors.append('Actor fingerprint changed')
    for seed in protocol['training_seeds']:
        for policy in families:
            cells=[r for r in final if r['training_seed']==seed and r['policy']==policy]
            if sorted(r['environment_seed'] for r in cells)!=sorted(seed+offset for offset in protocol['test_environment_seed_offsets']):errors.append('Missing/duplicated test environment seed')
            scores=[]
            for cell in cells:
                episodes=cell['episodes']
                if len(episodes)!=protocol['test_episodes_per_environment_seed'] or cell['truncated_episodes']!=sum(r['truncated'] for r in episodes):errors.append('Invalid final budget/truncation count')
                if any(r['truncated'] for r in episodes):errors.append('Final episode step cap reached')
                for n,row in enumerate(episodes,1):
                    if not finite_scores(row):errors.append('Non-finite final score')
                    if row['episode']!=n or abs(row['episode_score']-max(row['agent_0_return'],row['agent_1_return']))>1e-9:errors.append('Invalid final episode identity')
                if not math.isfinite(cell['mean_score']):errors.append('Non-finite final mean')
                if abs(statistics.mean(r['episode_score'] for r in episodes)-cell['mean_score'])>1e-9:errors.append('Changed final mean')
                scores.extend(r['episode_score'] for r in episodes)
            families[policy].append({'seed':seed,'mean':statistics.mean(scores),'median_episode_score':statistics.median(scores),'episodes':len(scores)})
    if any(not math.isfinite(v['mean']) for values in families.values() for v in values):
        return {'pass':False,'errors':errors or ['Non-finite aggregate'],'summary':{},'interpretation':'Invalid evidence; no numerical summary produced.'}
    summary={}
    for name,values in families.items():
        means=[r['mean'] for r in values]
        summary[name]={'per_seed':values,'mean':statistics.mean(means),'median':statistics.median(means),'sample_sd':statistics.stdev(means),
                       'min':min(means),'max':max(means),'seeds_at_least_0_5':sum(v>=.5 for v in means)}
    for name in protocol['methods']:
        summary[name]['seeds_beating_random']=sum(a['mean']>b['mean'] for a,b in zip(families[name],families['random']))
    runs=len(protocol['methods'])*len(protocol['training_seeds'])
    return {'pass':not errors,'errors':errors,'training_runs':runs,'training_steps':runs*protocol['environment_steps_per_seed'],'final_episodes':sum(len(c['episodes']) for c in final),
            'summary':summary,'unit_of_replication':'five independent trained policies per method, not the nested test episodes',
            'interpretation':'descriptive bounded comparison of complete pipelines; equal interactions do not equalize wall time or gradient compute'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=base.ROOT/'results/robust-v2');a=p.parse_args()
    report=verify(a.results);base.dump(a.results/'verification.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['pass'] else 1)

if __name__=='__main__':main()
