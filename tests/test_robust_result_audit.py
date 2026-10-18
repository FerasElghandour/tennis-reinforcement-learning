"""Synthetic bookkeeping fixtures; these are not simulator performance evidence."""
import copy,csv,json
from pathlib import Path
import pytest
import fresh_training as base
from verify_robust_results import verify

def episode(n,score):
    return dict(episode=n,agent_0_return=score,agent_1_return=0.,episode_score=score,steps=10,truncated=False)

@pytest.fixture
def campaign(tmp_path):
    protocol=json.loads((base.ROOT/'ROBUSTNESS_PROTOCOL.json').read_text());final=[]
    for method in protocol['methods']:
        folder=tmp_path/method;folder.mkdir()
        base.dump(folder/'manifest.json',dict(status='completed',protocol=protocol,source_sha256={}))
        for seed in protocol['training_seeds']:
            out=folder/f'seed-{seed}';out.mkdir();(out/'actor.pt').write_bytes(b'fixture-only')
            validation=[dict(environment_step=step,mean_score=score,episodes=[episode(i+1,score) for i in range(10)])
                        for step,score in [(10000,.2),(20000,.5),(30000,.5),(40000,.1)]]
            base.dump(out/'validation.json',validation)
            base.dump(out/'training-result.json',dict(selected_step=20000,validation_mean=.5,checkpoint_sha256=base.sha(out/'actor.pt')))
            with (out/'training.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['episode','environment_step','agent_0_return','agent_1_return','episode_score','steps','truncated','budget_cutoff']);w.writeheader()
                for n in range(1,5):w.writerow(dict(episode=n,environment_step=n*10000,agent_0_return=.1,agent_1_return=0.,episode_score=.1,steps=10000,truncated=True,budget_cutoff=False))
    for seed in protocol['training_seeds']:
        for offset in protocol['test_environment_seed_offsets']:
            for policy in [*protocol['methods'],'random','zero']:
                score=.7 if policy in protocol['methods'] else .01
                final.append(dict(training_seed=seed,environment_seed=seed+offset,policy=policy,mean_score=score,truncated_episodes=0,episodes=[episode(i+1,score) for i in range(50)]))
    target=tmp_path/'final-test';target.mkdir();base.dump(target/'results.json',final)
    base.dump(target/'manifest.json',dict(status='completed',protocol=protocol,results_sha256=base.sha(target/'results.json'),training_manifest_sha256={m:base.sha(tmp_path/m/'manifest.json') for m in protocol['methods']}))
    return tmp_path

def test_equal_budget_first_validation_tie_and_nested_replication(campaign):
    result=verify(campaign)
    assert result['pass'] and result['training_steps']==400000 and result['final_episodes']==2000
    assert len(result['summary']['td3_layernorm']['per_seed'])==5
    assert result['summary']['td3_layernorm']['seeds_at_least_0_5']==5

def test_rejects_checkpoint_selected_from_later_tie(campaign):
    p=campaign/'ddpg_exploration/seed-104729/training-result.json';d=json.loads(p.read_text());d['selected_step']=30000;base.dump(p,d)
    assert 'Selection was not first validation maximum' in verify(campaign)['errors']

def test_rejects_nonfinite_and_changed_final_scores(campaign):
    p=campaign/'final-test/results.json';d=json.loads(p.read_text());d[0]['episodes'][0]['episode_score']=float('nan')
    p.write_text(json.dumps(d))
    result=verify(campaign)
    assert not result['pass'] and 'Non-finite final score' in result['errors'] and 'Final result fingerprint changed' in result['errors']

def test_rejects_missing_final_environment_family(campaign):
    p=campaign/'final-test/results.json';d=json.loads(p.read_text());d.pop();base.dump(p,d)
    result=verify(campaign)
    assert not result['pass'] and 'Missing/duplicated test environment seed' in result['errors']
