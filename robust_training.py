"""Equal-interaction multi-seed DDPG / TD3 campaign, with separately invoked final tests."""
import argparse, copy, csv, hashlib, json, os, platform, random, sys, time, zipfile
from pathlib import Path
from importlib import metadata
import numpy as np
import torch
import fresh_training as base
from exploration_training import WarmupAgent
from tennis_control.td3 import TD3Agent

ROOT = Path(__file__).resolve().parent

def make_agent(method, seed, protocol):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if method == 'ddpg_exploration':
        return WarmupAgent(24, 2, 2, seed, warmup=protocol['uniform_random_warmup_steps'], sigma=.3)
    if method == 'td3_layernorm':
        return TD3Agent(seed=seed, warmup=protocol['uniform_random_warmup_steps'])
    raise ValueError('Unknown method')

def sources():
    return [ROOT/'robust_training.py',ROOT/'ROBUSTNESS_PROTOCOL.json',ROOT/'fresh_training.py',ROOT/'exploration_training.py',ROOT/'maddpg_agent.py',
            *sorted((ROOT/'src/tennis_control').glob('*.py')),*sorted((ROOT/'vendor').rglob('*.py'))]

def train(args, protocol):
    if args.out.exists(): raise ValueError('Output already exists; preserve prior attempts')
    args.out.mkdir(parents=True)
    manifest={'protocol':protocol,'source_sha256':{str(p.relative_to(ROOT)):base.sha(p) for p in sources()},
              'method':args.method,'status':'started','started_unix':time.time(),'host':platform.platform(),
              'architecture':platform.machine(),'python':platform.python_version(),
              'packages':{p:metadata.version(p) for p in ['torch','numpy','grpcio','protobuf']},
              'environment_executable_sha256':base.sha(args.env/'Contents/MacOS/Tennis')}
    with zipfile.ZipFile(args.out/'executed-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources():z.write(p,p.relative_to(ROOT))
    base.dump(args.out/'manifest.json',manifest)
    try:
        for seed in protocol['training_seeds']:
            out=args.out/f'seed-{seed}';out.mkdir()
            agent=make_agent(args.method,seed,protocol)
            env=base.environment(args.env,seed,args.worker);brain=env.brain_names[0]
            validation=[];selected=None;best=-float('inf');completed=0;ep=1;step_in_episode=0;returns=np.zeros(2);steps=0
            fields=['episode','environment_step','agent_0_return','agent_1_return','episode_score','steps','truncated','budget_cutoff']
            started=time.time()
            try:
                states=env.reset(train_mode=True)[brain].vector_observations;agent.reset()
                with (out/'training.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                    while steps < protocol['environment_steps_per_seed']:
                        actions=agent.act(states,True);info=env.step(actions)[brain]
                        agent.step(states,actions,info.rewards,info.vector_observations,info.local_done)
                        returns+=info.rewards;states=info.vector_observations;steps+=1;step_in_episode+=1
                        terminal=bool(np.any(info.local_done));truncated=step_in_episode>=protocol['max_steps_per_episode'] and not terminal
                        cutoff=steps==protocol['environment_steps_per_seed'] and not (terminal or truncated)
                        if terminal or truncated or cutoff:
                            writer.writerow(dict(episode=ep,environment_step=steps,agent_0_return=float(returns[0]),agent_1_return=float(returns[1]),episode_score=float(returns.max()),steps=step_in_episode,truncated=truncated,budget_cutoff=cutoff));f.flush()
                            completed+=int(terminal);ep+=1;step_in_episode=0;returns=np.zeros(2)
                            if steps<protocol['environment_steps_per_seed']:
                                states=env.reset(train_mode=True)[brain].vector_observations;agent.reset()
                        if steps % protocol['validation_every_steps']==0:
                            val=base.evaluate(args.env,seed+protocol['validation_environment_seed_offset'],args.worker+1,protocol['validation_episodes'],protocol['max_steps_per_episode'],agent)
                            if any(r['truncated'] for r in val):raise RuntimeError('Validation step cap reached')
                            mean=float(np.mean([r['episode_score'] for r in val]))
                            validation.append({'environment_step':steps,'mean_score':mean,'episodes':val})
                            if mean>best:
                                best=mean;selected=steps;torch.save(agent.actor_local.state_dict(),out/'actor.pt')
                            base.dump(out/'validation.json',validation)
                            print(f'{args.method} seed={seed} step={steps} validation={mean:.4f} selected={selected} wall={time.time()-started:.1f}s',flush=True)
            finally:env.close()
            base.dump(out/'training-result.json',{'seed':seed,'method':args.method,'environment_steps':steps,'completed_episodes':completed,
                       'selected_step':selected,'validation_mean':best,'checkpoint_sha256':base.sha(out/'actor.pt'),
                       'positive_transitions':agent.positive_transitions,'transitions':agent.transitions,
                       'clipped_action_fraction':agent.clipped_actions/max(1,agent.action_values),
                       'actor_parameters':sum(p.numel() for p in agent.actor_local.parameters()),
                       'last_losses':agent.last_losses,'wall_seconds':time.time()-started})
        manifest.update(status='completed',finished_unix=time.time());base.dump(args.out/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed',failure_type=type(exc).__name__,failure=str(exc),finished_unix=time.time());base.dump(args.out/'manifest.json',manifest);raise

def final_test(args, protocol):
    root=args.out
    manifests={m:json.loads((root/m/'manifest.json').read_text()) for m in protocol['methods']}
    for m,d in manifests.items():
        if d['status']!='completed' or d['protocol']!=protocol:raise ValueError('Both frozen training campaigns must complete first')
        for name,digest in d['source_sha256'].items():
            if base.sha(ROOT/name)!=digest:raise ValueError('Executed training source changed before final tests')
    target=root/'final-test'
    if target.exists():raise ValueError('Final-test output already exists')
    target.mkdir();results=[]
    base.dump(target/'manifest.json',{'status':'started','protocol':protocol,'started_unix':time.time(),'training_manifest_sha256':{m:base.sha(root/m/'manifest.json') for m in protocol['methods']}})
    for seed in protocol['training_seeds']:
        agents={m:make_agent(m,seed,protocol) for m in protocol['methods']}
        for m,agent in agents.items():
            checkpoint=root/m/f'seed-{seed}/actor.pt'
            expected=json.loads((checkpoint.parent/'training-result.json').read_text())['checkpoint_sha256']
            if base.sha(checkpoint)!=expected:raise ValueError('Checkpoint changed')
            agent.actor_local.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        for offset in protocol['test_environment_seed_offsets']:
            for name,agent in [*agents.items(),('random',None),('zero',None)]:
                records=base.evaluate(args.env,seed+offset,args.worker,protocol['test_episodes_per_environment_seed'],protocol['max_steps_per_episode'],agent,name)
                row={'training_seed':seed,'environment_seed':seed+offset,'policy':name,'mean_score':float(np.mean([r['episode_score'] for r in records])),'truncated_episodes':sum(r['truncated'] for r in records),'episodes':records}
                results.append(row);base.dump(target/'results.json',results)
                print(f'FINAL seed={seed} env={seed+offset} {name} mean={row["mean_score"]:.4f}',flush=True)
    base.dump(target/'manifest.json',{'status':'completed','protocol':protocol,'finished_unix':time.time(),'training_manifest_sha256':{m:base.sha(root/m/'manifest.json') for m in protocol['methods']},'results_sha256':base.sha(target/'results.json')})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=['ddpg_exploration','td3_layernorm']);p.add_argument('--worker',type=int,default=70)
    p.add_argument('--final-test',action='store_true');a=p.parse_args();a.env=a.env.resolve();a.out=a.out.resolve()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol=json.loads((ROOT/'ROBUSTNESS_PROTOCOL.json').read_text())
    if a.final_test:final_test(a,protocol)
    elif a.method:train(a,protocol)
    else:p.error('Choose --method or --final-test')

if __name__=='__main__':main()
