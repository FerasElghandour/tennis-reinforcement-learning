"""Run fixed-budget fresh Tennis seeds with independent validation and baselines."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
from importlib import metadata

import numpy as np
import torch
from maddpg_agent import MADDPGAgent

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def rollout(env,brain,agent=None,rng=None,training=False,max_steps=10000):
    info=env.reset(train_mode=True)[brain];states=info.vector_observations
    returns=np.zeros(len(info.agents));steps=0
    if agent is not None:agent.reset()
    for steps in range(1,max_steps+1):
        if agent is not None:actions=agent.act(states,add_noise=training)
        elif rng is not None:actions=rng.uniform(-1,1,(len(states),2)).astype(np.float32)
        else:actions=np.zeros((len(states),2),dtype=np.float32)
        info=env.step(actions)[brain]
        if training:agent.step(states,actions,info.rewards,info.vector_observations,info.local_done)
        returns+=info.rewards;states=info.vector_observations
        if np.any(info.local_done):break
    truncated=not np.any(info.local_done)
    return {'agent_0_return':float(returns[0]),'agent_1_return':float(returns[1]),'episode_score':float(returns.max()),'steps':steps,'truncated':bool(truncated)}

def environment(path,seed,worker):
    sys.path.insert(0,str(ROOT/'vendor'))
    from unityagents import UnityEnvironment
    return UnityEnvironment(file_name=str(path),seed=seed,worker_id=worker,no_graphics=True)

def evaluate(path,seed,worker,episodes,max_steps,agent=None,policy='learned'):
    python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state()
    env=None
    try:
        env=environment(path,seed,worker)
        brain=env.brain_names[0];rng=np.random.default_rng(seed+7) if policy=='random' else None
        # Independent environments make validation consume no training simulator
        # state. Restore exploration state because rollout resets OU noise.
        noise_state=None if agent is None else agent.noise.state.copy()
        try:return [dict(episode=i+1,**rollout(env,brain,agent,rng,False,max_steps)) for i in range(episodes)]
        finally:
            if agent is not None:agent.noise.state=noise_state
    finally:
        if env is not None:env.close()
        random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--env',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--probe',action='store_true');a=p.parse_args();a.env=a.env.resolve()
    if a.out.exists():raise ValueError('Choose a new output directory')
    executable=a.env/'Contents/MacOS/Tennis' if a.env.suffix=='.app' else a.env
    if not executable.is_file():raise ValueError('Official Tennis executable required')
    protocol=json.loads((ROOT/'FRESH_TRAINING_PROTOCOL.json').read_text());a.out.mkdir(parents=True)
    files=[Path(__file__),ROOT/'FRESH_TRAINING_PROTOCOL.json',*sorted((ROOT/'src/tennis_control').glob('*.py')),*sorted((ROOT/'vendor').rglob('*.py'))]
    manifest={'protocol':protocol,'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files},'environment_executable_sha256':sha(executable),
              'environment_path':str(a.env),'python':platform.python_version(),'platform':platform.platform(),'torch':str(torch.__version__),'numpy':np.__version__,'grpcio':metadata.version('grpcio'),'protobuf':metadata.version('protobuf'),'started_unix':time.time(),'status':'started'}
    dump(a.out/'manifest.json',manifest);torch.set_num_threads(1)
    try:
        env=environment(a.env,protocol['training_seeds'][0],10)
        try:
            brain=env.brain_names[0];info=env.reset(train_mode=True)[brain]
            if info.vector_observations.shape!=(2,24) or env.brains[brain].vector_action_space_size!=2:raise ValueError('Expected 2-agent/24-observation/2-action Tennis')
            env.step(np.zeros((2,2)))
        finally:env.close()
        if a.probe:manifest['status']='probe_passed';dump(a.out/'manifest.json',manifest);print('Official Tennis probe passed.',flush=True);return
        all_results=[]
        for seed in protocol['training_seeds']:
            out=a.out/f'seed-{seed}';out.mkdir();np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
            agent=MADDPGAgent(24,2,2,seed);scores=[];validations=[];best=-float('inf');selected=None
            env=environment(a.env,seed,10)
            try:
                brain=env.brain_names[0]
                with (out/'training.csv').open('w',newline='') as f:
                    fields=['episode','agent_0_return','agent_1_return','episode_score','steps','truncated'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                    for ep in range(1,protocol['episodes_per_seed']+1):
                        record=dict(episode=ep,**rollout(env,brain,agent,training=True,max_steps=protocol['max_steps_per_episode']));scores.append(record);writer.writerow(record);f.flush()
                        if ep%protocol['validation_every_episodes']==0:
                            validation=evaluate(a.env,seed+protocol['validation_environment_seed_offset'],11,protocol['validation_episodes'],protocol['max_steps_per_episode'],agent)
                            if any(r['truncated'] for r in validation):raise RuntimeError('Validation hit the step cap; no checkpoint selected from incomplete episodes')
                            mean=float(np.mean([r['episode_score'] for r in validation]));validations.append({'training_episode':ep,'mean_score':mean,'episodes':validation})
                            if mean>best:
                                best=mean;selected=ep;torch.save(agent.actor_local.state_dict(),out/'actor.pt')
                            dump(out/'validation.json',validations);print(f'seed {seed}, episode {ep}, validation {mean:.4f}, selected {selected}',flush=True)
            finally:env.close()
            agent.actor_local.load_state_dict(torch.load(out/'actor.pt',map_location='cpu',weights_only=True),strict=True)
            policies={}
            for name in protocol['evaluation_policies']:
                records=evaluate(a.env,seed+protocol['test_environment_seed_offset'],11,protocol['test_episodes'],protocol['max_steps_per_episode'],agent if name=='learned' else None,name)
                policies[name]={'mean_score':float(np.mean([r['episode_score'] for r in records])), 'median_score':float(np.median([r['episode_score'] for r in records])),'truncated_episodes':sum(r['truncated'] for r in records),'episodes':records}
            result={'seed':seed,'training_episodes':len(scores),'selected_episode':selected,'validation_mean':best,'checkpoint_sha256':sha(out/'actor.pt'),'evaluation':policies}
            dump(out/'result.json',result);all_results.append(result);dump(a.out/'results.json',all_results)
        manifest.update(status='completed',finished_unix=time.time());dump(a.out/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed',failure_type=type(exc).__name__,failure=str(exc),finished_unix=time.time());dump(a.out/'manifest.json',manifest);raise

if __name__=='__main__':main()
