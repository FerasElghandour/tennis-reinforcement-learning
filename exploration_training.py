"""One fixed sparse-reward exploration alternative, retaining the failed baseline."""
import argparse,csv,json,platform,random,time
from pathlib import Path
import numpy as np
import torch
from importlib import metadata
import fresh_training as base

class WarmupAgent(base.MADDPGAgent):
    def __init__(self,*args,warmup=2000,sigma=.3,**kwargs):
        super().__init__(*args,**kwargs);self.warmup=warmup;self.noise.sigma=sigma
        self.behavior_rng=np.random.default_rng(self.seed+17000);self.behavior_steps=0
        self.positive_transitions=0;self.transitions=0;self.clipped_actions=0;self.action_values=0
    def act(self,states,add_noise=True):
        if not add_noise:return super().act(states,False)
        self.behavior_steps+=1
        actions=(self.behavior_rng.uniform(-1,1,(self.num_agents,self.action_size)).astype(np.float32)
                 if self.behavior_steps<=self.warmup else super().act(states,True))
        self.clipped_actions+=int((np.abs(actions)>=.999).sum());self.action_values+=actions.size
        return actions
    def step(self,states,actions,rewards,next_states,dones):
        self.positive_transitions+=int((np.asarray(rewards)>0).sum());self.transitions+=len(rewards)
        return super().step(states,actions,rewards,next_states,dones)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--env',type=Path,required=True);p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((base.ROOT/'EXPLORATION_PROTOCOL.json').read_text());baseline=json.loads((a.baseline_run/'manifest.json').read_text())
    if baseline['status']!='completed' or a.out.exists():raise ValueError('Completed baseline campaign and new output directory required')
    a.env=a.env.resolve();a.out.mkdir(parents=True);torch.set_num_threads(1)
    manifest={'protocol':protocol,'baseline_manifest_sha256':base.sha(a.baseline_run/'manifest.json'),
              'source_sha256':{str(f.relative_to(base.ROOT)):base.sha(f) for f in [Path(__file__),base.ROOT/'fresh_training.py',base.ROOT/'EXPLORATION_PROTOCOL.json',*sorted((base.ROOT/'src/tennis_control').glob('*.py')),*sorted((base.ROOT/'vendor').rglob('*.py'))]},
              'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'grpcio':metadata.version('grpcio'),'protobuf':metadata.version('protobuf'),'started_unix':time.time(),'status':'started'}
    base.dump(a.out/'manifest.json',manifest);all_results=[]
    try:
        for seed in protocol['training_seeds']:
            out=a.out/f'seed-{seed}';out.mkdir();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            agent=WarmupAgent(24,2,2,seed,warmup=protocol['uniform_random_warmup_environment_steps'],sigma=protocol['ou_sigma_after_warmup'])
            env=base.environment(a.env,seed,30);best=-float('inf');selected=None;validations=[]
            try:
                brain=env.brain_names[0]
                with (out/'training.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=['episode','agent_0_return','agent_1_return','episode_score','steps','truncated']);writer.writeheader()
                    for ep in range(1,protocol['episodes_per_seed']+1):
                        row=dict(episode=ep,**base.rollout(env,brain,agent,training=True,max_steps=protocol['max_steps_per_episode']));writer.writerow(row);f.flush()
                        if ep%protocol['validation_every_episodes']==0:
                            val=base.evaluate(a.env,seed+protocol['validation_environment_seed_offset'],31,protocol['validation_episodes'],protocol['max_steps_per_episode'],agent)
                            if any(r['truncated'] for r in val):raise RuntimeError('Incomplete validation episode')
                            mean=float(np.mean([r['episode_score'] for r in val]));validations.append({'training_episode':ep,'mean_score':mean,'episodes':val})
                            if mean>best:best=mean;selected=ep;torch.save(agent.actor_local.state_dict(),out/'actor.pt')
                            base.dump(out/'validation.json',validations);print(f'seed {seed}, episode {ep}, validation {mean:.4f}, selected {selected}',flush=True)
            finally:env.close()
            agent.actor_local.load_state_dict(torch.load(out/'actor.pt',map_location='cpu',weights_only=True))
            frozen=base.MADDPGAgent(24,2,2,seed);baseline_checkpoint=a.baseline_run/f'seed-{seed}/actor.pt'
            baseline_result=json.loads((a.baseline_run/f'seed-{seed}/result.json').read_text())
            if base.sha(baseline_checkpoint)!=baseline_result['checkpoint_sha256']:raise ValueError('Baseline checkpoint changed')
            frozen.actor_local.load_state_dict(torch.load(baseline_checkpoint,map_location='cpu',weights_only=True))
            evaluation={}
            for name,policy in [('exploration',agent),('frozen_baseline',frozen),('random',None),('zero',None)]:
                records=base.evaluate(a.env,seed+protocol['test_environment_seed_offset'],31,protocol['test_episodes'],protocol['max_steps_per_episode'],policy,name)
                evaluation[name]={'mean_score':float(np.mean([r['episode_score'] for r in records])),'median_score':float(np.median([r['episode_score'] for r in records])),'truncated_episodes':sum(r['truncated'] for r in records),'episodes':records}
            result={'seed':seed,'selected_episode':selected,'validation_mean':best,'environment_steps':agent.behavior_steps,
                    'positive_training_transitions':agent.positive_transitions,'training_transitions':agent.transitions,
                    'clipped_training_action_fraction':agent.clipped_actions/agent.action_values,'checkpoint_sha256':base.sha(out/'actor.pt'),
                    'baseline_checkpoint_sha256':base.sha(baseline_checkpoint),'evaluation':evaluation}
            base.dump(out/'result.json',result);all_results.append(result);base.dump(a.out/'results.json',all_results)
        manifest.update(status='completed',finished_unix=time.time());base.dump(a.out/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed',failure=repr(exc),finished_unix=time.time());base.dump(a.out/'manifest.json',manifest);raise

if __name__=='__main__':main()
