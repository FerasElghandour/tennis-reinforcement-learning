"""Load a v2 CPU actor for bounded inference; no simulator is needed."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from robust_training import make_agent,ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['ddpg_exploration','td3_layernorm'],required=True)
    p.add_argument('--actor',type=Path,required=True);p.add_argument('--observations',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    protocol=json.loads((ROOT/'ROBUSTNESS_PROTOCOL.json').read_text())
    states=np.asarray(json.loads(a.observations.read_text()) if a.observations else np.zeros((2,24)),dtype=np.float32)
    agent=make_agent(a.method,0,protocol)
    agent.actor_local.load_state_dict(torch.load(a.actor,map_location='cpu',weights_only=True),strict=True)
    actions=agent.act(states,False)
    print(json.dumps({'method':a.method,'input':'provided observations' if a.observations else 'generated zero observations; demonstration only',
                      'observation_shape':list(states.shape),'actions':actions.tolist(),'bounded':bool(np.max(np.abs(actions))<=1.)},indent=2))

if __name__=='__main__':main()
