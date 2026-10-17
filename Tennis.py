"""Train/evaluate shared-policy DDPG on Udacity's supplied Tennis binary.

Every score is measured from the live environment. No historic scores or
recovered weights are used. See --help for probe, training and evaluation.
"""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import argparse
import random
import csv
import json
import platform
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from unityagents import UnityEnvironment
from maddpg_agent import MADDPGAgent


def environment_path(explicit):
    if explicit:
        return explicit
    candidates = ['/data/Tennis_Linux_NoVis/Tennis',
                  '../Tennis_Linux_NoVis/Tennis.x86_64',
                  'Tennis_Linux_NoVis/Tennis.x86_64',
                  'Tennis_Linux/Tennis.x86_64', 'Tennis.app']
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError('Supply --env with the Udacity Tennis executable.')


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def plot_scores(out, scores):
    rolling = np.full(len(scores), np.nan)
    if len(scores) >= 100:
        rolling[99:] = np.convolve(scores, np.ones(100)/100, mode='valid')
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(scores)+1)
    ax.plot(episodes, scores, alpha=.3, label='Episode max agent return')
    ax.plot(episodes, rolling, label='Full 100-episode mean')
    ax.axhline(.5, color='red', ls='--', label='Required threshold: 0.5')
    ax.set(xlabel='Completed episode', ylabel='Undiscounted return',
           title='Measured training rewards: DDPG shared-policy Tennis')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'scores.png', dpi=150)
    plt.close(fig)


def episode(env, brain, agent, training, max_steps):
    info = env.reset(train_mode=True)[brain]
    states = info.vector_observations
    returns = np.zeros(len(info.agents))
    agent.reset()
    for step in range(1, max_steps+1):
        actions = agent.act(states, add_noise=training)
        info = env.step(actions)[brain]
        if training:
            agent.step(states, actions, info.rewards,
                       info.vector_observations, info.local_done)
        returns += info.rewards
        states = info.vector_observations
        if np.any(info.local_done):
            return returns, step, False
    return returns, max_steps, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--mode', choices=['probe', 'train', 'evaluate'], default='train')
    parser.add_argument('--out', default='run-seed-0')
    parser.add_argument('--episodes', type=int, default=6000)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=float, default=0.75)
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = UnityEnvironment(file_name=environment_path(args.env),
                           worker_id=args.worker_id, seed=args.seed)
    started = time.time()
    try:
        brain = env.brain_names[0]
        info = env.reset(train_mode=True)[brain]
        state_size = info.vector_observations.shape[1]
        action_size = env.brains[brain].vector_action_space_size
        metadata = dict(seed=args.seed, state_size=state_size,
                        action_size=action_size, agents=len(info.agents),
                        torch=torch.__version__, numpy=np.__version__,
                        python=platform.python_version(),
                        environment=environment_path(args.env))
        atomic_json(out/'environment.json', metadata)
        print('ENVIRONMENT', json.dumps(metadata), flush=True)
        if args.mode == 'probe':
            info = env.step(np.zeros((len(info.agents), action_size)))[brain]
            print('PROBE_OK', info.vector_observations.shape, info.rewards, flush=True)
            return
        agent = MADDPGAgent(state_size, action_size, len(info.agents), args.seed)
        scores = []
        first_solved = None
        best = -float('inf')
        if args.mode == 'train':
            if args.resume:
                checkpoint = torch.load(out/'resume.pth', map_location='cpu', weights_only=False)
                for key in ['actor_local', 'actor_target', 'critic_local', 'critic_target']:
                    getattr(agent, key).load_state_dict(checkpoint[key])
                agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                agent.memory.memory = checkpoint['replay']
                agent.t_step = checkpoint['t_step']
                np.random.set_state(checkpoint['numpy_rng'])
                scores = checkpoint['scores']
                best = checkpoint['best']
                if 'python_rng' in checkpoint:
                    random.setstate(checkpoint['python_rng'])
                if 'torch_rng' in checkpoint:
                    torch.set_rng_state(checkpoint['torch_rng'])
                # Remove rows written after the last recovery checkpoint.
                csv_path = out/'training_scores.csv'
                rows = list(csv.DictReader(csv_path.open()))[:len(scores)]
                with csv_path.open('w', newline='') as recovered:
                    w = csv.DictWriter(recovered, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            csv_path = out/'training_scores.csv'
            with csv_path.open('a' if args.resume else 'w', newline='') as f:
                writer = csv.writer(f)
                if not args.resume:
                    writer.writerow(['episode', 'agent_0_return', 'agent_1_return',
                                     'episode_score', 'mean_100', 'steps', 'truncated'])
                for number in range(len(scores)+1, args.episodes+1):
                    returns, steps, truncated = episode(env, brain, agent, True, args.max_steps)
                    score = float(returns.max())
                    scores.append(score)
                    average = float(np.mean(scores[-100:])) if number >= 100 else None
                    writer.writerow([number, *returns.tolist(), score, average, steps, truncated])
                    f.flush()
                    if average is not None and not truncated and average > best:
                        best = average
                        torch.save(agent.actor_local.state_dict(), out/'checkpoint_actor.pth')
                        torch.save(agent.critic_local.state_dict(), out/'checkpoint_critic.pth')
                        atomic_json(out/'checkpoint.json', dict(episode=number, mean_100=best))
                    if number % 25 == 0:
                        status = dict(episode=number, mean_100=average, best_mean_100=best if np.isfinite(best) else None,
                                      seconds=round(time.time()-started, 1), solved=False)
                        atomic_json(out/'status.json', status)
                        print('PROGRESS', json.dumps(status), flush=True)
                    if number % 100 == 0:
                        plot_scores(out, scores)
                        torch.save(dict(**{k: getattr(agent,k).state_dict() for k in
                            ['actor_local','actor_target','critic_local','critic_target',
                             'actor_optimizer','critic_optimizer']}, replay=agent.memory.memory,
                            t_step=agent.t_step, scores=scores, best=best,
                            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                            torch_rng=torch.get_rng_state()), out/'resume.pth')
                    if average is not None and average >= .5 and first_solved is None and not truncated:
                        first_solved = number
                    if average is not None and average >= args.target and not truncated:
                        break
            plot_scores(out, scores)
            atomic_json(out/'training_result.json', dict(solved=first_solved is not None,
                        first_solved_episode=first_solved, completed_episodes=len(scores),
                        best_mean_100=best, elapsed_seconds=time.time()-started))
            if first_solved is None:
                print('TRAINING_NOT_SOLVED', flush=True)
                return
        agent.actor_local.load_state_dict(torch.load(out/'checkpoint_actor.pth',
                                                    map_location='cpu', weights_only=True))
        evaluation = []
        with (out/'evaluation_scores.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode','agent_0_return','agent_1_return','episode_score','steps','truncated'])
            for number in range(1, args.eval_episodes+1):
                returns, steps, truncated = episode(env, brain, agent, False, args.max_steps)
                evaluation.append(float(returns.max()))
                writer.writerow([number,*returns.tolist(),evaluation[-1],steps,truncated])
                f.flush()
                if number % 25 == 0:
                    print('EVALUATION', number, float(np.mean(evaluation)), flush=True)
        result = dict(episodes=len(evaluation), mean_score=float(np.mean(evaluation)),
                      threshold_met=bool(len(evaluation)>=100 and np.mean(evaluation)>=.5),
                      exploration=False)
        atomic_json(out/'evaluation_result.json', result)
        print('RESULT', json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
