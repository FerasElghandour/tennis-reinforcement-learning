# Report: shared-policy DDPG for Tennis

## Algorithm

Both agents apply one deterministic actor to their local 24-dimensional stacked observation. Their experiences are stored as separate `(state, action, reward, next_state, done)` records in a shared replay buffer. A shared critic estimates Q for each local state/action pair. This is shared-policy DDPG self-play, not centralized-critic MADDPG.

The critic minimizes squared temporal-difference error with target
`r + gamma * (1 - done) * Q_target(next_state, actor_target(next_state))`.
Targets are computed without gradients. Target networks operate in evaluation mode, and their BatchNorm buffers follow the local networks. Actor learning maximizes the local critic’s predicted Q. Critic parameters are frozen during actor optimization. Target parameters are soft-updated after each learning update. OU noise is used only during training, and actions are clipped to [-1, 1].

## Architecture

Actor: 24 → 256 (BatchNorm/ReLU) → 128 (ReLU) → 2 (tanh).
Critic: 24 → 256 (BatchNorm/ReLU), concatenate 2 actions → 128 (ReLU) → 1 (linear).
Hidden weights use fan-in uniform initialization; output weights use uniform ±0.003.

## Hyperparameters

| Parameter | Value |
|---|---|
| Seed | 0 |
| Replay capacity | 1,000,000 |
| Batch size | 128 |
| Discount gamma | 0.99 |
| Target tau | 0.01 |
| Actor Adam learning rate | 0.001 |
| Critic Adam learning rate | 0.001 |
| Critic weight decay | 0 |
| Critic gradient clipping | norm 1.0 |
| Learning schedule | 2 updates every 2 environment steps |
| OU theta / sigma | 0.15 / 0.10 |
| Episode safety cap | 10,000 steps; truncations are logged |

## Measured results

The final CSV contains 875 completed training episodes with no safety-cap truncations. The first 100-episode window meeting the 0.5 criterion ends at **episode 607**, with mean **0.510000** (episodes 508–607). Thus the environment was solved in 607 training episodes. Each episode score is the maximum of the two agents’ accumulated undiscounted returns.

The submitted actor and critic are the best training checkpoint, from **episode 634**, whose preceding 100-episode training mean is **0.533800**. Training was resumed from the recovery checkpoint at episode 600 and stopped after episode 875. The optional 0.75 training target was not reached; the best checkpoint was retained because subsequent exploration reduced training returns.

A separate evaluation loaded that saved actor, disabled exploration and learning, and completed **100 consecutive episodes** with mean score **2.067000**. Raw evaluation returns and metadata are included. These results are from one seed and one evaluation sequence; they do not establish repeatability across seeds.

![Actual training rewards and complete 100-episode means](run-seed-0/scores.png)

The plot is regenerated from all 875 rows in `run-seed-0/training_scores.csv`. Windows shorter than 100 episodes are excluded. `run-seed-0/evaluation_scores.csv` contains the independent evaluation. The historical recovered plot and weights are excluded.

## Future work

1. Compare several independent random seeds; one run does not establish robust repeatability.
2. Try centralized critics conditioned on both agents’ states/actions to address changing partner behavior.
3. Test lower actor learning rates and exploration-noise schedules for stability.
4. Compare prioritized replay against uniform sampling.
5. Compare entropy-regularized SAC to deterministic DDPG.

## References and provenance

The new code revises the recovered submission and uses Udacity’s supplied headless Tennis binary and legacy client. The shared-actor/shared-replay approach follows the course’s benchmark hint. DDPG: https://arxiv.org/abs/1509.02971. MADDPG: https://arxiv.org/abs/1706.02275. Historical recovered checkpoints are not reused.
