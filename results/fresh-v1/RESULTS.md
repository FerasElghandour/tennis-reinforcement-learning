# Fresh Tennis training and evaluation

## Six independent fresh training runs

Two fixed protocols each train seeds 11/22/33 for 1,000 episodes on the M2 Max CPU. Every 100 episodes, 20 separate validation episodes select the highest validation mean (first tie). Final tests use 100 episodes per policy, with exploration and learning disabled. All selected policies and every raw rollout are included. The score is the maximum of the two undiscounted agent returns.

The baseline has sparse positive rewards and performs poorly. The exploratory alternative adds 2,000 uniform-action environment steps before returning to policy/OU actions with sigma 0.3; all other learning settings stay fixed. It was chosen after seeing baseline failure, then frozen before its new final test scores. Validation seeds are development inputs already seen in the baseline campaign.

### New final test family, environment seeds 950011 / 950022 / 950033

| Training seed | Exploration selected at | Exploration | Frozen baseline actor | Random | Zero |
|---|---:|---:|---:|---:|---:|
| 11 | 900 | 2.1878 | 0.0000 | 0.0191 | 0.0018 |
| 22 | 100 | 0.0000 | 0.0168 | 0.0226 | 0.0029 |
| 33 | 100 | 0.0000 | 0.0000 | 0.0173 | 0.0037 |

Mean across three independently trained policies: exploration **0.7293**, frozen baseline **0.0056**, random **0.0197**. Exploration exceeded its matched random baseline in **1/3** seed groups; its three means were 2.1878, 0.0000, 0.0000. Read every seed above; n=3 is descriptive and does not establish a general algorithm advantage or reliable learning. Shared initialization seeds do not force identical trajectories across policies. The test family was not used for checkpoint selection or subsequent tuning.

![All training seed curves and all new final-test policy means](training-and-test.png)

### Baseline's earlier final test family, environment seeds 900011 / 900022 / 900033

| Training seed | Selected at | Learned | Random | Zero |
|---|---:|---:|---:|---:|
| 11 | 100 | 0.0000 | 0.0125 | 0.0027 |
| 22 | 700 | 0.0120 | 0.0159 | 0.0047 |
| 33 | 100 | 0.0000 | 0.0143 | 0.0019 |

These tests are separate from the new comparison above. The baseline actor is frozen before evaluation on the new family. The reported learned result is not substituted with historical saved-policy scores.

## Step counts and exploration diagnostics

Both protocols receive 1,000 training episodes, but their step counts and compute differ. Longer rallies create more transitions/updates. This is not a compute-matched experiment or an isolated causal test of one exploration setting.

| Seed | Exploration training environment steps | Positive / all player transitions | Action values clipped near bound |
|---|---:|---:|---:|
| 11 | 47,995 | 1,789/95,990 | 45.743% |
| 22 | 14,817 | 32/29,634 | 44.253% |
| 33 | 14,752 | 32/29,504 | 43.923% |

Baseline reward-sparsity, rolling-window and episode-length diagnostics are in [baseline/diagnostics.json](baseline/diagnostics.json). Training curves show noisy exploration behavior, whereas final test scores use deterministic policy actions. These quantities must not be interchanged when discussing a rolling-100 training criterion. No mean test score alone establishes that training met that criterion.

## Historical saved actor: 300 new evaluation episodes

One fixed historically trained actor was evaluated with three new environment initialization seeds. It scored **2.1729** across 300 episodes, with no external-step-cap truncations. This is one trained policy over three new sequences, not three independent training runs or an exact reproduction of the historical evaluation sequence.

| Environment seed | Episodes | Mean score | Truncated |
|---|---:|---:|---:|
| 810001 | 100 | 2.0597 | 0 |
| 810002 | 100 | 2.3229 | 0 |
| 810003 | 100 | 2.1360 | 0 |

[Raw historical-policy scores, checkpoint fingerprint and protocol](historical-policy). The earlier 875-episode training record and 2.067 saved-policy evaluation remain unchanged in `run-seed-0/`.

## Reference runtime and source

New baseline/exploration reference: Python 3.11.15, Torch 2.5.1, NumPy 1.26.3, protobuf 3.20.3 and gRPC 1.84.0, CPU, one Torch thread. The official Intel Mac Tennis binary ran under Apple Rosetta on the M2 Max. Its SHA-256 is recorded in the baseline manifest; the binary is acquired separately and is not redistributed. [Pinned dependencies](../../requirements-reference-macos.txt) · [Observed environment](reference-environment.json).

Each campaign includes its exact executed source archive and source hashes. The historical-policy evaluation retains its older matching source separately. The client fixes use per-instance pipes, bounded response waits and clean server shutdown; upstream Apache 2.0 notices are preserved. Root MIT applies only to explicitly listed original tooling, not the simulator or third-party code.

An earlier gRPC 1.58.0 exploration attempt completed seed-11 training, then aborted natively before writing final test results. [Failure record and source](superseded-runtime). It is not counted as a completed three-seed experiment. Both primary protocols were rerun under the same newer transport reference. An earlier complete baseline is retained there for audit, not pooled as additional independent evidence.

## Run and audit

```sh
python3.11 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r requirements-reference-macos.txt
.venv-reference/bin/python -m pip install --no-deps -e .
.venv-reference/bin/python fresh_training.py --env /path/to/Tennis.app --out runs/baseline
.venv-reference/bin/python exploration_training.py --env /path/to/Tennis.app --baseline-run runs/baseline --out runs/exploration
.venv-reference/bin/python verify_fresh_results.py --results results/fresh-v1
```

Training output directories must be new; these runners do not offer mid-run continuation. Rollout, validation and final-test budgets are explicit in `FRESH_TRAINING_PROTOCOL.json` and `EXPLORATION_PROTOCOL.json`. The offline audit checks all score identities, budgets, validation-only first-maximum checkpoint selection, frozen comparison actors and artifact hashes. It does not rerun Unity or guarantee cross-platform determinism. [Audit result](verification.json).
