# Saved control policy

## Intended use

Simulation-based control experiments and evaluation in the legacy two-agent Tennis environment. This policy is not validated for real-world control or other environments.

## Interface

- 24 continuous input features per agent: three stacked local observations.
- Two continuous actions per agent, bounded to `[-1, 1]`.
- Shared actor and critic; the critic receives local state/action inputs.
- Actor architecture: 24 → 256 (BatchNorm/ReLU) → 128 (ReLU) → 2 (tanh).

## Recorded evaluation

The included records contain 875 training episodes and a separate 100-episode evaluation. The first qualifying training window ends at episode 607. Saved-policy evaluation has mean score 2.067 with learning and exploration disabled. These are one-seed historical records, not a fresh performance measurement after the engineering updates.

The independent audit recomputes score identities and truncation-aware windows. Its circular block-bootstrap interval describes the saved evaluation sequence. It does not estimate variance across independently trained policies.

## Inference and export validation

Tests load the actual included actor checkpoint, check batches of 1, 2 and 5, verify finite bounded actions, and compare an exported graph against the actor on new test observations. Export batch bounds are 1–1,024. An interface/hash manifest accompanies each export.

## Reproducibility changes

Exploration and replay now use independent random generators, and replay stores owned copies. Future runs can therefore follow different random sequences from the historical run. The checkpoint and source score records are preserved; no new score improvement is attributed to these changes.

## Legacy environment setup

Use Linux and Python 3.11. Acquire the environment separately as described in DATA.md.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python Tennis.py --mode probe --env Tennis_Linux_NoVis/Tennis.x86_64
cp -R run-seed-0 run-local-evaluation
python Tennis.py --mode evaluate --env Tennis_Linux_NoVis/Tennis.x86_64 --out run-local-evaluation --eval-episodes 100
```

Use `--mode train --out run-new-training` for a separate training run. The optional historical training target of 0.75 was not reached. The environment was not rerun as part of the current code improvements.
