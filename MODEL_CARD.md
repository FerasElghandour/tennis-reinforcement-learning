# Saved control policies

## Intended use

Simulation-based control experiments and evaluation in the legacy two-agent Tennis environment. This policy is not validated for real-world control or other environments.

## Historical and released DDPG interface

- 24 continuous input features per agent: three stacked local observations.
- Two continuous actions per agent, bounded to `[-1, 1]`.
- Shared actor and critic; the critic receives local state/action inputs.
- Actor architecture: 24 → 256 (BatchNorm/ReLU) → 128 (ReLU) → 2 (tanh).



## Equal-interaction v2 policies

The new campaign contains five DDPG and five TD3 selected actors. The TD3 actor is 24→256 (LayerNorm/ReLU)→128 (ReLU)→2 (tanh), trained with two local critics. New CLI `robust_policy.py` requires the matching method and a two-player `[2,24]` input. It loads portable CPU state dictionaries using `weights_only=True`; the included zero-input demos are interface examples only. Existing export-parity checks below describe the DDPG path, not a newly tested TD3 graph export.

Validation alone selects each actor. Final evaluation contains 100 episodes per trained policy across two new environment initializations; five training seeds are the replication unit. [All seed returns, training diagnostics and limits](results/robust-v2/RESULTS.md). Eligible original code licensing does not extend the scoped grant to simulator binaries, imported source or checkpoint rights beyond their existing terms.

## Recorded evaluation

The included records contain 875 training episodes and a separate 100-episode evaluation. The first qualifying training window ends at episode 607. Saved-policy evaluation has mean score 2.067 with learning and exploration disabled. These are one-seed historical records, not a fresh performance measurement after the engineering updates.

The independent audit recomputes score identities and truncation-aware windows. Its circular block-bootstrap interval describes the saved evaluation sequence. It does not estimate variance across independently trained policies.

## New local validation

The historical actor completed 300 fresh episodes over three new environment initialization seeds, mean 2.1729. This is one fixed historically trained policy. Six additional independent training runs compare the released baseline with one exploratory warmup/noise alternative under separate validation and final tests. [Complete results and limits](results/fresh-v1/RESULTS.md).

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

Use `--mode train --out run-new-training` for a separate training run. The optional historical training target of 0.75 was not reached. The new Mac/Rosetta campaign and reference commands are recorded separately in `results/fresh-v1/RESULTS.md`. The historical Linux setup above is not the newly executed Mac reference.
