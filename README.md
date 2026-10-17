# Shared-Policy Continuous Control

Train a shared continuous-control policy for two Tennis players, with a local critic, replay learning and portable saved-policy inference.

**[Open the interactive report](https://feraselghandour.github.io/tennis-reinforcement-learning/)** · [Architecture](ARCHITECTURE.md) · [Methods and validation](VALIDATION.md)

[![Quality checks](https://github.com/FerasElghandour/tennis-reinforcement-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/FerasElghandour/tennis-reinforcement-learning/actions/workflows/ci.yml)

## How training works

Both players share one actor, one local critic and one replay buffer. The actor maps each player’s 24-value observation to two bounded actions. Sampled transitions train the critic against target-network estimates, then update the actor to maximize the critic’s estimate. Targets follow soft updates with τ = 0.01.

<picture>
  <source media="(max-width: 760px)" srcset="assets/shared-policy-mechanism-mobile.svg">
  <img src="assets/shared-policy-mechanism.svg" alt="Two Tennis players share an actor and local critic. Their transitions enter a shared replay buffer; minibatches update critic and actor, followed by soft target updates.">
</picture>

[Actor and critic code](src/tennis_control/model.py) · [Replay and learning updates](src/tennis_control/maddpg_agent.py). The critic uses a single player’s observation/action pair; it is not centralized over both players.

## Try the saved policy on CPU

Use Python 3.11 or later:

```sh
git clone https://github.com/FerasElghandour/tennis-reinforcement-learning.git
cd tennis-reinforcement-learning
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[agent]"
python examples/policy_demo.py
```

The supplied [synthetic observation fixture](examples/observations.json) has shape `[2, 24]`: one zero vector and one fixed signed-value vector. The saved actor returns two bounded actions for each row. Example action output:

```json
{"actions": [[-0.69705, -0.982914], [-0.621963, -0.939908]]}
```

[Complete observed output](examples/policy_demo-output.json) · [Example environment](examples/environment.json). Floating-point values may vary slightly across versions. These inputs demonstrate the interface; they are not recorded environment states, a Unity rollout or a performance evaluation.

To use your own valid observation array:

```sh
tennis-policy --checkpoint run-seed-0/checkpoint_actor.pth --observations examples/observations.json
```

## Try a learning update

```sh
python examples/learn_step.py
```

This generates transitions for two players and exercises the implemented replay, critic, actor and target-update path on CPU. The output reports **128 replay transitions**, `actor_weights_changed: true`, and finite actor/critic losses. [Observed output](examples/learn_step-output.json).

This is a synthetic optimization example. Training or evaluating actual Tennis behavior requires the separately acquired legacy Unity environment and the setup in [MODEL_CARD.md](MODEL_CARD.md).

## Historical training and evaluation

The recorded training sequence spans 875 episodes. Metadata identifies checkpoint 634, where the rolling 100-episode training mean is 0.5338. That saved policy has a **2.067 mean score over 100 evaluation episodes**, with learning and exploration disabled. Training continued afterward and regressed.

[![Training rolling mean with checkpoint 634 marked, alongside saved-policy evaluation scores and their 2.067 mean. Separate scales.](https://feraselghandour.github.io/assets/tennis-results.svg)](https://feraselghandour.github.io/assets/tennis-results.svg)

Left: rolling training mean. Right: individual evaluation scores and their mean. [Training CSV](run-seed-0/training_scores.csv) · [Evaluation CSV](run-seed-0/evaluation_scores.csv) · [Checkpoint metadata](run-seed-0/training_result.json).

The first qualifying 100-episode training window ends at episode 607. The evaluation’s sequence-level block-bootstrap interval is 1.758–2.349 using blocks of ten episodes. These records describe one historical training seed and one evaluation sequence, not multi-seed reliability. Both use the same episode-score definition.

The [interactive report](https://feraselghandour.github.io/tennis-reinforcement-learning/) includes selectable views, a numerical table and CSV/JSON downloads. Download `docs/index.html` to inspect it offline.

## Engineering

- Strict transition and observation validation before replay mutation.
- Independent random generators for exploration noise and replay sampling; replay stores defensive copies.
- Inference preserves the model’s train/evaluation mode; learning records finite actor/critic losses.
- A separate score-audit CLI and CPU policy inference/export CLI, without requiring Unity for those tasks.

The implementation is packaged under `src/tennis_control/`; `Report.md` contains the algorithm and historical experiment detail. Small root-level entry points preserve existing commands.

## Reproduce

Use Python 3.11 or later and a separate virtual environment for this repository.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m tennis_control --run run-seed-0 --output reports
python -m pytest -q
```

The score records are included. The audit runs without Torch or Unity; the CPU examples above require the agent dependencies.

Each run writes `index.html` and `metrics.json` to the chosen output directory. The JSON records the measured results, input SHA-256 fingerprints, source-code fingerprints, dependency versions and analytical settings. To use the notebooks, install the optional `notebooks` dependencies.

## Export the saved actor

```sh
tennis-policy --checkpoint run-seed-0/checkpoint_actor.pth --export exports/actor.pt2
```

The export includes interface metadata and artifact hashes. Exported programs depend on the framework version; validate parity in the intended runtime.

## Interpretation

The critic uses each agent’s local observation and action. The legacy class name `MADDPGAgent` does not imply a centralized critic. Scores come from one historical training seed and one evaluation sequence. The block-bootstrap interval is conditional on that sequence, not a multi-seed performance guarantee. Engineering updates have not been presented as a new environment benchmark.

## Portfolio

[Model training, deep learning and autonomous systems](https://feraselghandour.github.io/#work).

---

**Feras Elghandour** · [GitHub](https://github.com/FerasElghandour)  
Data sources and applicable license notices: [DATA.md](DATA.md).


