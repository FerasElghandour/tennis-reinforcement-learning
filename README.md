# Shared-Policy Continuous Control

A two-agent DDPG implementation with saved policies, independently audited score records, reproducibility fixes and portable CPU inference.

**[Open the interactive report](https://feraselghandour.github.io/tennis-reinforcement-learning/)** · [Architecture](ARCHITECTURE.md) · [Methods and validation](VALIDATION.md)

[![Quality checks](https://github.com/FerasElghandour/tennis-reinforcement-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/FerasElghandour/tennis-reinforcement-learning/actions/workflows/ci.yml)

## What the evidence shows

The saved policy achieved a **2.067 mean score over 100 historical evaluation episodes**, with learning and exploration disabled. This is the evaluation mean, not the height of the training curve.

The training record contains 875 episodes. Metadata identifies the saved checkpoint at episode 634, where the rolling 100-episode training mean is 0.5338. Training continued afterward and regressed. The two records answer different questions: how learning progressed, and how the saved policy performed under evaluation.

| Measure | Result |
|---|---|
| Recorded training episodes | 875 |
| First qualifying 100-episode window | Ends at episode 607 |
| Saved checkpoint | Episode 634; rolling training mean 0.5338 |
| Saved-policy evaluation | 100 episodes; mean score 2.067 |
| Sequence-level block-bootstrap interval | 1.758–2.349; blocks of ten episodes |
| Policy interface | 24 input features → 2 bounded continuous actions |

The interactive report includes selectable comparisons, a matching numerical table, CSV/JSON downloads and the reproducibility record. It also works offline: download `docs/index.html` and open it in a browser.

[![Training rolling mean over 875 episodes with saved checkpoint 634 marked, alongside 100 evaluation scores with mean 2.067. Separate axes and scales.](https://feraselghandour.github.io/assets/tennis-results.svg)](https://feraselghandour.github.io/assets/tennis-results.svg)

Training uses a rolling mean; evaluation shows individual episode scores and their mean. The axes have different scales. [Training CSV](run-seed-0/training_scores.csv) · [Evaluation CSV](run-seed-0/evaluation_scores.csv) · [Checkpoint metadata](run-seed-0/training_result.json). Both sequences use the same episode-score definition. One historical training seed does not establish multi-seed reliability.

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

The score records are included. The audit runs without Torch or Unity. Install the optional agent dependencies for policy inference and export; see below.

Each run writes `index.html` and `metrics.json` to the chosen output directory. The JSON records the measured results, input SHA-256 fingerprints, source-code fingerprints, dependency versions and analytical settings. To use the notebooks, install the optional `notebooks` dependencies.

## Run the saved policy

```sh
python -m pip install -e ".[agent,dev]"
tennis-policy --checkpoint run-seed-0/checkpoint_actor.pth --observations observations.json
tennis-policy --checkpoint run-seed-0/checkpoint_actor.pth --export exports/actor.pt2
python verify_agent.py
```

`observations.json` must contain environment observations as a JSON array with shape `[batch, 24]`. The output has two actions per observation, each in `[-1, 1]`. The export includes interface metadata and source/artifact hashes. Exported programs are framework-version dependent; validate parity in the intended runtime.

For new environment runs, use the separate legacy Linux/Python 3.11 setup in [MODEL_CARD.md](MODEL_CARD.md). The environment binary is acquired separately.

## Interpretation

The critic uses each agent’s local observation and action. The legacy class name `MADDPGAgent` does not imply a centralized critic. Scores come from one historical training seed and one evaluation sequence. The block-bootstrap interval is conditional on that sequence, not a multi-seed performance guarantee. Engineering updates have not been presented as a new environment benchmark.

## Portfolio

[Model training, deep learning and autonomous systems](https://feraselghandour.github.io/#work).

---

**Feras Elghandour** · [GitHub](https://github.com/FerasElghandour)  
Data sources and applicable license notices: [DATA.md](DATA.md).

