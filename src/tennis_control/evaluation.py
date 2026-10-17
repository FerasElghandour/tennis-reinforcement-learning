"""Validate saved score records and audit evaluation independently of Unity."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .reporting import publish_report


def validate_scores(frame):
    required = {'episode', 'agent_0_return', 'agent_1_return', 'episode_score', 'truncated'}
    if missing := required - set(frame):
        raise ValueError(f'Missing score columns: {sorted(missing)}')
    data = frame.copy()
    if data.empty:
        raise ValueError('Score file is empty.')
    for col in ['episode', 'agent_0_return', 'agent_1_return', 'episode_score']:
        data[col] = pd.to_numeric(data[col], errors='raise')
    if not np.isfinite(data[['episode', 'agent_0_return', 'agent_1_return', 'episode_score']]).all().all():
        raise ValueError('Score records must be finite.')
    if not np.array_equal(data.episode.to_numpy(), np.arange(1, len(data)+1)):
        raise ValueError('Episodes must be consecutive, unique and start at one.')
    if not np.allclose(data.episode_score, data[['agent_0_return', 'agent_1_return']].max(axis=1), atol=1e-7, rtol=0):
        raise ValueError('Episode score must equal the maximum accumulated agent return.')
    flags = data.truncated.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if flags.isna().any():
        raise ValueError('Truncation flags must be boolean.')
    data['truncated'] = flags.astype(bool)
    return data


def full_window_scores(data, window=100):
    if window < 1:
        raise ValueError('Window must be positive.')
    # Any truncated episode invalidates its entire rolling window.
    completed = data.episode_score.mask(data.truncated)
    return completed.rolling(window, min_periods=window).mean()


def block_bootstrap_mean(values, block=10, draws=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if len(values) < block or block < 1 or not np.isfinite(values).all():
        raise ValueError('Finite evaluation values and at least one full block are required.')
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values)/block)))
        indices = np.concatenate([(start + np.arange(block)) % len(values) for start in starts])[:len(values)]
        means.append(float(values[indices].mean()))
    return np.quantile(means, [.025, .975]).tolist()


def summarize(training, evaluation):
    rolling = full_window_scores(training)
    solved = training.loc[rolling >= .5, 'episode']
    complete = evaluation.loc[~evaluation.truncated]
    if len(complete) < 10:
        raise ValueError('At least ten completed evaluation episodes are needed for the uncertainty audit.')
    interval = block_bootstrap_mean(complete.episode_score.to_numpy())
    return {'training': {'episodes': len(training), 'truncated': int(training.truncated.sum()),
                         'first_solved_episode': int(solved.iloc[0]) if len(solved) else None,
                         'best_full_window_mean': float(rolling.max()) if rolling.notna().any() else None},
            'evaluation': {'episodes': len(evaluation), 'completed_episodes': len(complete),
                           'mean': float(complete.episode_score.mean()), 'median': float(complete.episode_score.median()),
                           'std': float(complete.episode_score.std()), 'block_bootstrap_ci95': interval,
                           'criterion_met': bool(len(evaluation) >= 100 and not evaluation.truncated.any() and evaluation.episode_score.mean() >= .5)},
            'uncertainty': {'block_length': 10, 'draws': 2000, 'seed': 42,
                            'scope': 'Circular block bootstrap of one evaluation sequence; not variation across independent training seeds.'},
            'training_curve': [{'episode': int(r.episode), 'score': float(r.episode_score), 'mean_100': float(rolling.iloc[i]) if pd.notna(rolling.iloc[i]) else None} for i, r in enumerate(training.itertuples())],
            'evaluation_curve': [{'episode': int(r.episode), 'score': float(r.episode_score)} for r in complete.itertuples()]}


def report_spec(result):
    t, e = result['training'], result['evaluation']
    views = [{'label': 'Training: full 100-episode mean', 'kind': 'line', 'unit': 'episode score',
      'series': [{'name': '100-episode mean', 'points': [{'x': r['episode'], 'y': r['mean_100']} for r in result['training_curve'] if r['mean_100'] is not None]}]},
      {'label': 'Evaluation: episode returns', 'kind': 'line', 'unit': 'episode score',
       'series': [{'name': 'Saved policy, exploration disabled', 'points': [{'x': r['episode'], 'y': r['score']} for r in result['evaluation_curve']]}]},
      {'label': 'Evaluation: summary and interval', 'kind': 'bar', 'unit': 'episode score',
       'series': [{'name': 'Evaluation mean and 95% block-bootstrap interval', 'points': [{'x': 'Saved policy', 'y': e['mean'], 'low': e['block_bootstrap_ci95'][0], 'high': e['block_bootstrap_ci95'][1]}]}]}]
    return {'title': 'Learning a shared control policy', 'subtitle': 'Inspect the training record, verify the score definition and separate one-run uncertainty from repeatability.',
     'category': 'Reinforcement learning / continuous control', 'findings': [f'The first complete 100-episode window meeting the score criterion ends at episode {t["first_solved_episode"]}.',
      f'{e["completed_episodes"]} completed evaluation episodes have mean score {e["mean"]:.3f}.',
      f'The sequence-level block-bootstrap interval is {e["block_bootstrap_ci95"][0]:.3f} to {e["block_bootstrap_ci95"][1]:.3f}; this is not a multi-seed benchmark.'],
     'views': views, 'methodology': [{'title': 'Independent score audit', 'body': 'Validate consecutive episode IDs, finite agent returns, max-agent episode scores and truncation flags. No rolling window shorter than 100 episodes qualifies; any truncation invalidates its window.'},
      {'title': 'Saved policy evaluation', 'body': 'Evaluation records come from the saved policy with exploration and learning disabled. A circular block bootstrap uses blocks of ten episodes to retain local sequential structure.'},
      {'title': 'Decision boundary', 'body': 'These are historical simulation records from one training seed. No fresh Unity training or additional environment runs are claimed. The critic uses local observations and actions, not a centralized joint state.'}],
     'notes': 'The audit runs without the legacy Unity environment. Source score files and checkpoint weights remain inspectable in the repository.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('run-seed-0'))
    parser.add_argument('--output', type=Path, default=Path('reports'))
    args = parser.parse_args()
    paths = [args.run/'training_scores.csv', args.run/'evaluation_scores.csv']
    try:
        result = summarize(*(validate_scores(pd.read_csv(p)) for p in paths))
        publish_report(args.output, result, report_spec(result), paths)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Evaluation audit failed: {exc}\n')


if __name__ == '__main__':
    main()
