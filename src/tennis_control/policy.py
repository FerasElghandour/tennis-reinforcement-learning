"""Portable CPU inference and graph export for the saved actor policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .model import Actor


def load_policy(checkpoint):
    actor = Actor(24, 2, 0)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def predict(actor, observations):
    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 24 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Observations must be a nonempty finite array with shape [batch, 24].')
    with torch.inference_mode():
        actions = actor(torch.from_numpy(values)).numpy()
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1:
        raise ValueError('Policy returned invalid actions.')
    return actions.tolist()


def export_policy(checkpoint, output):
    actor = load_policy(checkpoint)
    example = torch.zeros(2, 24)
    with torch.no_grad():
        exported = torch.export.export(actor, (example,), dynamic_shapes={
            'state': {0: torch.export.Dim('batch', min=1, max=1024)}}, strict=True)
        torch.testing.assert_close(exported.module()(example), actor(example))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(output))
    metadata = {'input': {'shape': ['batch', 24], 'batch_range': [1, 1024], 'dtype': 'float32'},
                'output': {'shape': ['batch', 2], 'range': [-1, 1]},
                'source_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                'artifact_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'torch': torch.__version__, 'environment': 'Legacy Tennis stacked local observations',
                'validation': 'Export/inference parity; no new environment-performance claim.'}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('run-seed-0/checkpoint_actor.pth'))
    parser.add_argument('--observations', type=Path, help='JSON array [batch,24] of environment observations')
    parser.add_argument('--export', type=Path, help='Output torch.export .pt2 path')
    args = parser.parse_args()
    if not args.observations and not args.export:
        parser.error('Supply --observations or --export.')
    try:
        if args.export:
            print(json.dumps(export_policy(args.checkpoint, args.export), indent=2))
        if args.observations:
            print(json.dumps({'actions': predict(load_policy(args.checkpoint), json.loads(args.observations.read_text()))}))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'Policy operation failed: {exc}\n')


if __name__ == '__main__':
    main()
