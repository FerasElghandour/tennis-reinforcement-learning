# Equal-interaction validation extension

37 tests pass. New TD3 numerical oracles protect the actual learning loss, clipped minimum-Q target, terminal masking, target-action smoothing, delayed actor/target updates, Polyak transfers and detached targets. An isolated min-to-max mutation is rejected by two tests. Synthetic bookkeeping tests reject changed checkpoint selection, NaN scores and missing final evaluation families.

The new result audit covers ten independent trainings (five per method, 40,000 environment steps each), all validation selections and all 2,000 final episodes including controls. The unit of replication is five trained policies per method. Training source and settings were frozen before these runs; final tests opened after both campaigns completed. No final score was used to choose checkpoints or tune parameters. The comparison includes architecture/noise/optimizer differences and remains descriptive.

[Every result and limit](results/robust-v2/RESULTS.md) · [Audit](results/robust-v2/verification.json).

## Preserved earlier validation

# Methods and validation

The 28-test suite covers score identities, truncation-aware windows, block resampling, replay ownership, independent random generators, invalid transitions, mode-preserving inference and exported-policy parity. New numerical oracles check hand-computed DDPG targets, terminal masking, actor gradients, Polyak updates and BatchNorm buffers. Legacy-client checks protect per-instance pipes, bounded waits and complete shutdown. Exploration fixtures check seeded warmup and preservation of the training budget during evaluation.

```sh
python -m pip install -e ".[agent,dev]"
python -m pytest -q
python verify_fresh_results.py --results results/fresh-v1
```

Fixtures isolate correctness; they are not performance benchmarks. The [fresh campaign](results/fresh-v1/RESULTS.md) separately reports six complete training runs, validation-only selection, held-out policy/baseline comparisons and all raw records. Its offline audit checks scores, counts, checkpoint selection and actor fingerprints. The historical saved actor's 300 new episodes evaluate one policy, not three training seeds.

The original interactive report remains generated from the original historical data and records implementation/input hashes in `docs/metrics.json`. It is a historical view, while the new campaign has its own result page. Browser checks of that existing report cover desktop/mobile tables and downloads.

The fresh Mac Unity reference uses Python 3.11.15, Torch 2.5.1, NumPy 1.26.3, protobuf 3.20.3, gRPC 1.84.0 and the official Intel simulator under Rosetta. Exact source archives and dependency snapshots accompany the results. An earlier native transport failure is retained explicitly. The critic uses local state/action observations, not centralized joint observations. Simulation results do not establish real-world control safety or cross-platform determinism.
